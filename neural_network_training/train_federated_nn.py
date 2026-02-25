#!/usr/bin/env python3
"""Federated learning training with a simple neural network using Flower (FedAvg).

Example:
  python train_federated_nn.py --partition ML-EdgeIIoT-FEDT/dominant_client --num-rounds 40 --num-clients 10 --epochs 5 --batch-size 64 --learning-rate 0.001

O que esta versão adiciona:
- Run folder incremental por dataset: <output_dir>/<DATASET>-<RUN_ID>/<partition_tag>/client-id-*/nn_client-id-*.json
- Registry (_nodeid_to_clientidx.json) por execução (não persiste entre reruns do mesmo dataset)
- Pré-processamento agnóstico (get_dummies) + schema global de colunas para manter input_dim igual em todos os clientes
- Label mapping global (classes consistentes mesmo se algum cliente não tiver todas as classes)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import fcntl  # Linux file locks (process-safe)
import flwr as fl
from flwr.common import Context
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PARTITIONS_ROOT = "/home/yuri/FEDT_IDS2/partitions"
DEFAULT_OUTPUT_DIR = "/home/yuri/FEDT_IDS2/neural_network_training/results"


# -------------------------
# Data structures / model
# -------------------------
@dataclass
class ClientData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------
# Partition helpers
# -------------------------
def _resolve_partition_path(partition: str) -> str:
    if os.path.isabs(partition):
        return partition
    return os.path.join(PARTITIONS_ROOT, partition)


def _client_dir_sort_key(dir_name: str) -> Tuple[int, str]:
    """Sort client_0, client_1, ..., client_10 by numeric suffix."""
    m = re.fullmatch(r"client_(\d+)", dir_name)
    if m:
        return (int(m.group(1)), dir_name)
    return (10**12, dir_name)


def _find_client_dirs(partition_path: str, num_clients: int) -> List[str]:
    if not os.path.isdir(partition_path):
        raise FileNotFoundError(f"Partition path not found: {partition_path}")

    dir_names = [
        d
        for d in os.listdir(partition_path)
        if os.path.isdir(os.path.join(partition_path, d)) and d.startswith("client_")
    ]
    dir_names.sort(key=_client_dir_sort_key)

    client_dirs = [os.path.join(partition_path, d) for d in dir_names]

    if not client_dirs:
        raise FileNotFoundError(
            f"No client folders found in {partition_path}. Expected folders like client_0, client_1..."
        )

    if num_clients > len(client_dirs):
        raise ValueError(
            f"Requested {num_clients} clients, but only {len(client_dirs)} available in {partition_path}."
        )

    return client_dirs[:num_clients]


# -------------------------
# Dataset/run naming
# -------------------------
def _dataset_and_partition_tag(partition_path: str) -> Tuple[str, str]:
    """
    Para /partitions/<DATASET>/<SPLIT> -> (DATASET, SPLIT)
    Para caminhos fora de PARTITIONS_ROOT -> usa basename como dataset/tag.
    """
    rel = os.path.relpath(partition_path, PARTITIONS_ROOT)
    if rel.startswith(".."):
        base = os.path.basename(os.path.normpath(partition_path))
        return base, base

    parts = rel.split(os.sep)
    dataset_name = parts[0]
    partition_tag = os.path.join(*parts[1:]) if len(parts) > 1 else parts[0]
    partition_tag = partition_tag if partition_tag else "partition"
    return dataset_name, partition_tag


def _allocate_run_root(output_dir: str, dataset_name: str) -> Tuple[str, int]:
    """
    Cria um diretório novo: <output_dir>/<dataset_name>-<next_id>
    next_id = max(ids existentes) + 1, ou 0 se não existir.
    """
    os.makedirs(output_dir, exist_ok=True)
    pat = re.compile(rf"^{re.escape(dataset_name)}-(\d+)$")

    ids: List[int] = []
    for name in os.listdir(output_dir):
        p = os.path.join(output_dir, name)
        if not os.path.isdir(p):
            continue
        m = pat.match(name)
        if m:
            ids.append(int(m.group(1)))

    run_id = (max(ids) + 1) if ids else 0
    run_root = os.path.join(output_dir, f"{dataset_name}-{run_id}")
    os.makedirs(run_root, exist_ok=False)
    return run_root, run_id


def _write_run_meta(run_root: str, dataset_name: str, partition_path: str, partition_tag: str, args: argparse.Namespace) -> None:
    meta = {
        "dataset_name": dataset_name,
        "partition_path": partition_path,
        "partition_tag": partition_tag,
        "created_at_unix": time.time(),
        "args": {
            "num_rounds": int(args.num_rounds),
            "num_clients": int(args.num_clients),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
        },
    }
    path = os.path.join(run_root, "_run_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# -------------------------
# CSV loading / preprocessing
# -------------------------
def _infer_label_column(df: pd.DataFrame) -> str:
    preferred = ["label", "target", "class", "y"]
    for col in preferred:
        if col in df.columns:
            return col
    return df.columns[-1]


def _read_client_dfs(client_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    train_path = os.path.join(client_dir, "train.csv")
    test_path = os.path.join(client_dir, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing train.csv or test.csv in {client_dir}.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    label_col = _infer_label_column(train_df)
    return train_df, test_df, label_col


def _infer_global_schema(client_dirs: List[str]) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Cria:
      - feature_columns: união das colunas após get_dummies (ordem determinística)
      - label_classes: união das classes (strings) (ordem determinística)
      - label_to_index: mapeamento global label->id
    """
    feature_set: set[str] = set()
    label_set: set[str] = set()

    for cd in client_dirs:
        train_df, test_df, label_col = _read_client_dfs(cd)

        y_train = train_df[label_col].astype(str)
        y_test = test_df[label_col].astype(str)
        label_set.update(y_train.tolist())
        label_set.update(y_test.tolist())

        x_train_df = train_df.drop(columns=[label_col])
        x_test_df = test_df.drop(columns=[label_col])

        combined = pd.concat([x_train_df, x_test_df], axis=0)
        combined = pd.get_dummies(combined, drop_first=False)  # dtype default; ok para schema
        feature_set.update([str(c) for c in combined.columns.tolist()])

    feature_columns = sorted(feature_set)
    label_classes = sorted(label_set)
    label_to_index = {lab: i for i, lab in enumerate(label_classes)}
    return feature_columns, label_classes, label_to_index


def _load_client_data(
    client_dir: str,
    feature_columns: List[str],
    label_col_guess: Optional[str],
    label_to_index: Dict[str, int],
) -> ClientData:
    train_df, test_df, label_col = _read_client_dfs(client_dir)

    # Se o label_col variar entre clientes (raro), usamos o inferido por cliente.
    if label_col_guess is not None:
        label_col = label_col_guess if label_col_guess in train_df.columns else label_col

    x_train_df = train_df.drop(columns=[label_col])
    y_train = train_df[label_col].astype(str)

    x_test_df = test_df.drop(columns=[label_col])
    y_test = test_df[label_col].astype(str)

    combined = pd.concat([x_train_df, x_test_df], axis=0)
    combined = pd.get_dummies(combined, drop_first=False)

    # Alinha TODAS as features para o schema global (colunas faltantes viram 0)
    combined = combined.reindex(columns=feature_columns, fill_value=0)

    x_train_df = combined.iloc[: len(x_train_df)].copy()
    x_test_df = combined.iloc[len(x_train_df) :].copy()

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_df.values).astype(np.float32, copy=False)
    x_test = scaler.transform(x_test_df.values).astype(np.float32, copy=False)

    # Estabiliza extremos/NaN/Inf (baseline robusto)
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=10.0, neginf=-10.0)
    x_test = np.nan_to_num(x_test, nan=0.0, posinf=10.0, neginf=-10.0)
    x_train = np.clip(x_train, -10.0, 10.0).astype(np.float32, copy=False)
    x_test = np.clip(x_test, -10.0, 10.0).astype(np.float32, copy=False)

    y_train_enc = np.array([label_to_index[v] for v in y_train.tolist()], dtype=np.int64)
    y_test_enc = np.array([label_to_index[v] for v in y_test.tolist()], dtype=np.int64)

    return ClientData(
        x_train=x_train,
        y_train=y_train_enc,
        x_test=x_test,
        y_test=y_test_enc,
    )


# -------------------------
# Model param helpers
# -------------------------
def _get_model_params(model: nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def _set_model_params(model: nn.Module, params: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    if len(params) != len(state_dict):
        raise ValueError(
            f"Parameter length mismatch: got {len(params)}, expected {len(state_dict)}"
        )

    new_state_dict = {}
    for (key, _), val in zip(state_dict.items(), params, strict=True):
        new_state_dict[key] = torch.tensor(val)
    model.load_state_dict(new_state_dict, strict=True)


def _count_model_bytes(model: nn.Module) -> Tuple[int, int]:
    total_params = 0
    total_bytes = 0
    for param in model.parameters():
        total_params += param.numel()
        total_bytes += param.numel() * param.element_size()
    return total_params, total_bytes


# -------------------------
# Output paths + robust JSON update
# -------------------------
def _client_json_path(run_root: str, partition_tag: str, client_id: int) -> str:
    client_folder = os.path.join(run_root, partition_tag, f"client-id-{client_id}")
    os.makedirs(client_folder, exist_ok=True)
    return os.path.join(client_folder, f"nn_client-id-{client_id}.json")


def _registry_path(run_root: str, partition_tag: str) -> str:
    folder = os.path.join(run_root, partition_tag)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "_nodeid_to_clientidx.json")


def _update_results_round(output_path: str, server_round: int, updates: Dict[str, Any]) -> None:
    """
    Robust persistent JSON update (works with Ray recreating client instances):
      lock -> read JSON -> update round -> write tmp -> os.replace
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lock_path = output_path + ".lock"
    tmp_path = output_path + ".tmp"

    with open(lock_path, "a+", encoding="utf-8") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)

        data: Dict[str, Any] = {}
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        rec = data.setdefault(str(server_round), {})
        if not isinstance(rec, dict):
            rec = {}
            data[str(server_round)] = rec

        rec.update(updates)

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        os.replace(tmp_path, output_path)
        fcntl.flock(lockf, fcntl.LOCK_UN)


# -------------------------
# Map Flower/Ray context -> client index 0..N-1
# -------------------------
def _try_get_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _get_client_index_from_context(
    context: Context,
    num_clients: int,
    registry_path: str,
) -> int:
    """
    Preferimos um id estável se existir (cid/partition-id).
    Caso não exista, caímos no mapping persistente por node_id (por execução).
    """
    # 1) node_config hints
    if isinstance(getattr(context, "node_config", None), dict):
        nc = context.node_config
        for key in ["partition-id", "partition_id", "cid", "client_id"]:
            pid = _try_get_int(nc.get(key))
            if pid is not None:
                if not (0 <= pid < num_clients):
                    raise ValueError(f"{key} out of range: {pid} (num_clients={num_clients})")
                return pid

    # 2) context attributes hints
    for attr in ["cid", "client_id", "partition_id"]:
        pid = _try_get_int(getattr(context, attr, None))
        if pid is not None:
            if not (0 <= pid < num_clients):
                raise ValueError(f"{attr} out of range: {pid} (num_clients={num_clients})")
            return pid

    # 3) fallback: mapping node_id -> idx (por execução)
    node_id = int(getattr(context, "node_id"))

    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    with open(registry_path, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)

        f.seek(0)
        raw = f.read().strip()
        try:
            mapping: Dict[str, int] = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            mapping = {}

        key = str(node_id)
        if key in mapping:
            idx = int(mapping[key])
        else:
            used = set(int(v) for v in mapping.values())
            idx = None
            for cand in range(num_clients):
                if cand not in used:
                    idx = cand
                    break
            if idx is None:
                raise RuntimeError(
                    f"More node_ids than available clients. node_id={node_id}, num_clients={num_clients}"
                )
            mapping[key] = int(idx)

            f.seek(0)
            f.truncate()
            json.dump(mapping, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        fcntl.flock(f, fcntl.LOCK_UN)

    if not (0 <= idx < num_clients):
        raise RuntimeError(f"Invalid computed index: {idx}")
    return int(idx)


# -------------------------
# Flower client
# -------------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        data: ClientData,
        input_dim: int,
        num_classes: int,
        output_path: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        label_classes: List[str],
    ) -> None:
        self.client_id = client_id
        self.data = data
        self.model = SimpleMLP(input_dim=input_dim, num_classes=num_classes)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.output_path = output_path
        self.label_classes = label_classes
        self.round_start: Dict[int, float] = {}

    def get_parameters(self, config: Dict[str, str]):  # type: ignore[override]
        return _get_model_params(self.model)

    def fit(self, parameters, config):  # type: ignore[override]
        _set_model_params(self.model, parameters)

        server_round = int(config.get("server_round", 0))
        start_time = time.time()
        self.round_start[server_round] = start_time

        dataset = TensorDataset(
            torch.tensor(self.data.x_train, dtype=torch.float32),
            torch.tensor(self.data.y_train, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * batch_x.size(0)

        end_time = time.time()
        fit_time = end_time - start_time

        avg_loss = total_loss / max(1, len(dataset))
        params_count, params_bytes = _count_model_bytes(self.model)

        _update_results_round(
            self.output_path,
            server_round,
            {
                "fit_time": float(fit_time),
                "train_loss": float(avg_loss),
                "num_train_samples": int(len(dataset)),
                "model_param_count": int(params_count),
                "model_size_bytes": int(params_bytes),
                "epochs": int(self.epochs),
                "batch_size": int(self.batch_size),
                "learning_rate": float(self.learning_rate),
                "round_start_time": float(start_time),
                "target_labels": self.label_classes,
                "input_features": int(self.model.net[0].in_features),
                "output_classes": int(self.model.net[-1].out_features),
            },
        )

        return _get_model_params(self.model), len(dataset), {"train_loss": float(avg_loss)}

    def evaluate(self, parameters, config):  # type: ignore[override]
        _set_model_params(self.model, parameters)
        server_round = int(config.get("server_round", 0))

        dataset = TensorDataset(
            torch.tensor(self.data.x_test, dtype=torch.float32),
            torch.tensor(self.data.y_test, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        eval_start = time.time()
        total_loss = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []

        inference_start = time.time()
        with torch.no_grad():
            for batch_x, batch_y in loader:
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(batch_y.cpu().numpy().tolist())
        inference_end = time.time()
        eval_end = time.time()

        avg_loss = total_loss / max(1, len(dataset))
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)

        acc = float(accuracy_score(y_true, y_pred))
        f1w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        # Mantém matriz com shape fixo (C x C), mesmo se alguma classe não aparecer no teste
        num_classes = len(self.label_classes)
        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            normalize="true",
        ).tolist()

        start_time = self.round_start.get(server_round, eval_start)
        round_end = eval_end

        _update_results_round(
            self.output_path,
            server_round,
            {
                "evaluate_time": float(eval_end - eval_start),
                "inference_time": float(inference_end - inference_start),
                "test_loss": float(avg_loss),
                "accuracy": acc,
                "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
                "f1_score": f1w,
                "mcc": float(matthews_corrcoef(y_true, y_pred)),
                "confusion_matrix": cm,
                "confusion_matrix_labels": [str(x) for x in self.label_classes],
                "num_test_samples": int(len(dataset)),
                "round_end_time": float(round_end),
                "round_time": float(round_end - start_time),
            },
        )

        return float(avg_loss), len(dataset), {"accuracy": acc, "f1_score": f1w}


# -------------------------
# Client factory
# -------------------------
def _build_client_fn(
    client_dirs: List[str],
    input_dim: int,
    num_classes: int,
    run_root: str,
    partition_tag: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    label_classes: List[str],
    label_to_index: Dict[str, int],
    feature_columns: List[str],
    registry_path: str,
    label_col_guess: Optional[str],
):
    data_cache: Dict[int, ClientData] = {}

    def client_fn(context: Context):
        idx = _get_client_index_from_context(context, len(client_dirs), registry_path)

        if idx not in data_cache:
            data_cache[idx] = _load_client_data(
                client_dirs[idx],
                feature_columns=feature_columns,
                label_col_guess=label_col_guess,
                label_to_index=label_to_index,
            )
        data = data_cache[idx]

        output_path = _client_json_path(run_root, partition_tag, idx)

        client = FlowerClient(
            client_id=idx,
            data=data,
            input_dim=input_dim,
            num_classes=num_classes,
            output_path=output_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            label_classes=label_classes,
        )

        # Deve retornar Client (não NumPyClient)
        return client.to_client()

    return client_fn


# -------------------------
# Main
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Federated NN training with Flower (FedAvg).")
    parser.add_argument(
        "--partition",
        required=True,
        help=(
            f"Partition subfolder under {PARTITIONS_ROOT} (e.g., ML-EdgeIIoT-FEDT/dominant_client) "
            "or an absolute path containing client_*/train.csv and test.csv."
        ),
    )
    parser.add_argument("--num-rounds", type=int, required=True, help="Number of federated rounds.")
    parser.add_argument("--num-clients", type=int, required=True, help="Number of clients to use.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per round.")
    parser.add_argument("--batch-size", type=int, default=32, help="Local batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--client-cpus", type=float, default=3.0, help="CPUs per virtual client (reduz paralelismo/RAM).")
    parser.add_argument("--client-gpus", type=float, default=0.0, help="GPUs per virtual client.")
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Opcional: sobrescreve o nome detectado do dataset (ex: ML-EdgeIIoT).",
    )

    args = parser.parse_args()

    partition_path = _resolve_partition_path(args.partition)
    client_dirs = _find_client_dirs(partition_path, args.num_clients)

    detected_dataset_name, partition_tag = _dataset_and_partition_tag(partition_path)
    dataset_name = args.dataset_name if args.dataset_name else detected_dataset_name

    # Run folder incremental por dataset
    run_root, run_id = _allocate_run_root(args.output_dir, dataset_name)
    _write_run_meta(run_root, dataset_name, partition_path, partition_tag, args)

    # Schema global (features + labels) para ficar agnóstico e consistente
    feature_columns, label_classes, label_to_index = _infer_global_schema(client_dirs)
    input_dim = len(feature_columns)
    num_classes = len(label_classes)

    # Registry por execução (não persiste entre reruns)
    registry_path = _registry_path(run_root, partition_tag)

    # Guess do label col (só para "ajudar"; se não existir, cada cliente infere)
    _, _, label_col_guess = _read_client_dfs(client_dirs[0])

    client_fn = _build_client_fn(
        client_dirs=client_dirs,
        input_dim=input_dim,
        num_classes=num_classes,
        run_root=run_root,
        partition_tag=partition_tag,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        label_classes=label_classes,
        label_to_index=label_to_index,
        feature_columns=feature_columns,
        registry_path=registry_path,
        label_col_guess=label_col_guess,
    )

    def fit_config(server_round: int) -> Dict[str, str]:
        return {"server_round": str(server_round)}

    def eval_config(server_round: int) -> Dict[str, str]:
        return {"server_round": str(server_round)}

    strategy = fl.server.strategy.FedAvg(
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=eval_config,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_dirs),
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": float(args.client_cpus), "num_gpus": float(args.client_gpus)},
        ray_init_args={"include_dashboard": False},
    )

    print(f"\n[OK] Resultados salvos em: {run_root}  (run_id={run_id})")


if __name__ == "__main__":
    main()