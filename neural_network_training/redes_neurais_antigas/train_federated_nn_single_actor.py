#!/usr/bin/env python3
"""Federated learning training with a simple neural network using Flower (FedAvg).

Example:
ray stop --force 2>/dev/null || true

python train_federated_nn_single_actor.py \
  --partition ML-EdgeIIoT-FEDT/dominant_client \
  --label-col Attack_type_6 \
  --num-rounds 40 \
  --num-clients 10 \
  --epochs 5 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --scaler global

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
import math
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
            "label_col": getattr(args, "label_col", None),
            "scaler": getattr(args, "scaler", None),
            "optimizer": getattr(args, "optimizer", None),
            "momentum": getattr(args, "momentum", None),
        },
    }
    path = os.path.join(run_root, "_run_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# -------------------------
# CSV loading / preprocessing
# -------------------------
def _read_client_dfs(client_dir: str, label_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    train_path = os.path.join(client_dir, "train.csv")
    test_path = os.path.join(client_dir, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing train.csv or test.csv in {client_dir}.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if not label_col:
        raise ValueError("--label-col must be a non-empty string.")

    # Strict validation: enforce label_col for every client
    if label_col not in train_df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in {train_path}. "
            f"Available columns: {list(train_df.columns)[:10]}... (total={len(train_df.columns)})"
        )
    if label_col not in test_df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in {test_path}. "
            f"Available columns: {list(test_df.columns)[:10]}... (total={len(test_df.columns)})"
        )

    return train_df, test_df, label_col
def _infer_global_schema(
    client_dirs: List[str],
    label_col: str,
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Cria:
      - feature_columns: união das colunas após get_dummies (ordem determinística)
      - label_classes: união das classes (strings) com ordenação determinística
      - label_to_index: mapeamento global label->id

    A coluna de label é definida explicitamente via --label-col (obrigatório).

    Observação: se os rótulos forem inteiros (ex.: 0..K), ordenamos numericamente
    para evitar a armadilha de ordenação lexicográfica ('10' < '2').
    """
    feature_set: set[str] = set()
    label_set: set[str] = set()

    for cd in client_dirs:
        train_df, test_df, resolved_label = _read_client_dfs(cd, label_col)

        y_train = train_df[resolved_label]
        y_test = test_df[resolved_label]

        # Guardamos como strings (estável para JSON), mas decidimos a ordem depois
        label_set.update(y_train.astype(str).tolist())
        label_set.update(y_test.astype(str).tolist())

        x_train_df = train_df.drop(columns=[resolved_label])
        x_test_df = test_df.drop(columns=[resolved_label])

        combined = pd.concat([x_train_df, x_test_df], axis=0)
        combined = pd.get_dummies(combined, drop_first=False)
        feature_set.update([str(c) for c in combined.columns.tolist()])

    feature_columns = sorted(feature_set)

    # Ordenação robusta das classes
    def _is_intlike(s: str) -> bool:
        s2 = s.strip()
        if s2.startswith("-"):
            s2 = s2[1:]
        return s2.isdigit()

    label_list = list(label_set)
    if label_list and all(_is_intlike(x) for x in label_list):
        label_classes = sorted(label_list, key=lambda x: int(x))
    else:
        label_classes = sorted(label_list)

    label_to_index = {lab: i for i, lab in enumerate(label_classes)}
    return feature_columns, label_classes, label_to_index

def _fit_global_scaler(
    client_dirs: List[str],
    feature_columns: List[str],
    label_col: str,
) -> StandardScaler:
    """
    Ajusta um StandardScaler GLOBAL apenas nos dados de TREINO (de todos os clientes),
    garantindo que todos os clientes usem a MESMA transformação. Isso é importante
    para o FedAvg: se cada cliente normaliza de um jeito diferente, o espaço de
    features muda e o modelo tende a não convergir.
    """
    scaler = StandardScaler()
    for cd in client_dirs:
        train_df, _, resolved_label = _read_client_dfs(cd, label_col)

        x_train_df = train_df.drop(columns=[resolved_label])
        x_train_df = pd.get_dummies(x_train_df, drop_first=False)
        x_train_df = x_train_df.reindex(columns=feature_columns, fill_value=0)

        # partial_fit para evitar concatenar tudo em memória
        scaler.partial_fit(x_train_df.values)

    return scaler


def _load_client_data(
    client_dir: str,
    feature_columns: List[str],
    label_col: str,
    label_to_index: Dict[str, int],
    scaler_mode: str,
    global_scaler: Optional[StandardScaler],
) -> ClientData:
    """Load a single client's data with a global feature schema.

    Memory notes:
    - Avoid concatenating train/test before get_dummies (can double peak memory).
    - Avoid DataFrame .copy() and repeated NumPy copies.
    - Apply NaN/Inf handling and clipping in-place.
    """
    train_df, test_df, resolved_label = _read_client_dfs(client_dir, label_col)

    # Labels (kept as strings for mapping stability)
    y_train_str = train_df[resolved_label].astype(str).to_numpy()
    y_test_str = test_df[resolved_label].astype(str).to_numpy()

    x_train_raw = train_df.drop(columns=[resolved_label])
    x_test_raw = test_df.drop(columns=[resolved_label])

    # One-hot encode separately to avoid concatenation peak memory
    try:
        x_train_df = pd.get_dummies(x_train_raw, drop_first=False, dtype=np.uint8)
        x_test_df = pd.get_dummies(x_test_raw, drop_first=False, dtype=np.uint8)
    except TypeError:
        # Older pandas without dtype=...
        x_train_df = pd.get_dummies(x_train_raw, drop_first=False)
        x_test_df = pd.get_dummies(x_test_raw, drop_first=False)

    # Align to global schema (missing columns become 0)
    x_train_df = x_train_df.reindex(columns=feature_columns, fill_value=0)
    x_test_df = x_test_df.reindex(columns=feature_columns, fill_value=0)

    # Convert to float32 once (keeps memory low)
    x_train = x_train_df.to_numpy(dtype=np.float32, copy=False)
    x_test = x_test_df.to_numpy(dtype=np.float32, copy=False)

    # Apply scaling in float32 (avoid sklearn's float64 output copies)
    if scaler_mode == "none":
        pass
    elif scaler_mode == "per-client":
        mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32, copy=False)
        var = x_train.var(axis=0, dtype=np.float64).astype(np.float32, copy=False)
        scale = np.sqrt(var, dtype=np.float32) if hasattr(np, "sqrt") else np.sqrt(var)
        # Avoid division by zero for constant columns
        scale = np.where(scale == 0.0, 1.0, scale).astype(np.float32, copy=False)

        x_train -= mean
        x_train /= scale
        x_test -= mean
        x_test /= scale
    elif scaler_mode == "global":
        if global_scaler is None:
            raise ValueError("global_scaler is None but scaler_mode='global'")

        mean = getattr(global_scaler, "mean_", None)
        scale = getattr(global_scaler, "scale_", None)
        if mean is None or scale is None:
            raise ValueError("global_scaler does not expose mean_/scale_ (unexpected)")

        mean = np.asarray(mean, dtype=np.float32)
        scale = np.asarray(scale, dtype=np.float32)
        scale = np.where(scale == 0.0, 1.0, scale)

        x_train -= mean
        x_train /= scale
        x_test -= mean
        x_test /= scale
    else:
        raise ValueError(f"Unknown scaler_mode: {scaler_mode}")

    # Stabilize NaN/Inf and clip in-place
    try:
        np.nan_to_num(x_train, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
        np.nan_to_num(x_test, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
    except TypeError:
        x_train = np.nan_to_num(x_train, nan=0.0, posinf=10.0, neginf=-10.0)
        x_test = np.nan_to_num(x_test, nan=0.0, posinf=10.0, neginf=-10.0)

    np.clip(x_train, -10.0, 10.0, out=x_train)
    np.clip(x_test, -10.0, 10.0, out=x_test)

    # Encode labels
    y_train_enc = np.fromiter((label_to_index[v] for v in y_train_str), dtype=np.int64, count=len(y_train_str))
    y_test_enc = np.fromiter((label_to_index[v] for v in y_test_str), dtype=np.int64, count=len(y_test_str))

    return ClientData(
        x_train=x_train,
        y_train=y_train_enc,
        x_test=x_test,
        y_test=y_test_enc,
    )

def _get_model_params(model: nn.Module) -> List[np.ndarray]:
    """Return model parameters as a list of NumPy ndarrays (Flower format)."""
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def _set_model_params(model: nn.Module, params: List[np.ndarray]) -> None:
    """Load Flower parameters (list of NumPy ndarrays) into the model."""
    state_dict = model.state_dict()
    if len(params) != len(state_dict):
        raise ValueError(
            f"Parameter length mismatch: got {len(params)}, expected {len(state_dict)}"
        )

    new_state_dict: Dict[str, torch.Tensor] = {}
    for (key, ref_tensor), arr in zip(state_dict.items(), params, strict=True):
        arr_np = np.asarray(arr)
        t = torch.from_numpy(arr_np)

        # Shape check helps catch mismatched architectures early
        if tuple(t.shape) != tuple(ref_tensor.shape):
            raise ValueError(
                f"Shape mismatch for '{key}': got {tuple(t.shape)}, expected {tuple(ref_tensor.shape)}"
            )

        # Match dtype exactly (avoids silently promoting to float64)
        if t.dtype != ref_tensor.dtype:
            t = t.to(dtype=ref_tensor.dtype)

        new_state_dict[key] = t

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
        optimizer_name: str,
        momentum: float,
    ) -> None:
        self.client_id = client_id
        self.data = data
        self.model = SimpleMLP(input_dim=input_dim, num_classes=num_classes)

        # "Estilo Flower docs": move modelo para GPU se existir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.output_path = output_path
        self.label_classes = label_classes

        self.optimizer_name = optimizer_name
        self.momentum = momentum

        self.round_start: Dict[int, float] = {}

    def get_parameters(self, config: Dict[str, str]):  # type: ignore[override]
        return _get_model_params(self.model)

    def _make_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=float(self.momentum),
            )
        raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    def fit(self, parameters, config):  # type: ignore[override]
        _set_model_params(self.model, parameters)

        server_round = int(config.get("server_round", 0))
        start_time = time.time()
        self.round_start[server_round] = start_time

        dataset = TensorDataset(
            torch.tensor(self.data.x_train, dtype=torch.float32),
            torch.tensor(self.data.y_train, dtype=torch.long),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

        self.model.train()
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = self._make_optimizer()

        total_loss = 0.0
        total_seen = 0

        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                bs = int(batch_x.size(0))
                total_loss += float(loss.item()) * bs
                total_seen += bs

        end_time = time.time()
        fit_time = end_time - start_time

        # média por amostra (ao longo de TODOS os epochs)
        denom = max(1, total_seen)
        avg_loss = total_loss / denom

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
                "optimizer": str(self.optimizer_name),
                "momentum": float(self.momentum) if self.optimizer_name == "sgd" else None,
                "device": str(self.device),
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
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

        self.model.eval()
        criterion = nn.CrossEntropyLoss().to(self.device)

        eval_start = time.time()
        total_loss = 0.0
        total_seen = 0
        all_preds: List[int] = []
        all_targets: List[int] = []

        inference_start = time.time()
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)

                bs = int(batch_x.size(0))
                total_loss += float(loss.item()) * bs
                total_seen += bs

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.detach().cpu().numpy().tolist())
                all_targets.extend(batch_y.detach().cpu().numpy().tolist())

        inference_end = time.time()
        eval_end = time.time()

        denom = max(1, total_seen)
        avg_loss = total_loss / denom

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
    label_col: str,
    scaler_mode: str,
    global_scaler: Optional[StandardScaler],
    optimizer_name: str,
    momentum: float,
):
    data_cache: Dict[int, ClientData] = {}

    def client_fn(context: Context):
        idx = _get_client_index_from_context(context, len(client_dirs), registry_path)

        if idx not in data_cache:
            data_cache[idx] = _load_client_data(
                client_dirs[idx],
                feature_columns=feature_columns,
                label_col=label_col,
                label_to_index=label_to_index,
                scaler_mode=scaler_mode,
                global_scaler=global_scaler,
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
            optimizer_name=optimizer_name,
            momentum=momentum,
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
    parser.add_argument(
        "--label-col",
        required=True,
        help="Nome exato da coluna de label (obrigatório). Ex.: label, Label, attack_cat.",
    )
    parser.add_argument(
    "--scaler",
    choices=["global", "per-client", "none"],
    default="global",
    help=(
        "Normalização das features: "
        "'global' (recomendado para FedAvg), 'per-client' (baseline antigo), ou 'none'."
    ),
)
    parser.add_argument(
    "--optimizer",
    choices=["adam", "sgd"],
    default="adam",
    help="Otimizador local (estilo docs: SGD; default: Adam).",
)
    parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    help="Momentum (somente para SGD).",
)
    parser.add_argument("--max-actors", type=int, default=1, help="Número máximo de atores Ray (virtual clients) em paralelo. Default=1 (mínimo) para evitar OOM.")
    parser.add_argument("--client-cpus", type=float, default=None, help="CPUs por virtual client. Se omitido, será calculado a partir de --max-actors.")
    parser.add_argument("--client-gpus", type=float, default=0.0, help="GPUs per virtual client.")
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Opcional: sobrescreve o nome detectado do dataset (ex: ML-EdgeIIoT).",
    )

    args = parser.parse_args()


    # --- Ray actor parallelism control (avoid OOM) ---
    if args.client_cpus is None:
        total_cpus = os.cpu_count() or 1
        max_actors = max(1, int(args.max_actors))
        # Choose CPUs per actor so that Ray creates at most `max_actors` actors:
        # actor_pool_size ~= floor(total_cpus / client_cpus)
        args.client_cpus = float(max(1, math.ceil(total_cpus / max_actors)))


    partition_path = _resolve_partition_path(args.partition)
    client_dirs = _find_client_dirs(partition_path, args.num_clients)

    detected_dataset_name, partition_tag = _dataset_and_partition_tag(partition_path)
    dataset_name = args.dataset_name if args.dataset_name else detected_dataset_name

    # Run folder incremental por dataset
    run_root, run_id = _allocate_run_root(args.output_dir, dataset_name)
    _write_run_meta(run_root, dataset_name, partition_path, partition_tag, args)

    # Schema global (features + labels) para ficar agnóstico e consistente
    # Resolve/valida label_col (se --label-col foi passado, é estrito)
    _, _, label_col = _read_client_dfs(client_dirs[0], args.label_col)

    feature_columns, label_classes, label_to_index = _infer_global_schema(client_dirs, label_col)
    input_dim = len(feature_columns)
    num_classes = len(label_classes)

    # Registry por execução (não persiste entre reruns)
    registry_path = _registry_path(run_root, partition_tag)
    # Normalização consistente entre clientes (evita "cada cliente em um espaço" no FedAvg)
    global_scaler: Optional[StandardScaler] = None
    if args.scaler == "global":
        global_scaler = _fit_global_scaler(
            client_dirs=client_dirs,
            feature_columns=feature_columns,
            label_col=label_col,
        )
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
        label_col=label_col,
        scaler_mode=args.scaler,
        global_scaler=global_scaler,
        optimizer_name=args.optimizer,
        momentum=args.momentum,
    )

    def fit_config(server_round: int) -> Dict[str, str]:
        return {"server_round": str(server_round)}

    def eval_config(server_round: int) -> Dict[str, str]:
        return {"server_round": str(server_round)}

    # Agregação de métricas (remove warnings do Flower e dá métricas globais mais úteis)
    def _weighted_average_fit(metrics):
        # metrics: List[Tuple[num_examples, Dict[str, float]]]
        losses = [num_examples * m.get("train_loss", 0.0) for num_examples, m in metrics]
        total = sum(num_examples for num_examples, _ in metrics)
        return {"train_loss": float(sum(losses) / max(1, total))}

    def _weighted_average_eval(metrics):
        accs = [num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics]
        f1s = [num_examples * m.get("f1_score", 0.0) for num_examples, m in metrics]
        total = sum(num_examples for num_examples, _ in metrics)
        return {
            "accuracy": float(sum(accs) / max(1, total)),
            "f1_score": float(sum(f1s) / max(1, total)),
        }

    strategy = fl.server.strategy.FedAvg(
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=eval_config,
        fit_metrics_aggregation_fn=_weighted_average_fit,
        evaluate_metrics_aggregation_fn=_weighted_average_eval,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_dirs),
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": float(args.client_cpus), "num_gpus": float(args.client_gpus)},
        ray_init_args={"include_dashboard": False},
    )

    print()
    print(f"[OK] Resultados salvos em: {run_root}  (run_id={run_id})")


if __name__ == "__main__":
    main()
