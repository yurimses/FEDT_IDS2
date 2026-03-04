from __future__ import annotations

import os
import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic settings (may reduce performance, but helps reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Schema inference (dataset-agnostic)
# ----------------------------
_LABEL_CANDIDATES = (
    "label",
    "Label",
    "target",
    "Target",
    "class",
    "Class",
    "y",
    "Y",
    "label_target",
    "Label_target",
)

def _infer_label_col(columns: List[str], preferred: Optional[str] = None) -> str:
    if preferred and preferred in columns:
        return preferred
    for c in _LABEL_CANDIDATES:
        if c in columns:
            return c
    # Fallback: last column
    return columns[-1]


def _iter_unique_labels_csv(csv_path: Path, label_col: str, chunksize: int = 200_000) -> set[str]:
    uniq: set[str] = set()
    for chunk in pd.read_csv(csv_path, usecols=[label_col], chunksize=chunksize):
        # Normalize as string to keep mapping stable across numeric/categorical labels
        uniq.update(chunk[label_col].astype(str).unique().tolist())
    return uniq


def list_client_dirs(data_root: Path) -> List[Path]:
    return sorted(
        [p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("client_")],
        key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else p.name,
    )


def infer_schema(
    data_root: str,
    save_dir: str,
    preferred_label_col: str = "label",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Create a schema.json inside save_dir to keep:
      - feature_cols (ordered)
      - label_col
      - class_names (sorted)
      - class_to_index
      - input_dim, num_classes
      - data_root, seed
    """
    set_seed(seed)

    data_root_p = Path(data_root)
    save_dir_p = Path(save_dir)
    save_dir_p.mkdir(parents=True, exist_ok=True)

    client_dirs = list_client_dirs(data_root_p)
    if not client_dirs:
        raise FileNotFoundError(f"No 'client_*' folders found under: {data_root_p}")

    # Use client_0/train.csv headers (fast) to define base columns
    header_df = pd.read_csv(client_dirs[0] / "train.csv", nrows=0)
    columns = header_df.columns.tolist()
    label_col = _infer_label_col(columns, preferred_label_col)
    feature_cols = [c for c in columns if c != label_col]

    # Collect unique labels across all clients (train + test) using chunked reading
    all_labels: set[str] = set()
    for cdir in client_dirs:
        train_csv = cdir / "train.csv"
        test_csv = cdir / "test.csv"
        if train_csv.exists():
            all_labels |= _iter_unique_labels_csv(train_csv, label_col)
        if test_csv.exists():
            all_labels |= _iter_unique_labels_csv(test_csv, label_col)

    # Deterministic ordering
    class_names = sorted(list(all_labels))
    class_to_index = {name: i for i, name in enumerate(class_names)}

    schema = {
        "seed": seed,
        "data_root": str(data_root_p),
        "label_col": label_col,
        "feature_cols": feature_cols,
        "class_names": class_names,
        "class_to_index": class_to_index,
        "input_dim": len(feature_cols),
        "num_classes": len(class_names),
        "num_clients_detected": len(client_dirs),
    }

    with open(save_dir_p / "schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    return schema


# ----------------------------
# Model
# ----------------------------
class Net(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()

        # 6 layers (Linear): 5 hidden + 1 output
        # Total hidden neurons = 600 (5 * 120)
        # Added Dropout and BatchNorm for better generalization
        hidden_sizes = [120, 120, 120, 120, 120]

        layers = []
        in_dim = input_dim
        for i, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))  # BatchNorm for stability
            layers.append(nn.ReLU())
            if i < len(hidden_sizes) - 1:  # Don't add dropout before last hidden layer
                layers.append(nn.Dropout(dropout))  # Dropout for regularization
            in_dim = h

        # Output layer (no activation here; CrossEntropyLoss expects logits)
        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_weights(net: nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: nn.Module, parameters: List[np.ndarray]) -> None:
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# ----------------------------
# Data loading
# ----------------------------
def _to_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Convert everything to numeric; non-numeric -> NaN -> 0
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def load_partition_from_csv(
    partition_dir: str,
    schema: Dict[str, Any],
    batch_size: int = 256,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Loads train.csv and test.csv from a partition dir (client_i).
    Returns (trainloader, testloader, n_train, n_test)
    """
    pdir = Path(partition_dir)
    train_path = pdir / "train.csv"
    test_path = pdir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing train.csv/test.csv in {pdir}")

    label_col = schema["label_col"]
    feature_cols = schema["feature_cols"]
    class_to_index = schema["class_to_index"]

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if label_col not in train_df.columns or label_col not in test_df.columns:
        raise KeyError(
            f"Label column '{label_col}' not found. "
            f"Train cols: {train_df.columns.tolist()[:10]}...  "
            f"Test cols: {test_df.columns.tolist()[:10]}..."
        )

    # Ensure identical feature ordering
    X_train = _to_numeric_frame(train_df.reindex(columns=feature_cols, fill_value=0.0))
    X_test = _to_numeric_frame(test_df.reindex(columns=feature_cols, fill_value=0.0))

    # NORMALIZE the features (critical for neural networks!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)

    # Map labels -> indices (normalize to str)
    y_train = train_df[label_col].astype(str).map(class_to_index).astype(np.int64)
    y_test = test_df[label_col].astype(str).map(class_to_index).astype(np.int64)

    # Tensor conversion
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.long)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.long)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    # Deterministic shuffling
    g = torch.Generator()
    g.manual_seed(int(schema.get("seed", 42)))

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    testloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return trainloader, testloader, len(train_ds), len(test_ds)


# ----------------------------
# Train/Eval
# ----------------------------
def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
) -> float:
    epochs = min(int(epochs), 200)
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    total_loss = 0.0
    total_batches = 0

    for _ in range(epochs):
        for xb, yb in trainloader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = net(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

    return total_loss / max(total_batches, 1)


def _macro_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Tuple[float, float, float, float]:
    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    total = cm.sum()
    acc = float(np.trace(cm) / total) if total > 0 else 0.0

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    precision = float(np.mean(precisions)) if precisions else 0.0
    recall = float(np.mean(recalls)) if recalls else 0.0
    f1 = float(np.mean(f1s)) if f1s else 0.0

    return acc, precision, recall, f1


@torch.no_grad()
def evaluate_classification(
    net: nn.Module,
    testloader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, Dict[str, float]]:
    net.eval()
    criterion = nn.CrossEntropyLoss()

    losses: List[float] = []
    y_true_list: List[int] = []
    y_pred_list: List[int] = []

    for xb, yb in testloader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = net(xb)
        loss = criterion(logits, yb)
        losses.append(float(loss.item()))

        preds = torch.argmax(logits, dim=1)
        y_true_list.extend(yb.detach().cpu().numpy().tolist())
        y_pred_list.extend(preds.detach().cpu().numpy().tolist())

    y_true = np.asarray(y_true_list, dtype=np.int64)
    y_pred = np.asarray(y_pred_list, dtype=np.int64)

    acc, precision, recall, f1 = _macro_metrics(y_true, y_pred, num_classes)

    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    return float(np.mean(losses) if losses else 0.0), metrics
