from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from my_simulation.task import Net, get_weights, load_partition_from_csv, set_seed, set_weights, train, evaluate_classification


SEED = 42

# Training hyperparameters (can be edited here)
BATCH_SIZE = 256
LR = 5e-4


class FlowerClient(NumPyClient):
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net: Optional[Net] = None
        self.trainloader = None
        self.testloader = None
        self.n_train = 0
        self.n_test = 0
        self.schema: Optional[Dict[str, Any]] = None

        set_seed(SEED)

    def _ensure_initialized(self, config: Dict[str, Any]) -> None:
        if self.net is not None:
            return

        run_dir = Path(config["run_dir"])
        schema_path = run_dir / "schema.json"
        if not schema_path.exists():
            raise FileNotFoundError(f"schema.json not found at {schema_path}")

        self.schema = json.loads(schema_path.read_text(encoding="utf-8"))
        data_root = Path(self.schema["data_root"])
        partition_dir = data_root / f"client_{self.pid}"

        self.net = Net(
            input_dim=int(self.schema["input_dim"]),
            num_classes=int(self.schema["num_classes"]),
        ).to(self.device)

        self.trainloader, self.testloader, self.n_train, self.n_test = load_partition_from_csv(
            partition_dir=str(partition_dir),
            schema=self.schema,
            batch_size=BATCH_SIZE,
        )

    def get_parameters(self, config: Dict[str, Any]):
        self._ensure_initialized(config)
        return get_weights(self.net)

    def fit(self, parameters, config: Dict[str, Any]):
        self._ensure_initialized(config)
        set_weights(self.net, parameters)

        local_epochs = int(config.get("local_epochs", 200))
        train_loss = train(self.net, self.trainloader, epochs=local_epochs, device=self.device, lr=LR)

        return get_weights(self.net), self.n_train, {"train_loss": float(train_loss)}

    def evaluate(self, parameters, config: Dict[str, Any]):
        self._ensure_initialized(config)
        set_weights(self.net, parameters)

        loss, metrics = evaluate_classification(
            self.net,
            self.testloader,
            device=self.device,
            num_classes=int(self.schema["num_classes"]),
        )

        # IMPORTANT: return (loss, num_examples, metrics)
        return float(loss), self.n_test, metrics


def client_fn(context: Context):
    # In simulation, Flower provides these keys:
    partition_id = int(context.node_config["partition-id"])
    return FlowerClient(partition_id).to_client()


app = ClientApp(client_fn=client_fn)
