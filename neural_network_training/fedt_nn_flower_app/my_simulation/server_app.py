from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flwr.common import Context, Parameters, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from my_simulation.task import Net, get_weights, infer_schema


# ----------------------------
# Paths (edit here if needed)
# ----------------------------
SEED = 42

DATA_ROOT = "/home/yuri/FEDT_IDS2/partitions/ML-EdgeIIoT-FEDT/iid"
SAVE_ROOT = "/home/yuri/FEDT_IDS2/neural_network_training"

DEFAULT_LABEL_COL = "label"


def _safe_load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return []
        return json.loads(content)
    except json.JSONDecodeError:
        return []


def _safe_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


class LoggingFedAvg(FedAvg):
    """FedAvg that logs per-round metrics for EACH client (evaluate) + aggregated metrics."""

    def __init__(self, run_dir: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.run_dir = Path(run_dir)
        self.rounds_path = self.run_dir / "round_metrics.json"
        self.clients_dir = self.run_dir / "clients"
        self.clients_dir.mkdir(parents=True, exist_ok=True)

    def _append_round_entry(self, entry: Dict[str, Any]) -> None:
        rounds = _safe_load_json_list(self.rounds_path)
        rounds.append(entry)
        _safe_write_json(self.rounds_path, rounds)

    def _append_client_entry(self, cid: str, entry: Dict[str, Any]) -> None:
        cpath = self.clients_dir / f"client_{cid}.json"
        logs = _safe_load_json_list(cpath)
        logs.append(entry)
        _safe_write_json(cpath, logs)

    def aggregate_evaluate(self, server_round: int, results, failures):
        # results: List[Tuple[ClientProxy, EvaluateRes]]
        if not results:
            return None, {}

        # Per-client metrics
        per_client: Dict[str, Dict[str, Any]] = {}
        total_examples = sum(res.num_examples for _, res in results) or 1

        # Weighted aggregation
        def wavg(key: str) -> float:
            return float(
                sum(res.num_examples * float(res.metrics.get(key, 0.0)) for _, res in results) / total_examples
            )

        agg_loss = float(sum(res.num_examples * float(res.loss) for _, res in results) / total_examples)
        agg_metrics = {
            "accuracy": wavg("accuracy"),
            "precision": wavg("precision"),
            "recall": wavg("recall"),
            "f1": wavg("f1"),
        }

        for client, res in results:
            cid = getattr(client, "cid", "unknown")
            m = dict(res.metrics) if res.metrics else {}
            m["loss"] = float(res.loss)
            m["num_examples"] = int(res.num_examples)
            per_client[str(cid)] = m

            self._append_client_entry(str(cid), {"round": server_round, **m})

        round_entry = {
            "round": server_round,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "aggregate": {"loss": agg_loss, **agg_metrics},
            "clients": per_client,
            "failures": len(failures) if failures else 0,
        }
        self._append_round_entry(round_entry)

        return agg_loss, agg_metrics


def server_fn(context: Context) -> ServerAppComponents:
    # Read run config
    num_rounds = int(context.run_config.get("num-server-rounds", 20))
    fraction_fit = float(context.run_config.get("fraction-fit", 1.0))
    local_epochs = int(context.run_config.get("local-epochs", 1))
    label_col = str(context.run_config.get("label-col", DEFAULT_LABEL_COL))

    # Create run directory (everything saved under SAVE_ROOT)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(SAVE_ROOT) / f"run_{ts}_seed{SEED}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Infer schema and save it to run_dir/schema.json
    schema = infer_schema(
        data_root=DATA_ROOT,
        save_dir=str(run_dir),
        preferred_label_col=label_col,
        seed=SEED,
    )

    # Initialize model parameters
    net = Net(input_dim=int(schema["input_dim"]), num_classes=int(schema["num_classes"]))
    parameters = ndarrays_to_parameters(get_weights(net))

    # Detect number of clients from partitions on disk (for better min_* defaults)
    num_clients = int(schema.get("num_clients_detected", 1))
    min_fit_clients = max(1, int(round(num_clients * fraction_fit)))
    min_evaluate_clients = num_clients

    # Send only scalar config values; schema stays in run_dir/schema.json
    def fit_config(server_round: int) -> Dict[str, Any]:
        return {
            "server_round": server_round,
            "local_epochs": local_epochs,
            "run_dir": str(run_dir),
        }

    def eval_config(server_round: int) -> Dict[str, Any]:
        return {
            "server_round": server_round,
            "run_dir": str(run_dir),
        }

    strategy = LoggingFedAvg(
        run_dir=str(run_dir),
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=num_clients,
        initial_parameters=parameters,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=eval_config,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
