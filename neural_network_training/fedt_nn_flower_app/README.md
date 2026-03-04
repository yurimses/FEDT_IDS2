# FEDT - Neural Network Training (Flower + PyTorch)

This app trains a simple MLP for **multi-class classification** using your existing partitions:
`/home/yuri/FEDT_IDS2/partitions/ML-EdgeIIoT-FEDT/iid/client_i/{train.csv,test.csv}`

It writes logs to:
`/home/yuri/FEDT_IDS2/neural_network_training/run_<timestamp>_seed42/`

Inside that folder you get:
- `schema.json` (feature cols, label col, class mapping)
- `round_metrics.json` (per-round aggregate + **per-client** metrics)
- `clients/client_<cid>.json` (per-client metrics over rounds)

## Run

```bash
cd fedt_nn_flower_app
python -m pip install -e .
flwr run .
```

### IMPORTANT
Set `options.num-supernodes` in `pyproject.toml` to match your number of `client_*` folders.

### Changing label column
Edit `label-col` in `pyproject.toml` OR pass it in run config:
```bash
flwr run . --run-config label-col=your_label_column
```
