"""
graficos_unlearning.py

Script adaptado para visualizar resultados com UNLEARNING (machine unlearning).
Trata adequadamente casos onde clientes são "esquecidos" durante rounds específicos.

Diferenças em relação a graficos_edgeiot.py:
- Detecta automaticamente rounds onde clientes desaparecem
- Visualizações incluem anotações sobre pontos de unlearning
- Calcula métricas separadas para períodos antes/depois do unlearning
- Gráficos de boxplot mostram redução de clientes após unlearning
"""

import json
from pathlib import Path
from statistics import mean, pstdev
import math
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from collections import defaultdict
import tomllib
import pandas as pd
from fedt.settings import dataset_path as SETTINGS_DATASET_PATH, label_target as SETTINGS_LABEL_TARGET

AXIS_LABEL_SIZE = 14
AXIS_LABEL_WEIGHT = "bold"
TICK_LABEL_SIZE = 14
plt.rcParams.update({
    "axes.labelsize": AXIS_LABEL_SIZE,
    "axes.labelweight": AXIS_LABEL_WEIGHT,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
})

CONFUSION_MATRIX_CELL_SIZE = 8
CONFUSION_MATRIX_TEXT_SIZE = 18
CONFUSION_MATRIX_AXIS_LABEL_SIZE = 18
CONFUSION_MATRIX_TICK_LABEL_SIZE = 18
CONFUSION_MATRIX_TITLE_SIZE = 18
CONFUSION_MATRIX_COLORBAR_TICK_SIZE = 18

# ==========================
# CONFIGURAÇÃO DE ARQUIVOS
# ==========================

# Arquivos dos clientes - UNLEARNING (dominant_client)
CLIENT_FILES = [
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/dominant_client/client-id-0/best_trees_client-id-0_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/dominant_client/client-id-1/best_trees_client-id-1_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/dominant_client/client-id-2/best_trees_client-id-2_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/dominant_client/client-id-3/best_trees_client-id-3_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/dominant_client/client-id-4/best_trees_client-id-4_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/dominant_client/client-id-5/best_trees_client-id-5_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/dominant_client/client-id-6/best_trees_client-id-6_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/dominant_client/client-id-7/best_trees_client-id-7_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/dominant_client/client-id-8/best_trees_client-id-8_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/dominant_client/client-id-9/best_trees_client-id-9_1.json"),
]

# Arquivo do servidor
SERVER_FILE = Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/dominant_client/server/best_trees_server_1.json")

# Arquivo de monitoramento de CPU/RAM
CPU_FILE = Path("/home/yuri/FEDT_IDS2/logs/cpu_ram/ML-EdgeIIoT-FEDT/dominant_client/best_trees/cpu_and_ram_yuri_best_trees_0.json")

# Pasta onde as figuras serão salvas
FIG_DIR = Path("/home/yuri/FEDT_IDS2/figures/best_trees/edgeiot_unlearning/")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
# FUNÇÕES AUXILIARES
# ==========================

def save_figure_pdf(fig_or_plt, output_dir: Path, filename: str):
    """Salva figura em PDF de alta resolução (300 DPI) para publicação."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{filename}.pdf"
    
    if hasattr(fig_or_plt, 'savefig'):
        fig_or_plt.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def detect_unlearning_point(client_files):
    """Detecta automaticamente o round onde clientes desaparecem (unlearning).
    
    Retorna:
      - unlearning_round: primeiro round onde o número de clientes diminui
      - client_counts: dicionário {round: num_clientes}
    """
    client_counts = {}
    
    for path in client_files:
        if not path.exists():
            continue
        data = load_json(path)
        for round_id_str in data.keys():
            round_id = int(round_id_str)
            client_counts[round_id] = client_counts.get(round_id, 0) + 1
    
    # Encontrar transição
    unlearning_round = None
    prev_count = None
    for r in sorted(client_counts.keys()):
        if prev_count is not None and client_counts[r] < prev_count:
            unlearning_round = r
            break
        prev_count = client_counts[r]
    
    return unlearning_round, client_counts


def aggregate_client_metrics(client_files):
    """Lê os JSONs dos clientes e devolve um dicionário:
    
    rounds_data[round_id:int] = {
        "f1": [f1_c0, f1_c1, ...],
        "acc": [acc_c0, acc_c1, ...],
        "round_time": [rt_c0, rt_c1, ...],
        "fit_time": [],
        "inference_time": [],
        "round_start": [],
        "round_end": [],
    }
    """
    rounds_data = {}

    for path in client_files:
        if not path.exists():
            continue
        data = load_json(path)
        for round_id_str, metrics in data.items():
            round_id = int(round_id_str)
            if round_id not in rounds_data:
                rounds_data[round_id] = {
                    "f1": [],
                    "acc": [],
                    "round_time": [],
                    "fit_time": [],
                    "inference_time": [],
                    "round_start": [],
                    "round_end": [],
                }

            rounds_data[round_id]["f1"].append(metrics["f1_score"])
            rounds_data[round_id]["acc"].append(metrics["accuracy"])
            rounds_data[round_id]["round_time"].append(metrics["round_time"])
            if "fit_time" in metrics:
                rounds_data[round_id]["fit_time"].append(metrics["fit_time"])
            rounds_data[round_id]["round_start"].append(metrics["round_start_time"])
            rounds_data[round_id]["round_end"].append(metrics["round_end_time"])
            if "inference_time" in metrics:
                rounds_data[round_id]["inference_time"].append(metrics["inference_time"])

    return rounds_data


def summarize_rounds(rounds_data):
    """A partir de rounds_data, calcula média e desvio padrão
    para F1, acurácia, tempo de treinamento (fit_time) e tempo de inferência.
    
    Retorna listas ordenadas por round (int):
      rounds_int, f1_mean, f1_std, acc_mean, acc_std,
      train_mean, train_std, infer_mean, infer_std.
    """
    if not rounds_data:
        return [], [], [], [], [], [], [], [], []

    rounds_int = sorted(rounds_data.keys())

    f1_mean, f1_std = [], []
    acc_mean, acc_std = [], []
    train_mean, train_std = [], []
    infer_mean, infer_std = [], []

    for r in rounds_int:
        f1_vals = rounds_data[r]["f1"]
        acc_vals = rounds_data[r]["acc"]

        # treinamento: usa fit_time se existir, senão round_time
        train_vals = rounds_data[r]["fit_time"] or rounds_data[r]["round_time"]
        infer_vals = rounds_data[r]["inference_time"]

        f1_mean.append(mean(f1_vals))
        acc_mean.append(mean(acc_vals))
        train_mean.append(mean(train_vals))

        f1_std.append(pstdev(f1_vals) if len(f1_vals) > 1 else 0.0)
        acc_std.append(pstdev(acc_vals) if len(acc_vals) > 1 else 0.0)
        train_std.append(pstdev(train_vals) if len(train_vals) > 1 else 0.0)

        if infer_vals:
            infer_mean.append(mean(infer_vals))
            infer_std.append(pstdev(infer_vals) if len(infer_vals) > 1 else 0.0)
        else:
            infer_mean.append(0.0)
            infer_std.append(0.0)

    return (
        rounds_int,
        f1_mean,
        f1_std,
        acc_mean,
        acc_std,
        train_mean,
        train_std,
        infer_mean,
        infer_std,
    )


def load_server_metrics(path: Path):
    """Lê o JSON do servidor, esperado no formato:
    {
      "0": {
        "trees_by_client": ...,
        "aggregation_time": ...,
        "avg_execution_time": ...
      },
      ...
    }
    """
    if not path.exists():
        return [], [], [], []

    data = load_json(path)
    rounds_int = sorted(int(r) for r in data.keys())

    trees_by_client = []
    aggregation_time = []
    avg_execution_time = []

    for r in rounds_int:
        m = data[str(r)]
        trees_by_client.append(m["trees_by_client"])
        aggregation_time.append(m["aggregation_time"])
        avg_execution_time.append(m["avg_execution_time"])

    return rounds_int, trees_by_client, aggregation_time, avg_execution_time


def load_cpu_ram_series(path: Path):
    """Lê o JSON de monitoramento de CPU/RAM."""
    if not path.exists():
        return {}

    data = load_json(path)
    series = {}

    def build_mean_series(target_key, label, bin_size_s: float = 1.0):
        if target_key not in data:
            return
        pid_to_samples = data[target_key]
        if not pid_to_samples:
            return
        t0 = min(s["timestamp"] for samples in pid_to_samples.values() for s in samples)

        per_pid_bin_cpu = {}
        per_pid_bin_mem = {}
        for pid, samples in pid_to_samples.items():
            bins_cpu = {}
            bins_mem = {}
            for s in samples:
                t_rel = s["timestamp"] - t0
                b = int(t_rel // bin_size_s)
                bins_cpu.setdefault(b, []).append(s["cpu_percent"])
                bins_mem.setdefault(b, []).append(s["memory_mb"])
            per_pid_bin_cpu[pid] = {b: mean(v) for b, v in bins_cpu.items()}
            per_pid_bin_mem[pid] = {b: mean(v) for b, v in bins_mem.items()}

        all_bins = sorted({b for d in per_pid_bin_cpu.values() for b in d.keys()})
        t = []
        cpu = []
        mem = []
        for b in all_bins:
            cpu_vals = [d[b] for d in per_pid_bin_cpu.values() if b in d]
            mem_vals = [d[b] for d in per_pid_bin_mem.values() if b in d]
            if not cpu_vals or not mem_vals:
                continue
            t.append(b * bin_size_s)
            cpu.append(mean(cpu_vals))
            mem.append(mean(mem_vals))

        series[label] = {"t": t, "cpu": cpu, "mem": mem}

    build_mean_series("--client-id", "client")
    build_mean_series("fedt run server", "server")

    return series


def compute_cpu_usage_per_round(rounds_data, cpu_log_path: Path):
    """Computa CPU média por round para clientes e servidor."""
    if not cpu_log_path.exists() or not rounds_data:
        return [], [], []

    data = load_json(cpu_log_path)
    client_pids = data.get("--client-id", {})
    server_pids = data.get("fedt run server", {})

    rounds_int = sorted(rounds_data.keys())
    client_means = []
    server_means = []

    for r in rounds_int:
        starts = rounds_data[r]["round_start"]
        ends = rounds_data[r]["round_end"]
        if not starts or not ends:
            client_means.append(0.0)
            server_means.append(0.0)
            continue

        start_r = min(starts)
        end_r = max(ends)

        per_pid_means = []
        for samples in client_pids.values():
            vals = [s["cpu_percent"] for s in samples if start_r <= s["timestamp"] <= end_r]
            if vals:
                per_pid_means.append(mean(vals))
        client_means.append(mean(per_pid_means) if per_pid_means else 0.0)

        server_vals = []
        for samples in server_pids.values():
            vals = [s["cpu_percent"] for s in samples if start_r <= s["timestamp"] <= end_r]
            server_vals.extend(vals)
        server_means.append(mean(server_vals) if server_vals else 0.0)

    return rounds_int, client_means, server_means


# ==========================
# FUNÇÕES DE PLOT
# ==========================

def plot_f1_and_accuracy_with_unlearning(rounds, f1_mean, f1_std, acc_mean, acc_std, 
                                         unlearning_round, output_dir: Path):
    """Plota F1 e Acurácia, destacando o ponto de unlearning."""
    if not rounds:
        return

    round_labels = [r + 1 for r in rounds]

    # F1 médio por round
    plt.figure(figsize=(10, 6))
    plt.errorbar(round_labels, f1_mean, yerr=f1_std, fmt="-o", capsize=5, label="F1-Score")
    
    if unlearning_round is not None:
        ul_x = unlearning_round + 1
        plt.axvline(x=ul_x, color='red', linestyle='--', linewidth=2, label='Unlearning Point')
    
    # Auto-zoom: ajusta ylim baseado nos dados
    y_min = min(f1_mean) - 0.01
    y_max = max(f1_mean) + 0.01
    y_margin = (y_max - y_min) * 0.15
    plt.ylim(max(0.0, y_min - y_margin), min(1.0, y_max + y_margin))
    
    plt.xlabel("Round")
    plt.ylabel("F1-Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    save_figure_pdf(plt, output_dir, "fig1_f1_mean_unlearning")
    plt.close()

    # Acurácia média por round
    plt.figure(figsize=(10, 6))
    plt.errorbar(round_labels, acc_mean, yerr=acc_std, fmt="-o", capsize=5, label="Accuracy")
    
    if unlearning_round is not None:
        ul_x = unlearning_round + 1
        plt.axvline(x=ul_x, color='red', linestyle='--', linewidth=2, label='Unlearning Point')
    
    # Auto-zoom: ajusta ylim baseado nos dados
    y_min = min(acc_mean) - 0.001
    y_max = max(acc_mean) + 0.001
    y_margin = (y_max - y_min) * 0.15
    plt.ylim(max(0.0, y_min - y_margin), min(1.0, y_max + y_margin))
    
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    save_figure_pdf(plt, output_dir, "fig2_accuracy_mean_unlearning")
    plt.close()


def plot_f1_and_accuracy_per_client_and_mean(client_files, output_dir: Path):
    """Plota F1 e Accuracy por cliente e a média entre os clientes."""
    client_metrics = {}
    f1_per_round = defaultdict(list)
    acc_per_round = defaultdict(list)

    for path in client_files:
        if not path.exists():
            continue
        data = load_json(path)
        if not data:
            continue

        parent_name = path.parent.name
        
        if "client-id-" not in parent_name:
            gp = path.parent.parent.name
            if "client-id-" in gp:
                parent_name = gp

        if "client-id-" in parent_name:
            cid = parent_name.split("client-id-")[-1]
            label = f"Client {cid}"
        else:
            label = parent_name

        round_ids = sorted(int(r) for r in data.keys())
        rounds_client = []
        f1_client = []
        acc_client = []

        for r in round_ids:
            m = data[str(r)]
            rounds_client.append(r)
            f1_client.append(m["f1_score"])
            acc_client.append(m["accuracy"])
            f1_per_round[r].append(m["f1_score"])
            acc_per_round[r].append(m["accuracy"])

        client_metrics[label] = {
            "rounds": rounds_client,
            "f1": f1_client,
            "acc": acc_client,
        }

    if not client_metrics:
        return

    all_rounds = sorted(f1_per_round.keys())
    mean_f1 = [mean(f1_per_round[r]) for r in all_rounds]
    mean_acc = [mean(acc_per_round[r]) for r in all_rounds]
    round_labels_mean = [r + 1 for r in all_rounds]

    # F1 por cliente + média
    plt.figure(figsize=(12, 6))
    for label, m in client_metrics.items():
        x = [r + 1 for r in m["rounds"]]
        plt.plot(x, m["f1"], marker="o", alpha=0.6, linewidth=1.0, label=label)

    plt.plot(round_labels_mean, mean_f1, marker="o", linewidth=2.5, label="Mean (clients)")
    plt.xlabel("Round")
    plt.ylabel("F1-Score")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.3)
    save_figure_pdf(plt, output_dir, "fig9_f1_clients_and_mean")
    plt.close()

    # Accuracy por cliente + média
    plt.figure(figsize=(12, 6))
    for label, m in client_metrics.items():
        x = [r + 1 for r in m["rounds"]]
        plt.plot(x, m["acc"], marker="o", alpha=0.6, linewidth=1.0, label=label)

    plt.plot(round_labels_mean, mean_acc, marker="o", linewidth=2.5, label="Mean (clients)")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.3)
    save_figure_pdf(plt, output_dir, "fig10_accuracy_clients_and_mean")
    plt.close()


def plot_round_times(rounds, train_mean, train_std, server_rounds, aggregation_time, output_dir: Path):
    """Tempo médio de treinamento dos clientes + tempo de agregação do servidor."""
    if not rounds:
        return

    round_labels = [r + 1 for r in rounds]
    server_labels = [r + 1 for r in server_rounds] if server_rounds else []

    plt.figure()
    plt.errorbar(
        round_labels,
        train_mean,
        yerr=train_std,
        fmt="-o",
        capsize=5,
        label="Client training time (mean)",
    )
    if server_rounds and aggregation_time:
        plt.plot(server_labels, aggregation_time, "-s", label="Aggregation time (server)")
    plt.xlabel("Round")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    save_figure_pdf(plt, output_dir, "fig3_client_train_and_server_agg_time")
    plt.close()


def plot_memory_over_time(series, output_dir: Path):
    """Memória vs tempo (clientes e servidor)."""
    if not series:
        return

    plt.figure()
    plotted_any = False
    for label, v in series.items():
        if "t" not in v or "mem" not in v:
            continue
        plt.plot(v["t"], v["mem"], label=label)
        plotted_any = True

    if not plotted_any:
        plt.close()
        return

    plt.xlabel("Time (s)")
    plt.ylabel("Memory (MB)")
    plt.legend()
    save_figure_pdf(plt, output_dir, "fig4_memory_vs_time")
    plt.close()


def plot_inference_time(rounds, infer_mean, infer_std, output_dir: Path):
    """Tempo de inferência médio por round."""
    if not rounds or not infer_mean:
        return

    round_labels = [r + 1 for r in rounds]

    plt.figure()
    plt.errorbar(round_labels, infer_mean, yerr=infer_std, fmt="-o", capsize=5)
    plt.xlabel("Round")
    plt.ylabel("Inference time (s)")
    plt.grid(True, linestyle="--", alpha=0.3)
    save_figure_pdf(plt, output_dir, "fig5_inference_time_per_round")
    plt.close()


def plot_cpu_usage_per_round(rounds, client_cpu, server_cpu, output_dir: Path):
    """Uso médio de CPU por round."""
    if not rounds or not client_cpu or not server_cpu:
        return

    round_labels = [r + 1 for r in rounds]

    plt.figure()
    plt.plot(round_labels, client_cpu, "-o", label="Clients (mean)")
    plt.plot(round_labels, server_cpu, "-s", label="Server")
    plt.xlabel("Round")
    plt.ylabel("CPU usage (%)")
    plt.legend()
    save_figure_pdf(plt, output_dir, "fig6_cpu_usage_per_round")
    plt.close()


def extract_metrics_at_round(client_files, target_round=None):
    """Extrai métricas de um round específico."""
    from statistics import mean as _mean, pstdev as _pstdev

    acc_vals, prec_vals, rec_vals, f1_vals = [], [], [], []

    for path in client_files:
        if not path.exists():
            continue
        data = load_json(path)
        if not data:
            continue
        round_ids = sorted([int(r) for r in data.keys()])
        
        if target_round is None:
            r = max(round_ids)
        else:
            if target_round in round_ids:
                r = target_round
            else:
                r = max([rid for rid in round_ids if rid <= target_round], default=max(round_ids))
            
        if str(r) not in data:
            continue
            
        m = data[str(r)]
        acc_vals.append(m["accuracy"])
        prec_vals.append(m["precision"])
        rec_vals.append(m["recall"])
        f1_vals.append(m["f1_score"])

    if not acc_vals:
        return None

    acc_mean = _mean(acc_vals)
    prec_mean = _mean(prec_vals)
    rec_mean = _mean(rec_vals)
    f1_mean = _mean(f1_vals)

    acc_std = _pstdev(acc_vals) if len(acc_vals) > 1 else 0.0
    prec_std = _pstdev(prec_vals) if len(prec_vals) > 1 else 0.0
    rec_std = _pstdev(rec_vals) if len(rec_vals) > 1 else 0.0
    f1_std = _pstdev(f1_vals) if len(f1_vals) > 1 else 0.0

    return acc_mean, acc_std, prec_mean, prec_std, rec_mean, rec_std, f1_mean, f1_std


def plot_last_round_metrics_bar(client_files, output_dir: Path):
    """Gráfico de barras com média e desvio padrão de Accuracy, Recall, Precision e F1 no último round."""
    from statistics import mean as _mean, pstdev as _pstdev

    acc_vals, prec_vals, rec_vals, f1_vals = [], [], [], []
    last_round = None

    for path in client_files:
        if not path.exists():
            continue
        data = load_json(path)
        if not data:
            continue
        round_ids = [int(r) for r in data.keys()]
        r_last = max(round_ids)
        if last_round is None or r_last > last_round:
            last_round = r_last
        m = data[str(r_last)]
        acc_vals.append(m["accuracy"])
        prec_vals.append(m["precision"])
        rec_vals.append(m["recall"])
        f1_vals.append(m["f1_score"])

    if not acc_vals:
        return

    acc_mean = _mean(acc_vals)
    prec_mean = _mean(prec_vals)
    rec_mean = _mean(rec_vals)
    f1_mean = _mean(f1_vals)

    acc_std = _pstdev(acc_vals) if len(acc_vals) > 1 else 0.0
    prec_std = _pstdev(prec_vals) if len(prec_vals) > 1 else 0.0
    rec_std = _pstdev(rec_vals) if len(rec_vals) > 1 else 0.0
    f1_std = _pstdev(f1_vals) if len(f1_vals) > 1 else 0.0

    labels = ["Accuracy", "Recall", "Precision", "F1-score"]
    values = [acc_mean, rec_mean, prec_mean, f1_mean]
    errors = [acc_std, rec_std, prec_std, f1_std]
    x = range(len(labels))
    capped_errors = [min(err, max(0.0, 1.0 - val + 1e-6)) for val, err in zip(values, errors)]

    colors = ["#07a791", "#FF9900", "#ff000086", "#A864AA"]

    plt.figure()
    bars = plt.bar(
        x,
        values,
        yerr=capped_errors,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=1.0,
        width=0.6,
    )
    for bar, value, err in zip(bars, values, capped_errors):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + err + 0.01,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.08)
    plt.ylabel("Score")
    plt.grid(True, linestyle="--", alpha=0.3, axis="y")
    save_figure_pdf(plt, output_dir, "fig7_metrics")
    plt.close()


def plot_confusion_matrices_clients(client_files, output_dir: Path):
    """Plota subplots com a matriz de confusão do último round de cada cliente."""
    cms = []
    labels = []
    json_class_names = None

    for path in client_files:
        if not path.exists():
            continue
        data = load_json(path)
        if not data:
            continue
        round_ids = [int(r) for r in data.keys()]
        r_last = max(round_ids)
        m_last = data[str(r_last)]
        cm = m_last.get("confusion_matrix")
        if cm is None:
            continue

        if json_class_names is None:
            names_from_json = m_last.get("confusion_matrix_labels")
            if isinstance(names_from_json, list):
                json_class_names = names_from_json

        cms.append(cm)
        parent_name = path.parent.name
        
        if "client-id-" not in parent_name:
            gp = path.parent.parent.name
            if "client-id-" in gp:
                parent_name = gp

        if "client-id-" in parent_name:
            cid = parent_name.split("client-id-")[-1]
            labels.append(f"Client {cid}")
        else:
            labels.append(parent_name)

    if not cms:
        return

    n = len(cms)
    cols = min(5, n)
    rows = math.ceil(n / cols)

    fig_width = CONFUSION_MATRIX_CELL_SIZE * cols
    fig_height = (CONFUSION_MATRIX_CELL_SIZE + 0.5) * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    num_classes = len(cms[0])

    if isinstance(json_class_names, list) and len(json_class_names) == num_classes:
        class_names = json_class_names
    else:
        class_names = None

    ims = []
    for idx, cm in enumerate(cms):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        im = ax.imshow(cm, vmin=0.0, vmax=1.0)
        ims.append(im)
        ax.set_title(labels[idx], fontsize=CONFUSION_MATRIX_TITLE_SIZE, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=CONFUSION_MATRIX_AXIS_LABEL_SIZE, fontweight="bold")
        ax.set_ylabel("True", fontsize=CONFUSION_MATRIX_AXIS_LABEL_SIZE, fontweight="bold")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        if class_names is not None:
            ax.set_xticklabels(
                class_names, rotation=45, ha="right",
                fontsize=CONFUSION_MATRIX_TICK_LABEL_SIZE, fontweight="bold"
            )
            ax.set_yticklabels(
                class_names, fontsize=CONFUSION_MATRIX_TICK_LABEL_SIZE, fontweight="bold"
            )

        for i in range(num_classes):
            for j in range(num_classes):
                val = cm[i][j]
                text_color = "black" if i == j else "white"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color=text_color,
                    fontsize=CONFUSION_MATRIX_TEXT_SIZE,
                )

    for idx in range(len(cms), rows * cols):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        ax.axis("off")

    plt.subplots_adjust(hspace=0.3, wspace=0.7, right=0.88)

    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(ims[-1], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=CONFUSION_MATRIX_COLORBAR_TICK_SIZE)
    for t in cbar.ax.get_yticklabels():
        t.set_fontweight("bold")

    save_figure_pdf(fig, output_dir, "fig8_confusion_matrices_clients")
    plt.close(fig)


def plot_client_count_over_rounds(client_counts, unlearning_round, output_dir: Path):
    """Novo gráfico: mostra o número de clientes ativos em cada round."""
    rounds = sorted(client_counts.keys())
    counts = [client_counts[r] for r in rounds]
    round_labels = [r + 1 for r in rounds]

    plt.figure(figsize=(10, 6))
    plt.plot(round_labels, counts, "-o", linewidth=2, markersize=6, label="Active clients")
    
    if unlearning_round is not None:
        ul_x = unlearning_round + 1
        plt.axvline(x=ul_x, color='red', linestyle='--', linewidth=2, label='Unlearning Point')
    
    plt.xlabel("Round")
    plt.ylabel("Number of Active Clients")
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    save_figure_pdf(plt, output_dir, "fig0_client_count_unlearning")
    plt.close()


def plot_aggregated_metrics_over_rounds(client_files, unlearning_round, output_dir: Path):
    """Plota as médias agregadas de Accuracy, F1, Recall e Precision por round.
    
    Mostra apenas as linhas de médias (sem desempenho individual de clientes),
    com marcação do ponto de unlearning.
    """
    rounds_metrics = defaultdict(lambda: {
        "acc": [],
        "prec": [],
        "rec": [],
        "f1": []
    })

    for path in client_files:
        if not path.exists():
            continue
        data = load_json(path)
        if not data:
            continue

        for round_id_str, metrics in data.items():
            round_id = int(round_id_str)
            rounds_metrics[round_id]["acc"].append(metrics["accuracy"])
            rounds_metrics[round_id]["prec"].append(metrics["precision"])
            rounds_metrics[round_id]["rec"].append(metrics["recall"])
            rounds_metrics[round_id]["f1"].append(metrics["f1_score"])

    if not rounds_metrics:
        return

    all_rounds = sorted(rounds_metrics.keys())
    round_labels = [r + 1 for r in all_rounds]
    
    acc_means = [mean(rounds_metrics[r]["acc"]) for r in all_rounds]
    prec_means = [mean(rounds_metrics[r]["prec"]) for r in all_rounds]
    rec_means = [mean(rounds_metrics[r]["rec"]) for r in all_rounds]
    f1_means = [mean(rounds_metrics[r]["f1"]) for r in all_rounds]

    plt.figure(figsize=(12, 7))
    
    plt.plot(round_labels, acc_means, marker="o", linewidth=2.5, label="Accuracy", color="#07a791")
    plt.plot(round_labels, rec_means, marker="s", linewidth=2.5, label="Recall", color="#FF9900")
    plt.plot(round_labels, prec_means, marker="^", linewidth=2.5, label="Precision", color="#ff000086")
    plt.plot(round_labels, f1_means, marker="d", linewidth=2.5, label="F1-Score", color="#A864AA")
    
    if unlearning_round is not None:
        ul_x = unlearning_round + 1
        plt.axvline(x=ul_x, color='red', linestyle='--', linewidth=2.5, alpha=0.8, label='Unlearning Point')
    
    # Auto-zoom: ajusta ylim baseado nos dados de todas as métricas
    all_values = acc_means + rec_means + prec_means + f1_means
    y_min = min(all_values)
    y_max = max(all_values)
    y_margin = (y_max - y_min) * 0.15 if (y_max - y_min) > 0.01 else 0.02
    plt.ylim(max(0.0, y_min - y_margin), min(1.0, y_max + y_margin))
    
    plt.xlabel("Round")
    plt.ylabel("Score")
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.3)
    save_figure_pdf(plt, output_dir, "fig11_aggregated_metrics_unlearning")
    plt.close()


# ==========================
# MAIN
# ==========================

def main():
    print("="*70)
    print("GRAFICOS_UNLEARNING.PY - Processando dados com Machine Unlearning")
    print("="*70)
    
    # Detectar ponto de unlearning
    unlearning_round, client_counts = detect_unlearning_point(CLIENT_FILES)
    print(f"\n✓ Ponto de unlearning detectado: Round {unlearning_round}")
    print(f"  - Clientes antes: {max(client_counts.get(r, 0) for r in range(0, unlearning_round+1) if r in client_counts)}")
    print(f"  - Clientes depois: {min(client_counts.get(r, 0) for r in range(unlearning_round, 40) if r in client_counts)}")
    
    rounds_data = aggregate_client_metrics(CLIENT_FILES)
    (
        rounds,
        f1_mean,
        f1_std,
        acc_mean,
        acc_std,
        train_mean,
        train_std,
        infer_mean,
        infer_std,
    ) = summarize_rounds(rounds_data)

    server_rounds, trees_by_client, aggregation_time, avg_exec_time = load_server_metrics(SERVER_FILE)

    cpu_rounds, client_cpu_per_round, server_cpu_per_round = compute_cpu_usage_per_round(rounds_data, CPU_FILE)

    print(f"\n✓ Gerando gráficos em: {FIG_DIR}")
    
    # Gráficos adaptados para unlearning
    plot_client_count_over_rounds(client_counts, unlearning_round, FIG_DIR)
    plot_f1_and_accuracy_with_unlearning(rounds, f1_mean, f1_std, acc_mean, acc_std, unlearning_round, FIG_DIR)
    plot_f1_and_accuracy_per_client_and_mean(CLIENT_FILES, FIG_DIR)
    plot_aggregated_metrics_over_rounds(CLIENT_FILES, unlearning_round, FIG_DIR)
    plot_round_times(rounds, train_mean, train_std, server_rounds, aggregation_time, FIG_DIR)
    plot_memory_over_time(load_cpu_ram_series(CPU_FILE), FIG_DIR)
    plot_inference_time(rounds, infer_mean, infer_std, FIG_DIR)
    if cpu_rounds:
        plot_cpu_usage_per_round(cpu_rounds, client_cpu_per_round, server_cpu_per_round, FIG_DIR)
    plot_last_round_metrics_bar(CLIENT_FILES, FIG_DIR)
    plot_confusion_matrices_clients(CLIENT_FILES, FIG_DIR)
    
    print(f"\n✓ Gráficos gerados com sucesso!")


if __name__ == "__main__":
    main()
