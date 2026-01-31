import json
from pathlib import Path
from statistics import mean, pstdev
import math
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from collections import defaultdict
import tomllib  # [CLASSIF]
import pandas as pd  # [CLASSIF]
from fedt.settings import dataset_path as SETTINGS_DATASET_PATH, label_target as SETTINGS_LABEL_TARGET  # [CLASSIF]

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

# Arquivos dos clientes
CLIENT_FILES = [
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/iid/client-id-0/edgeiot/best_trees_client-id-0_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/iid/client-id-1/edgeiot/best_trees_client-id-1_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/iid/client-id-2/edgeiot/best_trees_client-id-2_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/iid/client-id-3/edgeiot/best_trees_client-id-3_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/iid/client-id-4/edgeiot/best_trees_client-id-4_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/iid/client-id-5/edgeiot/best_trees_client-id-5_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/iid/client-id-6/edgeiot/best_trees_client-id-6_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/iid/client-id-7/edgeiot/best_trees_client-id-7_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/iid/client-id-8/edgeiot/best_trees_client-id-8_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/iid/client-id-9/edgeiot/best_trees_client-id-9_1.json"),
]

# Arquivo do servidor
SERVER_FILE = Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/iid/server/edgeiot/best_trees_server_1.json")
# Arquivo de monitoramento de CPU/RAM IID
CPU_FILE = Path("/home/yuri/FEDT_IDS2/logs/cpu_ram/ML-EdgeIIoT-FEDT/iid/best_trees/cpu_and_ram_yuri_best_trees_0.json")

# Arquivos dos clientes NON-IID
CLIENT_FILES_NONIID = [
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/noniid/client-id-0/edgeiot/best_trees_client-id-0_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/noniid/client-id-1/edgeiot/best_trees_client-id-1_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/noniid/client-id-2/edgeiot/best_trees_client-id-2_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/noniid/client-id-3/edgeiot/best_trees_client-id-3_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/noniid/client-id-4/edgeiot/best_trees_client-id-4_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/noniid/client-id-5/edgeiot/best_trees_client-id-5_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/noniid/client-id-6/edgeiot/best_trees_client-id-6_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/noniid/client-id-7/edgeiot/best_trees_client-id-7_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/noniid/client-id-8/edgeiot/best_trees_client-id-8_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/noniid/client-id-9/edgeiot/best_trees_client-id-9_1.json"),
]

# Pasta onde as figuras serão salvas
FIG_DIR = Path("/home/yuri/FEDT_IDS2/figures/best_trees/edgeiot/")
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
        return [], [], [], [], [], [], [], []

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
    """Lê o JSON de monitoramento de CPU/RAM.

    Retorna:
      series = {
        "client": {"t": [...], "cpu": [...], "mem": [...]},
        "server": {...}
      }

    Escolhe sempre um PID por tipo (o primeiro encontrado).
    """
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
        # timestamp global mínimo (para começar em 0)
        t0 = min(s["timestamp"] for samples in pid_to_samples.values() for s in samples)

        # 1) média por PID dentro de cada bin de tempo
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

        # 2) média entre PIDs (peso igual por PID/cliente) em cada bin
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

    # média de TODOS os clientes
    build_mean_series("--client-id", "client")
    # média do servidor (se tiver múltiplos PIDs, também faz média)
    build_mean_series("fedt run server", "server")

    return series


def compute_cpu_usage_per_round(rounds_data, cpu_log_path: Path):
    """Computa CPU média por round para clientes e servidor.

    Para cada round, usa o menor round_start_time e o maior round_end_time
    entre os clientes para definir o intervalo [start_r, end_r] e,
    dentro desse intervalo, calcula:

      - CPU média dos clientes: média das médias de cpu_percent por PID de cliente
      - CPU média do servidor: média de todas as amostras de cpu_percent do servidor
    """
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

        # clientes: média das médias por PID
        per_pid_means = []
        for samples in client_pids.values():
            vals = [s["cpu_percent"] for s in samples if start_r <= s["timestamp"] <= end_r]
            if vals:
                per_pid_means.append(mean(vals))
        client_means.append(mean(per_pid_means) if per_pid_means else 0.0)

        # servidor: média de todas as amostras no intervalo
        server_vals = []
        for samples in server_pids.values():
            vals = [s["cpu_percent"] for s in samples if start_r <= s["timestamp"] <= end_r]
            server_vals.extend(vals)
        server_means.append(mean(server_vals) if server_vals else 0.0)

    return rounds_int, client_means, server_means


# ==========================
# FUNÇÕES DE PLOT
# ==========================

def plot_f1_and_accuracy(rounds, f1_mean, f1_std, acc_mean, acc_std, output_dir: Path):
    if not rounds:
        return

    round_labels = [r + 1 for r in rounds]

    # F1 médio por round
    plt.figure()
    plt.errorbar(round_labels, f1_mean, yerr=f1_std, fmt="-o", capsize=5)
    plt.xlabel("Round")
    plt.ylabel("F1-Score")
    # plt.title("F1")
    save_figure_pdf(plt, output_dir, "fig1_f1_mean")
    plt.close()

    # Acurácia média por round
    plt.figure()
    plt.errorbar(round_labels, acc_mean, yerr=acc_std, fmt="-o", capsize=5)
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    # plt.title("Accuracy")
    save_figure_pdf(plt, output_dir, "fig2_accuracy_mean")
    plt.close()


def plot_f1_and_accuracy_per_client_and_mean(client_files, output_dir: Path):
    """Plota F1 e Accuracy por cliente e a média entre os clientes (sem desvio padrão)."""
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
        
        # No layout .../client-id-X/<dataset>/arquivo.json, o parent é o dataset.
        # Então subimos um nível para capturar "client-id-X".
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
    plt.figure()
    for label, m in client_metrics.items():
        x = [r + 1 for r in m["rounds"]]
        plt.plot(x, m["f1"], marker="o", alpha=0.6, linewidth=1.0, label=label)

    plt.plot(
        round_labels_mean,
        mean_f1,
        marker="o",
        linewidth=2.5,
        label="Mean (clients)",
    )
    plt.xlabel("Round")
    plt.ylabel("F1-Score")
    # plt.title("F1 per client and mean")
    plt.legend(ncol=2)
    save_figure_pdf(plt, output_dir, "fig9_f1_clients_and_mean")
    plt.close()

    # Accuracy por cliente + média
    plt.figure()
    for label, m in client_metrics.items():
        x = [r + 1 for r in m["rounds"]]
        plt.plot(x, m["acc"], marker="o", alpha=0.6, linewidth=1.0, label=label)

    plt.plot(
        round_labels_mean,
        mean_acc,
        marker="o",
        linewidth=2.5,
        label="Mean (clients)",
    )
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    # plt.title("Accuracy per client and mean")
    plt.legend(ncol=2)
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
    # plt.title("Client training and server aggregation time per round")
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
    # plt.title("Memory usage over time")
    plt.legend()
    save_figure_pdf(plt, output_dir, "fig4_memory_vs_time")
    plt.close()


def plot_inference_time(rounds, infer_mean, infer_std, output_dir: Path):
    """Tempo de inferência médio por round (média entre clientes)."""
    if not rounds or not infer_mean:
        return

    round_labels = [r + 1 for r in rounds]

    plt.figure()
    plt.errorbar(round_labels, infer_mean, yerr=infer_std, fmt="-o", capsize=5)
    plt.xlabel("Round")
    plt.ylabel("Inference time (s)")
    # plt.title("Inference time per round (mean clients)")
    plt.grid(True, linestyle="--", alpha=0.3)
    save_figure_pdf(plt, output_dir, "fig5_inference_time_per_round")
    plt.close()


def plot_cpu_usage_per_round(rounds, client_cpu, server_cpu, output_dir: Path):
    """Uso médio de CPU por round (clientes e servidor)."""
    if not rounds or not client_cpu or not server_cpu:
        return

    round_labels = [r + 1 for r in rounds]

    plt.figure()
    plt.plot(round_labels, client_cpu, "-o", label="Clients (mean)")
    plt.plot(round_labels, server_cpu, "-s", label="Server")
    plt.xlabel("Round")
    plt.ylabel("CPU usage (%)")
    # plt.title("CPU usage per round")
    plt.legend()
    save_figure_pdf(plt, output_dir, "fig6_cpu_usage_per_round")
    plt.close()


def extract_metrics_at_round(client_files, target_round=None):
    """Extrai métricas de um round específico (ou do último round se target_round=None).
    
    Retorna tupla: (acc_mean, acc_std, prec_mean, prec_std, rec_mean, rec_std, f1_mean, f1_std)
    """
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
            # Se o round específico não existe, usa o mais próximo
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


def plot_iid_vs_noniid_metrics_comparison(client_files_iid, client_files_noniid, target_round=40, output_dir: Path = None):
    """Plota um gráfico de barras comparando métricas IID vs NON-IID no round especificado.
    
    Compara Accuracy, Recall, Precision e F1-score lado a lado.
    Se o round não existir em algum dos cenários, usa o round mais próximo disponível.
    """
    metrics_iid = extract_metrics_at_round(client_files_iid, target_round)
    metrics_noniid = extract_metrics_at_round(client_files_noniid, target_round)
    
    if metrics_iid is None or metrics_noniid is None:
        print(f"Aviso: Dados não disponíveis para round {target_round} em IID ou NON-IID")
        return
    
    acc_mean_iid, acc_std_iid, prec_mean_iid, prec_std_iid, rec_mean_iid, rec_std_iid, f1_mean_iid, f1_std_iid = metrics_iid
    acc_mean_noniid, acc_std_noniid, prec_mean_noniid, prec_std_noniid, rec_mean_noniid, rec_std_noniid, f1_mean_noniid, f1_std_noniid = metrics_noniid
    
    labels = ["Accuracy", "Recall", "Precision", "F1-score"]
    values_iid = [acc_mean_iid, rec_mean_iid, prec_mean_iid, f1_mean_iid]
    errors_iid = [acc_std_iid, rec_std_iid, prec_std_iid, f1_std_iid]
    values_noniid = [acc_mean_noniid, rec_mean_noniid, prec_mean_noniid, f1_mean_noniid]
    errors_noniid = [acc_std_noniid, rec_std_noniid, prec_std_noniid, f1_std_noniid]
    capped_errors_iid = [min(err, max(0.0, 1.0 - val + 1e-6)) for val, err in zip(values_iid, errors_iid)]
    capped_errors_noniid = [min(err, max(0.0, 1.0 - val + 1e-6)) for val, err in zip(values_noniid, errors_noniid)]
    
    x = range(len(labels))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    
    bars1 = plt.bar([i - width/2 for i in x], values_iid, width, yerr=capped_errors_iid, capsize=5,
                    label="IID", color="#07a791", edgecolor="black", linewidth=1.0)
    bars2 = plt.bar([i + width/2 for i in x], values_noniid, width, yerr=capped_errors_noniid, capsize=5,
                    label="NON-IID", color="#FF9900", edgecolor="black", linewidth=1.0)
    
    # Adiciona valores nas barras (acima das barras de erro)
    for bars, errors in [(bars1, capped_errors_iid), (bars2, capped_errors_noniid)]:
        for bar, err in zip(bars, errors):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + err + 0.01,
                    f"{height:.2f}", ha="center", va="bottom")
    
    plt.ylabel("Score")
    plt.xticks([i for i in x], labels)
    plt.ylim(0.0, 1.08)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3, axis="y")
    
    if output_dir is None:
        output_dir = FIG_DIR
    
    save_figure_pdf(plt, output_dir, f"fig13_iid_vs_noniid_round_{target_round}")
    plt.close()


def _plot_metric_boxplots_clients_by_round(  # [CLASSIF]
    rounds_data, selected_rounds, metric_key, metric_label, fig_name, output_dir: Path
):  # [CLASSIF]
    """
    [CLASSIF] Gera uma figura com 4 subplots (2x2), um para cada
    intervalo de rounds:

      - [1–10], [11–20], [21–30], [31–40]

    (assumindo que selected_rounds = [10, 20, 30, 40]).

    Em cada subplot:
      - eixo X: clientes (C1, C2, ..., Cn);
      - eixo Y: valores da métrica (F1 ou Accuracy);
      - cada boxplot de cliente usa todos os valores desse cliente
        dentro do intervalo de rounds correspondente.  # [CLASSIF]
    """  # [CLASSIF]
    if not rounds_data or not selected_rounds:  # [CLASSIF]
        return  # [CLASSIF]

    # usamos selected_rounds como últimos rounds de cada intervalo (1-based):
    # 10 -> [1–10], 20 -> [11–20], etc.  # [CLASSIF]
    unique_ends = sorted(set(selected_rounds))[:4]  # [CLASSIF]
    if not unique_ends:  # [CLASSIF]
        return  # [CLASSIF]

    intervals = []  # lista de (start_idx, end_idx, label_str)  # [CLASSIF]
    for end_disp in unique_ends:  # [CLASSIF]
        start_disp = max(1, end_disp - 9)  # 10 rounds por intervalo  # [CLASSIF]
        start_idx = start_disp - 1  # 0-based  # [CLASSIF]
        end_idx = end_disp - 1  # 0-based  # [CLASSIF]
        # label_str = f"Rounds {start_disp}-{end_disp}"  # [CLASSIF]
        label_str = f"Round {end_disp}"  # [CLASSIF]
        intervals.append((start_idx, end_idx, label_str))  # [CLASSIF]

    # número de clientes = máximo comprimento da lista de métricas em qualquer round  # [CLASSIF]
    n_clients = 0  # [CLASSIF]
    for r_metrics in rounds_data.values():  # [CLASSIF]
        n_clients = max(n_clients, len(r_metrics.get(metric_key, [])))  # [CLASSIF]
    if n_clients == 0:  # [CLASSIF]
        return  # [CLASSIF]

    # calcula limites globais de Y para dar "zoom" na faixa relevante  # [CLASSIF]
    global_vals = []  # [CLASSIF]
    for start_idx, end_idx, _ in intervals:  # [CLASSIF]
        for r in range(start_idx, end_idx + 1):  # [CLASSIF]
            metrics_r = rounds_data.get(r)  # [CLASSIF]
            if not metrics_r:  # [CLASSIF]
                continue  # [CLASSIF]
            vals_r = metrics_r.get(metric_key) or []  # [CLASSIF]
            for cid in range(min(n_clients, len(vals_r))):  # [CLASSIF]
                v = vals_r[cid]  # [CLASSIF]
                if v is not None:  # [CLASSIF]
                    global_vals.append(v)  # [CLASSIF]

    if not global_vals:  # [CLASSIF]
        return  # [CLASSIF]

    y_min = min(global_vals)  # [CLASSIF]
    y_max = max(global_vals)  # [CLASSIF]
    if y_min == y_max:  # [CLASSIF]
        y_margin = 0.001  # [CLASSIF]
    else:  # [CLASSIF]
        y_margin = 0.1 * (y_max - y_min)  # [CLASSIF]
    y_lower = max(0.0, y_min - y_margin)  # [CLASSIF]
    y_upper = min(1.0, y_max + y_margin)  # [CLASSIF]

    n_rows, n_cols = 2, 2  # [CLASSIF]
    fig, axes = plt.subplots(  # [CLASSIF]
        n_rows,
        n_cols,
        figsize=(5.0 * n_cols, 4.0 * n_rows),
        sharey=False,
    )  # [CLASSIF]
    axes = axes.flatten()  # [CLASSIF]

    for idx, (start_idx, end_idx, label_str) in enumerate(intervals):  # [CLASSIF]
        if idx >= len(axes):  # [CLASSIF]
            break  # [CLASSIF]
        ax = axes[idx]  # [CLASSIF]

        # constrói série de valores por cliente ao longo do intervalo  # [CLASSIF]
        data = []  # [CLASSIF]
        client_labels = []  # [CLASSIF]
        for cid in range(n_clients):  # [CLASSIF]
            series = []  # [CLASSIF]
            for r in range(start_idx, end_idx + 1):  # [CLASSIF]
                metrics_r = rounds_data.get(r)  # [CLASSIF]
                if not metrics_r:  # [CLASSIF]
                    continue  # [CLASSIF]
                vals_r = metrics_r.get(metric_key) or []  # [CLASSIF]
                if cid < len(vals_r):  # [CLASSIF]
                    v = vals_r[cid]  # [CLASSIF]
                    if v is not None:  # [CLASSIF]
                        series.append(v)  # [CLASSIF]
            if series:  # [CLASSIF]
                data.append(series)  # [CLASSIF]
                client_labels.append(f"C{cid+1}")  # [CLASSIF]

        if not data:  # [CLASSIF]
            ax.set_visible(False)  # [CLASSIF]
            continue  # [CLASSIF]

        ax.boxplot(data, showmeans=False, vert=True)  # [CLASSIF]
        ax.set_xticks(range(1, len(client_labels) + 1))  # [CLASSIF]
        ax.set_xticklabels(client_labels, rotation=45)  # [CLASSIF]
        ax.set_ylabel(f"{metric_label} ({label_str})")  # [CLASSIF]
        # ax.set_title(label_str)  # [CLASSIF]
        ax.set_ylim(y_lower, y_upper)  # [CLASSIF]
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)  # [CLASSIF]

    # esconde subplots sobrando, se houver menos de 4 intervalos  # [CLASSIF]
    for j in range(len(intervals), len(axes)):  # [CLASSIF]
        axes[j].set_visible(False)  # [CLASSIF]

    # fig.suptitle(f"{metric_label} – comparison across clients (round intervals)")  # [CLASSIF]
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # [CLASSIF]
    save_figure_pdf(fig, output_dir, fig_name.replace('.png', ''))  # [CLASSIF]
    plt.close(fig)  # [CLASSIF]

def plot_f1_and_accuracy_boxplots_clients_by_round(  # [CLASSIF]
    rounds_data, selected_rounds, output_dir: Path
):  # [CLASSIF]
    """
    [CLASSIF] Wrapper que gera:
      - uma figura de boxplots para F1 (fig11);
      - uma figura de boxplots para Accuracy (fig12);
    ambas com 4 subplots (rounds 10, 20, 30, 40) comparando clientes.  # [CLASSIF]
    """  # [CLASSIF]
    _plot_metric_boxplots_clients_by_round(  # [CLASSIF]
        rounds_data,
        selected_rounds,
        "f1",
        "F1-score",
        "fig11_f1_clients_box_rounds_10_20_30_40.png",
        output_dir,
    )  # [CLASSIF]

    _plot_metric_boxplots_clients_by_round(  # [CLASSIF]
        rounds_data,
        selected_rounds,
        "acc",
        "Accuracy",
        "fig12_acc_clients_box_rounds_10_20_30_40.png",
        output_dir,
    )  # [CLASSIF]


def _get_class_names_for_confusion(num_classes):  # [CLASSIF]
    """
    [CLASSIF] Recupera os nomes das classes a partir do dataset em uso
    (dataset_path / label_target definidos em fedt.settings).  # [CLASSIF]
    """  # [CLASSIF]
    try:  # [CLASSIF]
        df = pd.read_csv(SETTINGS_DATASET_PATH)  # [CLASSIF]

        label_col = str(SETTINGS_LABEL_TARGET)  # [CLASSIF]
        candidates = []  # [CLASSIF]

        # 1) prioriza colunas mais informativas para rótulos binários  # [CLASSIF]
        if label_col == "Attack_label":  # [CLASSIF]
            for desc_col in ("Attack_type_6", "Attack_type"):  # [CLASSIF]
                if desc_col in df.columns and df[desc_col].nunique(dropna=True) == num_classes:  # [CLASSIF]
                    candidates.append(desc_col)  # [CLASSIF]

        # 2) coluna de rótulo propriamente dita (caso multi-classe)  # [CLASSIF]
        if label_col in df.columns and df[label_col].nunique(dropna=True) == num_classes:  # [CLASSIF]
            candidates.append(label_col)  # [CLASSIF]

        # 3) fallback genérico: outras colunas com a mesma cardinalidade  # [CLASSIF]
        for col in df.columns:  # [CLASSIF]
            if col in candidates:  # [CLASSIF]
                continue  # [CLASSIF]
            if df[col].nunique(dropna=True) == num_classes:  # [CLASSIF]
                candidates.append(col)  # [CLASSIF]

        # 4) extrai nomes ordenados (mesma ordem usada pelo LabelEncoder)  # [CLASSIF]
        for col in candidates:  # [CLASSIF]
            y_series = df[col].astype(str)  # [CLASSIF]
            uniques = sorted(y_series.dropna().unique())  # [CLASSIF]
            if len(uniques) == num_classes:  # [CLASSIF]
                return list(uniques)  # [CLASSIF]

        return None  # [CLASSIF]
    except Exception:  # [CLASSIF]
        return None  # [CLASSIF]


def plot_confusion_matrices_clients(client_files, output_dir: Path):
    """Plota subplots com a matriz de confusão do último round de cada cliente.

    Usa as matrizes já normalizadas salvas nos JSONs.
    """
    cms = []
    labels = []
    json_class_names = None  # [CLASSIF]

    for path in client_files:
        if not path.exists():
            continue
        data = load_json(path)
        if not data:
            continue
        round_ids = [int(r) for r in data.keys()]
        r_last = max(round_ids)
        m_last = data[str(r_last)]  # [CLASSIF]
        cm = m_last.get("confusion_matrix")
        if cm is None:
            continue

        # [CLASSIF] Tenta obter nomes de classes diretamente do JSON (se existirem)
        if json_class_names is None:  # [CLASSIF]
            names_from_json = m_last.get("confusion_matrix_labels")  # [CLASSIF]
            if isinstance(names_from_json, list):  # [CLASSIF]
                json_class_names = names_from_json  # [CLASSIF]

        cms.append(cm)
        parent_name = path.parent.name
        
        # No layout .../client-id-X/<dataset>/arquivo.json, o parent é o dataset.
        # Então subimos um nível para capturar "client-id-X".
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

    # [CLASSIF] Prioriza nomes vindos do JSON; se não houver, usa heurística anterior
    if isinstance(json_class_names, list) and len(json_class_names) == num_classes:  # [CLASSIF]
        class_names = json_class_names  # [CLASSIF]
    else:  # [CLASSIF]
        class_names = _get_class_names_for_confusion(num_classes)  # [CLASSIF]
    middle_row = num_classes // 2  # [CLASSIF]

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
        ax.set_xticks(range(num_classes))  # [CLASSIF]
        ax.set_yticks(range(num_classes))  # [CLASSIF]
        if class_names is not None:  # [CLASSIF]
            ax.set_xticklabels(
                class_names, rotation=45, ha="right",
                fontsize=CONFUSION_MATRIX_TICK_LABEL_SIZE, fontweight="bold"
            )  # [CLASSIF]
            ax.set_yticklabels(
                class_names, fontsize=CONFUSION_MATRIX_TICK_LABEL_SIZE, fontweight="bold"
            )  # [CLASSIF]

        # escreve o valor numérico em cada célula  # [CLASSIF]
        for i in range(num_classes):  # [CLASSIF]
            for j in range(num_classes):  # [CLASSIF]
                val = cm[i][j]  # [CLASSIF]
                # diagonal (classe correta) em preto, demais em branco  # [CLASSIF]
                text_color = "black" if i == j else "white"  # [CLASSIF]
                ax.text(  # [CLASSIF]
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=CONFUSION_MATRIX_TEXT_SIZE,
                )  # [CLASSIF]

    # esconder eixos sobrando
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

    # fig.suptitle("Confusion Matrices (last round of each client)")
    save_figure_pdf(fig, output_dir, "fig8_confusion_matrices_clients")
    plt.close(fig)


# ==========================
# MAIN
# ==========================

def main():
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

    plot_f1_and_accuracy(rounds, f1_mean, f1_std, acc_mean, acc_std, FIG_DIR)
    plot_f1_and_accuracy_per_client_and_mean(CLIENT_FILES, FIG_DIR)
    plot_round_times(rounds, train_mean, train_std, server_rounds, aggregation_time, FIG_DIR)
    plot_memory_over_time(load_cpu_ram_series(CPU_FILE), FIG_DIR)
    plot_inference_time(rounds, infer_mean, infer_std, FIG_DIR)
    if cpu_rounds:
        plot_cpu_usage_per_round(cpu_rounds, client_cpu_per_round, server_cpu_per_round, FIG_DIR)
    plot_last_round_metrics_bar(CLIENT_FILES, FIG_DIR)
    plot_iid_vs_noniid_metrics_comparison(CLIENT_FILES, CLIENT_FILES_NONIID, target_round=40, output_dir=FIG_DIR)
    plot_confusion_matrices_clients(CLIENT_FILES, FIG_DIR)
    plot_f1_and_accuracy_boxplots_clients_by_round(rounds_data, [10, 20, 30, 40], FIG_DIR)  # [CLASSIF]


if __name__ == "__main__":
    main()
