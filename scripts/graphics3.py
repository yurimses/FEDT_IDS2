import json
from pathlib import Path
from statistics import mean, pstdev
import math
import matplotlib.pyplot as plt


# ==========================
# CONFIGURAÇÃO DE ARQUIVOS
# ==========================

# Arquivos dos clientes
CLIENT_FILES = [
    Path("/home/yuri/FEDT_IDS2/results/best_trees/client-id-0/best_trees_client-id-0_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/client-id-1/best_trees_client-id-1_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/client-id-2/best_trees_client-id-2_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/client-id-3/best_trees_client-id-3_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/client-id-4/best_trees_client-id-4_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/client-id-5/best_trees_client-id-5_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/client-id-6/best_trees_client-id-6_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/client-id-7/best_trees_client-id-7_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/client-id-8/best_trees_client-id-8_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/client-id-9/best_trees_client-id-9_1.json"),
]

# Arquivo do servidor
SERVER_FILE = Path("/home/yuri/FEDT_IDS2/results/best_trees/server/best_trees_server_1.json")

# Arquivo de monitoramento de CPU/RAM
CPU_FILE = Path("/home/yuri/FEDT_IDS2/logs/cpu_ram/best_trees/cpu_and_ram_yuri_best_trees_0.json")

# Pasta onde as figuras serão salvas
FIG_DIR = Path("/home/yuri/FEDT_IDS2/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ==========================
# FUNÇÕES AUXILIARES
# ==========================

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_client_metrics(client_files):
    """Lê os JSONs dos clientes e devolve um dicionário:

    rounds_data[round_id:int] = {
        "f1": [f1_c0, f1_c1, ...],
        "acc": [acc_c0, acc_c1, ...],
        "round_time": [rt_c0, rt_c1, ...],
        "fit_time": [ft_c0, ft_c1, ...],
        "inference_time": [it_c0, it_c1, ...],
        "round_start": [t0_c0, t0_c1, ...],
        "round_end": [t1_c0, t1_c1, ...],
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
    """Lê o JSON de monitoramento de CPU/RAM no formato:

    {
      "--client-id": {
        "PID1": [ {timestamp, cpu_percent, memory_mb, num_threads}, ... ],
        "PID2": [ ... ]
      },
      "fedt run server": {
        "PID3": [ ... ]
      }
    }

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

    def pick_first_and_build(target_key, label):
        if target_key not in data:
            return
        pid_to_samples = data[target_key]
        if not pid_to_samples:
            return
        first_pid = sorted(pid_to_samples.keys(), key=int)[0]
        samples = pid_to_samples[first_pid]
        t = [s["timestamp"] for s in samples]
        cpu = [s["cpu_percent"] for s in samples]
        mem = [s["memory_mb"] for s in samples]
        series[label] = {"t": t, "cpu": cpu, "mem": mem}

    # primeiro cliente encontrado
    pick_first_and_build("--client-id", "client")
    # servidor
    pick_first_and_build("fedt run server", "server")

    # normalizar tempo para começar em zero
    if series:
        t_min = min(min(v["t"]) for v in series.values())
        for v in series.values():
            v["t"] = [x - t_min for x in v["t"]]

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
    # converter rounds 0,1,2 -> 1,2,3 no eixo x
    if not rounds:
        return

    round_labels = [r + 1 for r in rounds]

    # F1 médio por round
    plt.figure()
    plt.errorbar(round_labels, f1_mean, yerr=f1_std, fmt="-o", capsize=5)
    plt.xlabel("Round")
    plt.ylabel("F1")
    plt.title("F1")
    plt.savefig(output_dir / "fig1_f1_mean.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Acurácia média por round
    plt.figure()
    plt.errorbar(round_labels, acc_mean, yerr=acc_std, fmt="-o", capsize=5)
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.savefig(output_dir / "fig2_accuracy_mean.png", dpi=300, bbox_inches="tight")
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
    plt.title("Client training and server aggregation time per round")
    plt.legend()
    plt.savefig(output_dir / "fig3_client_train_and_server_agg_time.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_cpu_and_memory(series, output_dir: Path):
    if not series:
        return

    # CPU do cliente ao longo do tempo
    if "client" in series:
        v_client = series["client"]
        plt.figure()
        plt.plot(v_client["t"], v_client["cpu"])
        plt.xlabel("Time (s)")
        plt.ylabel("CPU usage (%)")
        plt.title("Client CPU usage over time")
        plt.savefig(output_dir / "fig4_cpu_client_vs_time.png", dpi=300, bbox_inches="tight")
        plt.close()

    # CPU do servidor ao longo do tempo
    if "server" in series:
        v_server = series["server"]
        plt.figure()
        plt.plot(v_server["t"], v_server["cpu"])
        plt.xlabel("Time (s)")
        plt.ylabel("CPU usage (%)")
        plt.title("Server CPU usage over time")
        plt.savefig(output_dir / "fig4_cpu_server_vs_time.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Memória vs tempo (cliente e servidor)
    plt.figure()
    for label, v in series.items():
        plt.plot(v["t"], v["mem"], label=label)
    plt.xlabel("Time (s)")
    plt.ylabel("Memory (MB)")
    plt.title("Memory usage over time")
    plt.legend()
    plt.savefig(output_dir / "fig5_memory_vs_time.png", dpi=300, bbox_inches="tight")
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
    plt.title("Inference time per round (mean clients)")
    plt.savefig(output_dir / "fig6_inference_time_per_round.png", dpi=300, bbox_inches="tight")
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
    plt.title("CPU usage per round")
    plt.legend()
    plt.savefig(output_dir / "fig7_cpu_usage_per_round.png", dpi=300, bbox_inches="tight")
    plt.close()


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

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    plt.figure()
    plt.bar(
        x,
        values,
        yerr=errors,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=1.0,
    )
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    if last_round is not None:
        plt.title(f"Metrics (round {last_round + 1})")
    else:
        plt.title("Metrics")
    plt.savefig(output_dir / "fig8_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices_clients(client_files, output_dir: Path):
    """Plota subplots com a matriz de confusão do último round de cada cliente.

    Usa as matrizes já normalizadas salvas nos JSONs.
    """
    cms = []
    labels = []

    for path in client_files:
        if not path.exists():
            continue
        data = load_json(path)
        if not data:
            continue
        round_ids = [int(r) for r in data.keys()]
        r_last = max(round_ids)
        cm = data[str(r_last)].get("confusion_matrix")
        if cm is None:
            continue

        cms.append(cm)
        parent_name = path.parent.name  # ex.: 'client-id-0'
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

    # aumenta um pouco a altura para caber melhor 2 linhas
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))

    # organiza axes como matriz 2D
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    num_classes = len(cms[0])

    ims = []
    for idx, cm in enumerate(cms):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        im = ax.imshow(cm, vmin=0.0, vmax=1.0)
        ims.append(im)
        ax.set_title(labels[idx])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))

    # esconder eixos sobrando (caso não feche certinho linhas x colunas)
    for idx in range(len(cms), rows * cols):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        ax.axis("off")

    # ajusta espaçamento entre linhas/colunas e reserva espaço à direita p/ colorbar
    plt.subplots_adjust(hspace=0.5, wspace=0.3, right=0.88)

    # colorbar em um eixo separado na lateral direita
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(ims[-1], cax=cbar_ax)

    fig.suptitle("Confusion Matrices (last round of each client)")
    fig.savefig(output_dir / "fig9_confusion_matrices_clients.png", dpi=300, bbox_inches="tight")
    plt.close(fig)



# ==========================
# MAIN
# ==========================

def main():
    # 1) Métricas dos clientes
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

    # 2) Métricas do servidor
    server_rounds, trees_by_client, aggregation_time, avg_exec_time = load_server_metrics(SERVER_FILE)

    # 3) Séries de CPU/RAM (para gráficos no tempo)
    cpu_series = load_cpu_ram_series(CPU_FILE)

    # 4) CPU por round (clientes e servidor)
    cpu_rounds, client_cpu_per_round, server_cpu_per_round = compute_cpu_usage_per_round(rounds_data, CPU_FILE)

    # 5) Plots
    plot_f1_and_accuracy(rounds, f1_mean, f1_std, acc_mean, acc_std, FIG_DIR)
    plot_round_times(rounds, train_mean, train_std, server_rounds, aggregation_time, FIG_DIR)
    plot_cpu_and_memory(cpu_series, FIG_DIR)
    plot_inference_time(rounds, infer_mean, infer_std, FIG_DIR)
    if cpu_rounds:
        plot_cpu_usage_per_round(cpu_rounds, client_cpu_per_round, server_cpu_per_round, FIG_DIR)
    plot_last_round_metrics_bar(CLIENT_FILES, FIG_DIR)
    plot_confusion_matrices_clients(CLIENT_FILES, FIG_DIR)


if __name__ == "__main__":
    main()
