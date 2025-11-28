import json
from pathlib import Path
from statistics import mean, pstdev
import matplotlib.pyplot as plt


# ==========================
# CONFIGURAÇÃO DE ARQUIVOS
# ==========================

# Arquivos dos clientes
CLIENT_FILES = [
    Path("/home/yuri/FEDT_IDS2/results/best_trees/client-id-0/best_trees_client-id-0_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/client-id-1/best_trees_client-id-1_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/client-id-2/best_trees_client-id-1_1.json"),
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
    """
    Lê os JSONs dos clientes e devolve um dicionário:
    rounds_data[round_id:int] = {
        "f1": [f1_c0, f1_c1, ...],
        "acc": [acc_c0, acc_c1, ...],
        "round_time": [rt_c0, rt_c1, ...],
        "round_start": [t0_c0, t0_c1, ...],
        "round_end": [t1_c0, t1_c1, ...],
    }
    """
    rounds_data = {}

    for path in client_files:
        data = load_json(path)
        for round_id_str, metrics in data.items():
            round_id = int(round_id_str)
            if round_id not in rounds_data:
                rounds_data[round_id] = {
                    "f1": [],
                    "acc": [],
                    "round_time": [],
                    "round_start": [],
                    "round_end": [],
                }

            rounds_data[round_id]["f1"].append(metrics["f1_score"])
            rounds_data[round_id]["acc"].append(metrics["accuracy"])
            rounds_data[round_id]["round_time"].append(metrics["round_time"])
            rounds_data[round_id]["round_start"].append(metrics["round_start_time"])
            rounds_data[round_id]["round_end"].append(metrics["round_end_time"])

    return rounds_data


def summarize_rounds(rounds_data):
    """
    A partir de rounds_data, calcula média e desvio padrão
    para F1, acurácia e round_time.
    Retorna listas ordenadas por round (int):
      rounds_int, f1_mean, f1_std, acc_mean, acc_std, rt_mean, rt_std.
    """
    rounds_int = sorted(rounds_data.keys())

    f1_mean, f1_std = [], []
    acc_mean, acc_std = [], []
    rt_mean, rt_std = [], []

    for r in rounds_int:
        f1_vals = rounds_data[r]["f1"]
        acc_vals = rounds_data[r]["acc"]
        rt_vals = rounds_data[r]["round_time"]

        f1_mean.append(mean(f1_vals))
        acc_mean.append(mean(acc_vals))
        rt_mean.append(mean(rt_vals))

        f1_std.append(pstdev(f1_vals) if len(f1_vals) > 1 else 0.0)
        acc_std.append(pstdev(acc_vals) if len(acc_vals) > 1 else 0.0)
        rt_std.append(pstdev(rt_vals) if len(rt_vals) > 1 else 0.0)

    return rounds_int, f1_mean, f1_std, acc_mean, acc_std, rt_mean, rt_std


def load_server_metrics(path: Path):
    """
    Lê o JSON do servidor, esperado no formato:
    {
      "0": {
        "trees_by_client": ...,
        "aggregation_time": ...,
        "avg_execution_time": ...
      },
      ...
    }
    """
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
    """
    Lê o JSON de monitoramento de CPU/RAM no formato:
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
    data = load_json(path)

    series = {}

    def pick_first_and_build(target_key, label):
        if target_key not in data:
            return
        pid_to_samples = data[target_key]
        if not pid_to_samples:
            return
        # pega o primeiro PID em ordem crescente
        first_pid = sorted(pid_to_samples.keys(), key=int)[0]
        samples = pid_to_samples[first_pid]
        t = [s["timestamp"] for s in samples]
        cpu = [s["cpu_percent"] for s in samples]
        mem = [s["memory_mb"] for s in samples]
        series[label] = {"t": t, "cpu": cpu, "mem": mem}

    # Clients (usa o primeiro client encontrado)
    pick_first_and_build("--client-id", "client")
    # Servidor
    pick_first_and_build("fedt run server", "server")

    # Normaliza o tempo para começar em zero (mínimo entre todas as séries)
    if series:
        t_min = min(min(v["t"]) for v in series.values())
        for v in series.values():
            v["t"] = [x - t_min for x in v["t"]]

    return series


# ==========================
# FUNÇÕES DE PLOT
# ==========================

def plot_f1_and_accuracy(rounds, f1_mean, f1_std, acc_mean, acc_std, output_dir: Path):
    # converter rounds 0,1,2 -> 1,2,3 no eixo x
    round_labels = [r + 1 for r in rounds]

    # Figura 1: F1 médio por round
    plt.figure()
    plt.errorbar(round_labels, f1_mean, yerr=f1_std, fmt="-o", capsize=5)
    plt.xlabel("Round")
    plt.ylabel("F1")
    plt.title("F1")
    plt.grid(True)
    plt.savefig(output_dir / "fig1_f1_medio_por_round.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figura 2: Acurácia média por round
    plt.figure()
    plt.errorbar(round_labels, acc_mean, yerr=acc_std, fmt="-o", capsize=5)
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.grid(True)
    plt.savefig(output_dir / "fig2_acuracia_media_por_round.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_round_times(rounds, rt_mean, rt_std, server_rounds, aggregation_time, output_dir: Path):
    # rounds clientes: 0,1,2 -> 1,2,3
    round_labels = [r + 1 for r in rounds]
    # rounds servidor: 0,1,2 -> 1,2,3 (se existir)
    server_labels = [r + 1 for r in server_rounds] if server_rounds else []

    # Figura 3: Tempo por round
    plt.figure()
    plt.errorbar(round_labels, rt_mean, yerr=rt_std, fmt="-o", capsize=5,
                 label="Round Time")
    if server_rounds and aggregation_time:
        plt.plot(server_labels, aggregation_time, "-s", label="Aggregation Time (server)")
    plt.xlabel("Round")
    plt.ylabel("Tempo (s)")
    plt.title("Tempos por round")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "fig3_tempos_por_round.png", dpi=300, bbox_inches="tight")
    plt.close()



def plot_round_times(rounds, rt_mean, rt_std, server_rounds, aggregation_time, output_dir: Path):
    # rounds clientes: 0,1,2 -> 1,2,3
    round_labels = [r + 1 for r in rounds]
    # rounds servidor: 0,1,2 -> 1,2,3 (se existir)
    server_labels = [r + 1 for r in server_rounds] if server_rounds else []

    # Figura 3: Tempo por round
    plt.figure()
    plt.errorbar(round_labels, rt_mean, yerr=rt_std, fmt="-o", capsize=5,
                 label="Round Time")
    if server_rounds and aggregation_time:
        plt.plot(server_labels, aggregation_time, "-s", label="Aggregation Time (server)")
    plt.xlabel("Round")
    plt.ylabel("Time (s)")
    plt.title("Times per round")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "fig3_tempos_por_round.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_cpu_and_memory(series, output_dir: Path):
    if not series:
        return

    # Figura 4: CPU vs tempo
    plt.figure()
    for label, v in series.items():
        plt.plot(v["t"], v["cpu"], label=label)
    plt.xlabel("Time (s)")
    plt.ylabel("CPU usage (%)")
    plt.title("CPU usage over time")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "fig4_cpu_vs_tempo.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figura 5: Memória vs tempo
    plt.figure()
    for label, v in series.items():
        plt.plot(v["t"], v["mem"], label=label)
    plt.xlabel("Time (s)")
    plt.ylabel("Memoty (MB)")
    plt.title("Memory usage over time")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "fig5_memoria_vs_tempo.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_last_round_metrics_bar(client_files, output_dir: Path):
    """
    Cria um gráfico de barras único com a MÉDIA entre os clientes
    de acurácia, recall, precisão e f1-score no ÚLTIMO round.
    """
    from statistics import mean

    acc_vals, prec_vals, rec_vals, f1_vals = [], [], [], []
    last_round = None

    for path in client_files:
        data = load_json(path)
        # pega o último round disponível nesse cliente
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

    acc_mean = mean(acc_vals)
    prec_mean = mean(prec_vals)
    rec_mean = mean(rec_vals)
    f1_mean = mean(f1_vals)

    acc_std = pstdev(acc_vals) if len(acc_vals) > 1 else 0.0
    prec_std = pstdev(prec_vals) if len(prec_vals) > 1 else 0.0
    rec_std = pstdev(rec_vals) if len(rec_vals) > 1 else 0.0
    f1_std = pstdev(f1_vals) if len(f1_vals) > 1 else 0.0

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
        edgecolor="black",   # contorno preto
        linewidth=1.0        # espessura do contorno
    )
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    if last_round is not None:
        plt.title(f"Metrics (round {last_round + 1})")
    else:
        plt.title("Métrics")
    plt.grid(axis="y")
    plt.savefig(output_dir / "fig6_metricas_medias_ultimo_round.png", dpi=300, bbox_inches="tight")
    plt.close()

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
        rt_mean,
        rt_std,
    ) = summarize_rounds(rounds_data)

    # 2) Métricas do servidor
    if SERVER_FILE.exists():
        server_rounds, trees_by_client, aggregation_time, avg_exec_time = load_server_metrics(SERVER_FILE)
    else:
        server_rounds, aggregation_time = [], []

    # 3) Séries de CPU/RAM
    if CPU_FILE.exists():
        cpu_series = load_cpu_ram_series(CPU_FILE)
    else:
        cpu_series = {}

    # 4) Plots (salvando em FIG_DIR)
    plot_f1_and_accuracy(rounds, f1_mean, f1_std, acc_mean, acc_std, FIG_DIR)
    plot_round_times(rounds, rt_mean, rt_std, server_rounds, aggregation_time, FIG_DIR)
    plot_cpu_and_memory(cpu_series, FIG_DIR)
    plot_last_round_metrics_bar(CLIENT_FILES, FIG_DIR)


if __name__ == "__main__":
    main()
