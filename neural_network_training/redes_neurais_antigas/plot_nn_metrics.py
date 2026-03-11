#!/usr/bin/env python3
"""
Script para gerar gráficos das métricas médias dos clientes por round
a partir dos resultados do treinamento federated neural network.

Uso:
    python plot_nn_metrics.py --run-dir results/ML-EdgeIIoT-FEDT-1
"""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# ===========================
# Configuração de Caminhos
# ===========================
# Diretório raiz do projeto
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Caminho padrão para os resultados do treinamento (entrada)
DEFAULT_RUN_DIR = PROJECT_ROOT / "neural_network_training" / "results" / "ML-EdgeIIoT-FEDT-1"

# Caminho padrão para salvar os gráficos (saída)
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "figures"

# ===========================
# Configuração de fontes
# ===========================
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


# ===========================
# Funções auxiliares
# ===========================

def load_json(path: Path) -> dict:
    """Carrega JSON de um arquivo."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_figure_pdf(fig_or_plt, output_dir: Path, filename: str):
    """Salva figura em PDF de alta resolução (300 DPI)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{filename}.pdf"

    if hasattr(fig_or_plt, 'savefig'):
        fig_or_plt.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
    
    print(f"  Saved: {filepath}")


def find_client_json_files(run_dir: Path) -> list:
    """
    Encontra todos os JSONs de clientes na estrutura:
    run_dir/<partition_tag>/client-id-<id>/nn_client-id-<id>.json
    """
    client_files = []
    
    # Procura por pastas client-id-*
    for partition_dir in run_dir.iterdir():
        if not partition_dir.is_dir() or partition_dir.name.startswith("_"):
            continue
        
        for client_dir in partition_dir.iterdir():
            if not client_dir.is_dir() or not client_dir.name.startswith("client-id-"):
                continue
            
            # Procura pelo arquivo nn_client-id-*.json
            json_files = list(client_dir.glob("nn_client-id-*.json"))
            if json_files:
                client_files.append(json_files[0])
    
    return sorted(client_files)


def aggregate_client_metrics(client_files: list) -> dict:
    """
    Lê JSONs dos clientes e agrega métricas por round.
    
    Retorna:
        rounds_data[round_id:int] = {
            "f1": [f1_c0, f1_c1, ...],
            "acc": [acc_c0, acc_c1, ...],
            "prec": [prec_c0, prec_c1, ...],
            "rec": [rec_c0, rec_c1, ...],
            "mcc": [mcc_c0, mcc_c1, ...],
            "fit_time": [fit_c0, fit_c1, ...],
            "eval_time": [eval_c0, eval_c1, ...],
            "inference_time": [inf_c0, inf_c1, ...],
            "test_loss": [loss_c0, loss_c1, ...],
            "train_loss": [loss_c0, loss_c1, ...],
            "round_time": [rt_c0, rt_c1, ...],
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
                    "prec": [],
                    "rec": [],
                    "mcc": [],
                    "fit_time": [],
                    "eval_time": [],
                    "inference_time": [],
                    "test_loss": [],
                    "train_loss": [],
                    "round_time": [],
                }

            rounds_data[round_id]["f1"].append(metrics.get("f1_score", 0.0))
            rounds_data[round_id]["acc"].append(metrics.get("accuracy", 0.0))
            rounds_data[round_id]["prec"].append(metrics.get("precision", 0.0))
            rounds_data[round_id]["rec"].append(metrics.get("recall", 0.0))
            rounds_data[round_id]["mcc"].append(metrics.get("mcc", 0.0))
            rounds_data[round_id]["fit_time"].append(metrics.get("fit_time", 0.0))
            rounds_data[round_id]["eval_time"].append(metrics.get("evaluate_time", 0.0))
            rounds_data[round_id]["inference_time"].append(metrics.get("inference_time", 0.0))
            rounds_data[round_id]["test_loss"].append(metrics.get("test_loss", 0.0))
            rounds_data[round_id]["train_loss"].append(metrics.get("train_loss", 0.0))
            rounds_data[round_id]["round_time"].append(metrics.get("round_time", 0.0))

    return rounds_data


def summarize_rounds(rounds_data: dict):
    """
    Calcula média e desvio padrão das métricas por round.
    
    Retorna tupla de listas (ordenadas por round):
        (rounds_int, f1_mean, f1_std, acc_mean, acc_std, 
         prec_mean, prec_std, rec_mean, rec_std,
         fit_mean, fit_std, eval_mean, eval_std,
         infer_mean, infer_std, train_loss_mean, train_loss_std,
         test_loss_mean, test_loss_std)
    """
    if not rounds_data:
        return ([], [], [], [], [], [], [], [], [], [], [], [], [], 
                [], [], [], [], [], [])

    rounds_int = sorted(rounds_data.keys())

    f1_mean, f1_std = [], []
    acc_mean, acc_std = [], []
    prec_mean, prec_std = [], []
    rec_mean, rec_std = [], []
    fit_mean, fit_std = [], []
    eval_mean, eval_std = [], []
    infer_mean, infer_std = [], []
    train_loss_mean, train_loss_std = [], []
    test_loss_mean, test_loss_std = [], []

    for r in rounds_int:
        metrics = rounds_data[r]
        
        for vals, means, stds in [
            (metrics["f1"], f1_mean, f1_std),
            (metrics["acc"], acc_mean, acc_std),
            (metrics["prec"], prec_mean, prec_std),
            (metrics["rec"], rec_mean, rec_std),
            (metrics["fit_time"], fit_mean, fit_std),
            (metrics["eval_time"], eval_mean, eval_std),
            (metrics["inference_time"], infer_mean, infer_std),
            (metrics["train_loss"], train_loss_mean, train_loss_std),
            (metrics["test_loss"], test_loss_mean, test_loss_std),
        ]:
            if vals:
                means.append(mean(vals))
                stds.append(pstdev(vals) if len(vals) > 1 else 0.0)
            else:
                means.append(0.0)
                stds.append(0.0)

    return (
        rounds_int,
        f1_mean, f1_std,
        acc_mean, acc_std,
        prec_mean, prec_std,
        rec_mean, rec_std,
        fit_mean, fit_std,
        eval_mean, eval_std,
        infer_mean, infer_std,
        train_loss_mean, train_loss_std,
        test_loss_mean, test_loss_std,
    )


# ===========================
# Funções de plotagem
# ===========================

def plot_f1_and_accuracy(rounds, f1_mean, f1_std, acc_mean, acc_std, output_dir: Path):
    """Plota F1 e Acurácia médias por round."""
    if not rounds:
        return

    round_labels = [r + 1 for r in rounds]

    # F1 médio por round
    print("Plotting F1...")
    plt.figure(figsize=(8, 5))
    plt.errorbar(round_labels, f1_mean, yerr=f1_std, fmt="-o", capsize=5)
    plt.xlabel("Round")
    plt.ylabel("F1-Score")
    plt.grid(True, linestyle="--", alpha=0.3)
    save_figure_pdf(plt, output_dir, "fig1_f1_mean")
    plt.close()

    # Acurácia média por round
    print("Plotting Accuracy...")
    plt.figure(figsize=(8, 5))
    plt.errorbar(round_labels, acc_mean, yerr=acc_std, fmt="-o", capsize=5)
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.3)
    save_figure_pdf(plt, output_dir, "fig2_accuracy_mean")
    plt.close()


def plot_precision_recall(rounds, prec_mean, prec_std, rec_mean, rec_std, output_dir: Path):
    """Plota Precision e Recall médios por round."""
    if not rounds:
        return

    round_labels = [r + 1 for r in rounds]

    print("Plotting Precision...")
    plt.figure(figsize=(8, 5))
    plt.errorbar(round_labels, prec_mean, yerr=prec_std, fmt="-o", capsize=5)
    plt.xlabel("Round")
    plt.ylabel("Precision")
    plt.grid(True, linestyle="--", alpha=0.3)
    save_figure_pdf(plt, output_dir, "fig3_precision_mean")
    plt.close()

    print("Plotting Recall...")
    plt.figure(figsize=(8, 5))
    plt.errorbar(round_labels, rec_mean, yerr=rec_std, fmt="-o", capsize=5)
    plt.xlabel("Round")
    plt.ylabel("Recall")
    plt.grid(True, linestyle="--", alpha=0.3)
    save_figure_pdf(plt, output_dir, "fig4_recall_mean")
    plt.close()


def plot_losses(rounds, train_loss_mean, train_loss_std, test_loss_mean, test_loss_std, output_dir: Path):
    """Plota Loss de treinamento e teste."""
    if not rounds:
        return

    round_labels = [r + 1 for r in rounds]

    print("Plotting Losses...")
    plt.figure(figsize=(10, 5))
    plt.errorbar(round_labels, train_loss_mean, yerr=train_loss_std, 
                 fmt="-o", capsize=5, label="Train Loss")
    plt.errorbar(round_labels, test_loss_mean, yerr=test_loss_std, 
                 fmt="-s", capsize=5, label="Test Loss")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    save_figure_pdf(plt, output_dir, "fig5_losses")
    plt.close()


def plot_times(rounds, fit_mean, fit_std, eval_mean, eval_std, infer_mean, infer_std, output_dir: Path):
    """Plota tempos de fit, avaliação e inferência."""
    if not rounds:
        return

    round_labels = [r + 1 for r in rounds]

    print("Plotting Times...")
    plt.figure(figsize=(10, 5))
    if any(fit_mean):
        plt.errorbar(round_labels, fit_mean, yerr=fit_std, 
                     fmt="-o", capsize=5, label="Fit Time")
    if any(eval_mean):
        plt.errorbar(round_labels, eval_mean, yerr=eval_std, 
                     fmt="-s", capsize=5, label="Evaluate Time")
    if any(infer_mean):
        plt.errorbar(round_labels, infer_mean, yerr=infer_std, 
                     fmt="-^", capsize=5, label="Inference Time")
    plt.xlabel("Round")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    save_figure_pdf(plt, output_dir, "fig6_times")
    plt.close()


def plot_all_metrics_per_client(client_files, output_dir: Path):
    """Plota F1, Accuracy, Precision e Recall por cliente e média geral."""
    client_metrics = {}
    f1_per_round = defaultdict(list)
    acc_per_round = defaultdict(list)
    prec_per_round = defaultdict(list)
    rec_per_round = defaultdict(list)

    for path in client_files:
        if not path.exists():
            continue
        
        data = load_json(path)
        if not data:
            continue

        # Extrai nome do cliente
        parent = path.parent.name
        if "client-id-" in parent:
            cid = parent.split("client-id-")[-1]
            label = f"Client {cid}"
        else:
            label = parent

        round_ids = sorted(int(r) for r in data.keys())
        rounds_client = []
        f1_client = []
        acc_client = []
        prec_client = []
        rec_client = []

        for r in round_ids:
            m = data[str(r)]
            rounds_client.append(r)
            f1_client.append(m.get("f1_score", 0.0))
            acc_client.append(m.get("accuracy", 0.0))
            prec_client.append(m.get("precision", 0.0))
            rec_client.append(m.get("recall", 0.0))
            
            f1_per_round[r].append(m.get("f1_score", 0.0))
            acc_per_round[r].append(m.get("accuracy", 0.0))
            prec_per_round[r].append(m.get("precision", 0.0))
            rec_per_round[r].append(m.get("recall", 0.0))

        client_metrics[label] = {
            "rounds": rounds_client,
            "f1": f1_client,
            "acc": acc_client,
            "prec": prec_client,
            "rec": rec_client,
        }

    if not client_metrics:
        return

    all_rounds = sorted(f1_per_round.keys())
    mean_f1 = [mean(f1_per_round[r]) for r in all_rounds]
    mean_acc = [mean(acc_per_round[r]) for r in all_rounds]
    mean_prec = [mean(prec_per_round[r]) for r in all_rounds]
    mean_rec = [mean(rec_per_round[r]) for r in all_rounds]
    round_labels_mean = [r + 1 for r in all_rounds]

    # F1 por cliente + média
    print("Plotting F1 per client and mean...")
    plt.figure(figsize=(12, 6))
    for label, m in sorted(client_metrics.items()):
        x = [r + 1 for r in m["rounds"]]
        plt.plot(x, m["f1"], marker="o", alpha=0.5, linewidth=1.0, label=label)

    plt.plot(
        round_labels_mean,
        mean_f1,
        marker="o",
        color="black",
        linewidth=2.5,
        label="Mean (clients)",
    )
    plt.xlabel("Round")
    plt.ylabel("F1-Score")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.3)
    save_figure_pdf(plt, output_dir, "fig7_f1_clients_and_mean")
    plt.close()

    # Accuracy por cliente + média
    print("Plotting Accuracy per client and mean...")
    plt.figure(figsize=(12, 6))
    for label, m in sorted(client_metrics.items()):
        x = [r + 1 for r in m["rounds"]]
        plt.plot(x, m["acc"], marker="o", alpha=0.5, linewidth=1.0, label=label)

    plt.plot(
        round_labels_mean,
        mean_acc,
        marker="o",
        color="black",
        linewidth=2.5,
        label="Mean (clients)",
    )
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.3)
    save_figure_pdf(plt, output_dir, "fig8_accuracy_clients_and_mean")
    plt.close()


def plot_metrics_bar(client_files, output_dir: Path):
    """Gráfico de barras com F1, Accuracy, Precision e Recall do último round."""
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
        acc_vals.append(m.get("accuracy", 0.0))
        prec_vals.append(m.get("precision", 0.0))
        rec_vals.append(m.get("recall", 0.0))
        f1_vals.append(m.get("f1_score", 0.0))

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

    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    values = [acc_mean, prec_mean, rec_mean, f1_mean]
    errors = [acc_std, prec_std, rec_std, f1_std]
    x = range(len(labels))

    colors = ["#07a791", "#FF9900", "#ff000086", "#A864AA"]

    print("Plotting metrics bar chart...")
    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        x,
        values,
        yerr=errors,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=1.0,
        width=0.6,
    )
    for bar, value, err in zip(bars, values, errors):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + err + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.08)
    plt.ylabel("Score")
    plt.grid(True, linestyle="--", alpha=0.3, axis="y")
    save_figure_pdf(plt, output_dir, "fig9_metrics_last_round")
    plt.close()


def plot_all_metrics_together(rounds, acc_mean, prec_mean, rec_mean, f1_mean, output_dir: Path):
    """Plota todas as 4 métricas principais no mesmo gráfico."""
    if not rounds:
        return

    round_labels = [r + 1 for r in rounds]

    print("Plotting all metrics together...")
    plt.figure(figsize=(12, 6))
    
    plt.plot(round_labels, acc_mean, marker="o", linewidth=2.0, label="Accuracy", color="#07a791")
    plt.plot(round_labels, prec_mean, marker="s", linewidth=2.0, label="Precision", color="#FF9900")
    plt.plot(round_labels, rec_mean, marker="^", linewidth=2.0, label="Recall", color="#ff6b6b")
    plt.plot(round_labels, f1_mean, marker="D", linewidth=2.0, label="F1-Score", color="#A864AA")
    
    plt.xlabel("Round", fontsize=14, fontweight="bold")
    plt.ylabel("Score", fontsize=14, fontweight="bold")
    plt.legend(fontsize=12, loc="best")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.ylim(0.75, 1.0)  # Zoom na faixa relevante
    
    save_figure_pdf(plt, output_dir, "fig0_all_metrics_together")
    plt.close()


def plot_boxplots_by_round_intervals(rounds_data: dict, selected_rounds: list, output_dir: Path):
    """
    Gera boxplots de F1 e Accuracy por intervalo de rounds.
    
    Cria uma figura com 4 subplots (2x2), um para cada intervalo de rounds:
    - [1–10], [11–20], [21–30], [31–40]
    
    Cada boxplot mostra a distribuição de valores para cada cliente.
    """
    if not rounds_data or not selected_rounds:
        return

    # Usar selected_rounds como últimos rounds de cada intervalo
    unique_ends = sorted(set(selected_rounds))[:4]
    if not unique_ends:
        return

    intervals = []
    for end_disp in unique_ends:
        start_disp = max(1, end_disp - 9)  # 10 rounds por intervalo
        start_idx = start_disp - 1  # 0-based
        end_idx = end_disp - 1  # 0-based
        label_str = f"Round {end_disp}"
        intervals.append((start_idx, end_idx, label_str))

    # Número de clientes das métricas
    n_clients = 0
    for r_metrics in rounds_data.values():
        n_clients = max(n_clients, len(r_metrics.get("f1", [])))
    if n_clients == 0:
        return

    # Plotar F1
    print("Plotting F1 boxplots by round intervals...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Calcular limites globais de Y
    all_f1_vals = []
    for r_metrics in rounds_data.values():
        all_f1_vals.extend(r_metrics.get("f1", []))
    y_min, y_max = min(all_f1_vals) if all_f1_vals else 0, max(all_f1_vals) if all_f1_vals else 1
    y_margin = 0.1 * (y_max - y_min) if y_max > y_min else 0.1
    y_lower, y_upper = max(0.0, y_min - y_margin), min(1.0, y_max + y_margin)

    for idx, (start_idx, end_idx, label_str) in enumerate(intervals):
        if idx >= len(axes):
            break
        ax = axes[idx]

        # Construir dados por cliente
        data = []
        client_labels = []
        for cid in range(n_clients):
            series = []
            for r in range(start_idx, end_idx + 1):
                if r in rounds_data and cid < len(rounds_data[r].get("f1", [])):
                    series.append(rounds_data[r]["f1"][cid])
            if series:
                data.append(series)
                client_labels.append(f"C{cid+1}")

        if data:
            ax.boxplot(data, showmeans=False, vert=True)
            ax.set_xticks(range(1, len(client_labels) + 1))
            ax.set_xticklabels(client_labels, rotation=45)
            ax.set_ylabel(f"F1-Score ({label_str})")
            ax.set_ylim(y_lower, y_upper)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        else:
            ax.set_visible(False)

    for j in range(len(intervals), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_figure_pdf(fig, output_dir, "fig11_f1_boxplots_by_rounds")
    plt.close(fig)

    # Plotar Accuracy
    print("Plotting Accuracy boxplots by round intervals...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    all_acc_vals = []
    for r_metrics in rounds_data.values():
        all_acc_vals.extend(r_metrics.get("acc", []))
    y_min, y_max = min(all_acc_vals) if all_acc_vals else 0, max(all_acc_vals) if all_acc_vals else 1
    y_margin = 0.1 * (y_max - y_min) if y_max > y_min else 0.1
    y_lower, y_upper = max(0.0, y_min - y_margin), min(1.0, y_max + y_margin)

    for idx, (start_idx, end_idx, label_str) in enumerate(intervals):
        if idx >= len(axes):
            break
        ax = axes[idx]

        data = []
        client_labels = []
        for cid in range(n_clients):
            series = []
            for r in range(start_idx, end_idx + 1):
                if r in rounds_data and cid < len(rounds_data[r].get("acc", [])):
                    series.append(rounds_data[r]["acc"][cid])
            if series:
                data.append(series)
                client_labels.append(f"C{cid+1}")

        if data:
            ax.boxplot(data, showmeans=False, vert=True)
            ax.set_xticks(range(1, len(client_labels) + 1))
            ax.set_xticklabels(client_labels, rotation=45)
            ax.set_ylabel(f"Accuracy ({label_str})")
            ax.set_ylim(y_lower, y_upper)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        else:
            ax.set_visible(False)

    for j in range(len(intervals), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_figure_pdf(fig, output_dir, "fig12_accuracy_boxplots_by_rounds")
    plt.close(fig)


def plot_confusion_matrices(client_files, output_dir: Path):
    """Plota matrizes de confusão do último round de cada cliente."""
    cms = []
    labels = []

    for path in client_files:
        if not path.exists():
            continue
        data = load_json(path)
        if not data:
            continue
        
        round_ids = sorted(int(r) for r in data.keys())
        r_last = max(round_ids)
        m_last = data[str(r_last)]
        
        cm = m_last.get("confusion_matrix")
        if cm is None:
            continue

        cms.append(cm)
        parent = path.parent.name
        if "client-id-" in parent:
            cid = parent.split("client-id-")[-1]
            labels.append(f"Client {cid}")
        else:
            labels.append(parent)

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

    num_classes = len(cms[0]) if cms else 0

    print("Plotting confusion matrices...")
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

        # Escreve valores nas células
        for i in range(num_classes):
            for j in range(num_classes):
                val = cm[i][j]
                text_color = "black" if i == j else "white"
                ax.text(
                    j, i,
                    f"{val:.2f}",
                    ha="center", va="center",
                    color=text_color,
                    fontsize=CONFUSION_MATRIX_TEXT_SIZE,
                )

    # Esconde eixos sobrando
    for idx in range(len(cms), rows * cols):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        ax.axis("off")

    plt.subplots_adjust(hspace=0.3, wspace=0.7, right=0.88)

    if ims:
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(ims[-1], cax=cbar_ax)
        cbar.ax.tick_params(labelsize=CONFUSION_MATRIX_COLORBAR_TICK_SIZE)

    save_figure_pdf(fig, output_dir, "fig10_confusion_matrices")
    plt.close(fig)


# ===========================
# MAIN
# ===========================

def main():
    parser = argparse.ArgumentParser(
        description="Gera gráficos das métricas de NN federated training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Caminhos padrão configurados:
  Entrada:  {DEFAULT_RUN_DIR}
  Saída:    {DEFAULT_OUTPUT_DIR}

Exemplos:
  python plot_nn_metrics.py
  python plot_nn_metrics.py --run-dir results/ML-EdgeIIoT-FEDT-2
  python plot_nn_metrics.py --run-dir results/ML-EdgeIIoT-FEDT-1 --output-dir ./figs
        """
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"Diretório do run (padrão: {DEFAULT_RUN_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Diretório de saída dos gráficos (padrão: {DEFAULT_OUTPUT_DIR})",
    )
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("CONFIGURAÇÃO")
    print("=" * 80)
    print(f"Entrada  (run-dir):   {run_dir}")
    print(f"Saída    (output-dir): {output_dir}")
    print("=" * 80)
    print()
    
    print(f"Reading client data from: {run_dir}")
    client_files = find_client_json_files(run_dir)
    
    if not client_files:
        print("ERROR: No client JSON files found!")
        return
    
    print(f"Found {len(client_files)} client files")
    print(f"Output directory: {output_dir}\n")
    
    # Agrega métricas
    rounds_data = aggregate_client_metrics(client_files)
    
    (rounds_int, 
     f1_mean, f1_std,
     acc_mean, acc_std,
     prec_mean, prec_std,
     rec_mean, rec_std,
     fit_mean, fit_std,
     eval_mean, eval_std,
     infer_mean, infer_std,
     train_loss_mean, train_loss_std,
     test_loss_mean, test_loss_std) = summarize_rounds(rounds_data)
    
    # Gera gráficos
    plot_all_metrics_together(rounds_int, acc_mean, prec_mean, rec_mean, f1_mean, output_dir)
    plot_f1_and_accuracy(rounds_int, f1_mean, f1_std, acc_mean, acc_std, output_dir)
    plot_precision_recall(rounds_int, prec_mean, prec_std, rec_mean, rec_std, output_dir)
    plot_losses(rounds_int, train_loss_mean, train_loss_std, test_loss_mean, test_loss_std, output_dir)
    plot_times(rounds_int, fit_mean, fit_std, eval_mean, eval_std, infer_mean, infer_std, output_dir)
    plot_all_metrics_per_client(client_files, output_dir)
    plot_metrics_bar(client_files, output_dir)
    plot_boxplots_by_round_intervals(rounds_data, [10, 20, 30, 40], output_dir)
    plot_confusion_matrices(client_files, output_dir)
    
    print(f"\n✓ All figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
