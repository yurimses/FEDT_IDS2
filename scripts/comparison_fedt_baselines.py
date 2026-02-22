import json
from pathlib import Path
from statistics import mean, pstdev
import math
import matplotlib.pyplot as plt
import pandas as pd

AXIS_LABEL_SIZE = 14
AXIS_LABEL_WEIGHT = "bold"
TICK_LABEL_SIZE = 14
plt.rcParams.update({
    "axes.labelsize": AXIS_LABEL_SIZE,
    "axes.labelweight": AXIS_LABEL_WEIGHT,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
})

# ==========================
# CONFIGURAÇÃO DE ARQUIVOS
# ==========================

# Arquivos dos clientes FEDT
CLIENT_FILES_FEDT = [
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/non_iid_allclasses/client-id-0/best_trees_client-id-0_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/non_iid_allclasses/client-id-1/best_trees_client-id-1_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/non_iid_allclasses/client-id-2/best_trees_client-id-2_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/non_iid_allclasses/client-id-3/best_trees_client-id-3_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/non_iid_allclasses/client-id-4/best_trees_client-id-4_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/non_iid_allclasses/client-id-5/best_trees_client-id-5_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/non_iid_allclasses/client-id-6/best_trees_client-id-6_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/non_iid_allclasses/client-id-7/best_trees_client-id-7_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/non_iid_allclasses/client-id-8/best_trees_client-id-8_1.json"),
    Path("/home/yuri/FEDT_IDS2/results/best_trees/ML-EdgeIIoT-FEDT/non_iid_allclasses/client-id-9/best_trees_client-id-9_1.json"),
]

# Pasta onde as figuras serão salvas
FIG_DIR = Path("/home/yuri/FEDT_IDS2/figures/comparisons/")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Baselines CSV paths
BASELINES = {
    "FeatureCloud": Path("/home/yuri/FEDT_IDS2/fedtxbaselines/featureCloud/non_iid_aggregated_metrics.csv"),
    "Flex-Trees": Path("/home/yuri/FEDT_IDS2/fedtxbaselines/flex-trees/non_iid_aggregated_metrics.csv"),
}

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


def load_baseline_metrics(csv_path: Path):
    """Carrega métricas de um baseline a partir do CSV.
    
    Retorna tupla: (acc_mean, acc_std, prec_mean, prec_std, rec_mean, rec_std, f1_mean, f1_std)
    """
    if not csv_path.exists():
        return None
    
    df = pd.read_csv(csv_path)
    
    # Verificar se 'Std_Dev' existe, senão usar 0.0 como padrão
    has_std_dev = 'Std_Dev' in df.columns
    
    metrics = {}
    for idx, row in df.iterrows():
        metric_name = row['Metric']
        mean_val = row['Mean']
        std_val = row['Std_Dev'] if has_std_dev else 0.0
        
        # Mapear para nome padrão - remove prefixo "Local_" ou "Global_"
        clean_metric = metric_name.replace('Local_', '').replace('Global_', '').lower()
        
        if clean_metric == 'accuracy':
            metrics['accuracy'] = (mean_val, std_val)
        elif clean_metric == 'precision':
            metrics['precision'] = (mean_val, std_val)
        elif clean_metric == 'recall':
            metrics['recall'] = (mean_val, std_val)
        elif clean_metric == 'f1':
            metrics['f1'] = (mean_val, std_val)
    
    if 'accuracy' in metrics and 'precision' in metrics and 'recall' in metrics and 'f1' in metrics:
        acc_mean, acc_std = metrics['accuracy']
        prec_mean, prec_std = metrics['precision']
        rec_mean, rec_std = metrics['recall']
        f1_mean, f1_std = metrics['f1']
        return acc_mean, acc_std, prec_mean, prec_std, rec_mean, rec_std, f1_mean, f1_std
    
    return None


def plot_comparison_metrics_bar(fedt_metrics, baselines_metrics, output_dir: Path):
    """Gráfico de barras comparando métricas de múltiplas soluções no último round.
    
    fedt_metrics: tupla (acc_mean, acc_std, prec_mean, prec_std, rec_mean, rec_std, f1_mean, f1_std)
    baselines_metrics: dicionário {nome: tupla_metricas}
    """
    if fedt_metrics is None:
        print("Aviso: Não foi possível extrair métricas do FEDT")
        return
    
    # Nomes dos modelos
    models = ["FEDT"] + list(baselines_metrics.keys())
    
    # Métricas: Accuracy, Recall, Precision, F1
    metric_labels = ["Accuracy", "Recall", "Precision", "F1-score"]
    
    # Extrair valores para FEDT
    acc_mean_fedt, acc_std_fedt, prec_mean_fedt, prec_std_fedt, rec_mean_fedt, rec_std_fedt, f1_mean_fedt, f1_std_fedt = fedt_metrics
    
    # Dicionário com valores por modelo
    all_values = {
        "Accuracy": [acc_mean_fedt],
        "Recall": [rec_mean_fedt],
        "Precision": [prec_mean_fedt],
        "F1-score": [f1_mean_fedt],
    }
    
    all_errors = {
        "Accuracy": [acc_std_fedt],
        "Recall": [rec_std_fedt],
        "Precision": [prec_std_fedt],
        "F1-score": [f1_std_fedt],
    }
    
    # Adicionar valores dos baselines
    for baseline_name in baselines_metrics.keys():
        if baselines_metrics[baseline_name] is None:
            print(f"Aviso: Não foi possível extrair métricas de {baseline_name}")
            continue
        
        acc_mean, acc_std, prec_mean, prec_std, rec_mean, rec_std, f1_mean, f1_std = baselines_metrics[baseline_name]
        all_values["Accuracy"].append(acc_mean)
        all_values["Recall"].append(rec_mean)
        all_values["Precision"].append(prec_mean)
        all_values["F1-score"].append(f1_mean)
        
        all_errors["Accuracy"].append(acc_std)
        all_errors["Recall"].append(rec_std)
        all_errors["Precision"].append(prec_std)
        all_errors["F1-score"].append(f1_std)
    
    # Cores para os modelos
    colors = ["#07a791", "#FF9900", "#ff000086"]
    colors = colors[:len(models)]
    
    # Criar figura com subplots para cada métrica
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Comparison of Federated Learning Solutions (EdgeIoT Dataset)", fontsize=14, fontweight="bold")
    
    for idx, metric_label in enumerate(metric_labels):
        ax = axes[idx]
        x = range(len(models))
        values = all_values[metric_label]
        errors = all_errors[metric_label]
        
        # Limitar erros para não ultrapassar 1.0
        capped_errors = [min(err, max(0.0, 1.0 - val + 1e-6)) for val, err in zip(values, errors)]
        
        bars = ax.bar(
            x,
            values,
            yerr=capped_errors,
            capsize=5,
            color=colors,
            edgecolor="black",
            linewidth=1.0,
            width=0.6,
        )
        
        # Adicionar valores nas barras
        for bar, value, err in zip(bars, values, capped_errors):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + err + 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel("Score")
        ax.set_title(metric_label, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    
    fig.tight_layout()
    save_figure_pdf(fig, output_dir, "comparison_metrics_all_solutions")
    plt.close(fig)


def plot_comparison_metrics_grouped(fedt_metrics, baselines_metrics, output_dir: Path):
    """Gráfico de barras agrupadas comparando todas as soluções lado a lado."""
    if fedt_metrics is None:
        print("Aviso: Não foi possível extrair métricas do FEDT")
        return
    
    # Nomes dos modelos
    models = ["FEDT"] + list(baselines_metrics.keys())
    
    # Métricas: Accuracy, Recall, Precision, F1
    metric_labels = ["Accuracy", "Recall", "Precision", "F1-score"]
    
    # Extrair valores para FEDT
    acc_mean_fedt, acc_std_fedt, prec_mean_fedt, prec_std_fedt, rec_mean_fedt, rec_std_fedt, f1_mean_fedt, f1_std_fedt = fedt_metrics
    
    # Dicionário com valores por modelo
    all_values = {
        "Accuracy": [acc_mean_fedt],
        "Recall": [rec_mean_fedt],
        "Precision": [prec_mean_fedt],
        "F1-score": [f1_mean_fedt],
    }
    
    all_errors = {
        "Accuracy": [acc_std_fedt],
        "Recall": [rec_std_fedt],
        "Precision": [prec_std_fedt],
        "F1-score": [f1_std_fedt],
    }
    
    # Adicionar valores dos baselines
    for baseline_name in baselines_metrics.keys():
        if baselines_metrics[baseline_name] is None:
            print(f"Aviso: Não foi possível extrair métricas de {baseline_name}")
            continue
        
        acc_mean, acc_std, prec_mean, prec_std, rec_mean, rec_std, f1_mean, f1_std = baselines_metrics[baseline_name]
        all_values["Accuracy"].append(acc_mean)
        all_values["Recall"].append(rec_mean)
        all_values["Precision"].append(prec_mean)
        all_values["F1-score"].append(f1_mean)
        
        all_errors["Accuracy"].append(acc_std)
        all_errors["Recall"].append(rec_std)
        all_errors["Precision"].append(prec_std)
        all_errors["F1-score"].append(f1_std)
    
    # Cores para os modelos
    colors = ["#07a791", "#FF9900", "#ff000086"]
    colors = colors[:len(models)]
    
    # Preparar dados para gráfico agrupado
    x = range(len(metric_labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plotar barras para cada modelo
    for model_idx, model_name in enumerate(models):
        offset = (model_idx - 1) * width
        metric_values = [all_values[metric][model_idx] for metric in metric_labels]
        metric_errors = [all_errors[metric][model_idx] for metric in metric_labels]
        
        bars = ax.bar(
            [i + offset for i in x],
            metric_values,
            width,
            label=model_name,
            yerr=metric_errors,
            capsize=5,
            color=colors[model_idx],
            edgecolor="black",
            linewidth=1.0,
        )
        
        # Adicionar valores nas barras
        for bar, value, err in zip(bars, metric_values, metric_errors):
            height = bar.get_height()
            capped_err = min(err, max(0.0, 1.0 - value + 1e-6))
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + capped_err + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Comparison of Federated Learning Solutions (EdgeIoT Dataset)", fontweight="bold")
    ax.set_xticks([i for i in x])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.0, 1.15)
    ax.legend(loc='lower right')
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    
    fig.tight_layout()
    save_figure_pdf(fig, output_dir, "comparison_metrics_grouped")
    plt.close(fig)


# ==========================
# MAIN
# ==========================

def main():
    print("Extracting FEDT metrics...")
    fedt_metrics = extract_metrics_at_round(CLIENT_FILES_FEDT)
    
    if fedt_metrics is None:
        print("Erro: Não foi possível extrair métricas do FEDT")
        return
    
    print(f"FEDT - Accuracy: {fedt_metrics[0]:.4f} ± {fedt_metrics[1]:.4f}")
    
    print("\nExtracting baseline metrics...")
    baselines_metrics = {}
    for baseline_name, csv_path in BASELINES.items():
        print(f"Loading {baseline_name}...")
        metrics = load_baseline_metrics(csv_path)
        baselines_metrics[baseline_name] = metrics
        if metrics:
            print(f"{baseline_name} - Accuracy: {metrics[0]:.4f} ± {metrics[1]:.4f}")
        else:
            print(f"{baseline_name} - Failed to load")
    
    print("\nGenerating comparison plots...")
    plot_comparison_metrics_bar(fedt_metrics, baselines_metrics, FIG_DIR)
    plot_comparison_metrics_grouped(fedt_metrics, baselines_metrics, FIG_DIR)
    
    print(f"\nComparison plots saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
