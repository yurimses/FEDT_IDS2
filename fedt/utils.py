from fedt.settings import (
    dataset_path,  
    validate_dataset_size, aggregation_strategies, 
    results_folder, logs_folder, label_target, # [CLASS]
    number_of_clients, partition_type, non_iid_alpha, partition_seed,
    min_samples_per_class,  # [CLASSIF]
    partitions_folder,
    )

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # [CLASSIF]
from sklearn.preprocessing import LabelEncoder  # [CLASS]


import pickle
import tempfile

from fedt import fedT_pb2

import logging
import colorlog

import time

import joblib, io

from pathlib import Path

import psutil

import signal, os

def set_initial_params(model: RandomForestClassifier, X_train, y_train): # [CLASSIF]
    """
    ### Função:
    Setar os parâmetros iniciais do modelo, treinando a floresta com apenas 3 amostras.
    ### Args:
    - Model: O modelo de floresta.
    - Data: As features para treinar o modelo.
    - Label: Os targets para treinar o modelo.
    ### Returns:
    - None.
    """
    model.fit(X_train, y_train)

def set_model_params(
    model: RandomForestClassifier, params: list
) -> RandomForestClassifier: # [CLASSIF]
    """
    ### Função:
    Setar os parâmetros do modelo.
    ### Args:
    - Modelo: Modelo de floresta.
    - Parâmetros: As árvores que serão utilizadas como estimators.
    ### Returns:
    - Modelo: Modelo de floresta, com os novos estimators.
    """
    model.estimators_ = params
    return model

def get_model_parameters(model: RandomForestClassifier) -> list: # [CLASSIF]
    """
    ### Função:
    Obter as árvores do modelo.
    ### Args:
    - Modelo: Modelo de floresta.
    ### Returns:
    - Parâmetros: As árvores que estão sendo utilizadas como estimators no modelo.
    """
    params = model.estimators_
    return params

def load_dataset():
    """
    ### Função:
    Carregar o dataset completo.
    ### Args:
    - None.
    ### Returns:
    - Data Train: As features para treinar o modelo.
    - Label Train: Os targets para treinar o modelo.
    - Data Test: As features para testar o modelo.
    - Label Test: Os targets para testar o modelo
    - Feature Names: Os nomes das features originais
    - Label Name: O nome da coluna de rótulo
    """
    df = pd.read_csv(dataset_path)  # [CLASS]
    if label_target not in df.columns:  # [CLASS]
        raise ValueError(f"Coluna de rótulo '{label_target}' não encontrada no dataset.")  # [CLASS]

    y_series = df[label_target]  # [CLASS]

    # [CLASS] Se for rótulo binário, converte direto para int; senão, faz encoding para inteiros
    if label_target == "Attack_label":  # [CLASS]
        y = y_series.astype(int).to_numpy()  # [CLASS]
    else:  # [CLASS]
        le = LabelEncoder()  # [CLASS]
        y = le.fit_transform(y_series.astype(str))  # [CLASS]

    # [CLASSIF] Remove o target e quaisquer colunas auxiliares de rótulo "Attack_*"
    drop_cols = [label_target] + [c for c in df.columns if c.startswith("Attack_") and c != label_target]  # [CLASSIF]
    X = df.drop(columns=drop_cols, errors="ignore")  # [CLASSIF]

    # [CLASSIF] Salva os nomes das features antes de converter para numpy
    feature_names = list(X.columns)  # [CLASSIF]

    # [CLASSIF] Garante que todas as features sejam numéricas (este dataset possui colunas categóricas como Protocol/Service)
    for col in X.columns:  # [CLASSIF]
        if X[col].dtype == "object":  # [CLASSIF]
            X[col] = pd.factorize(X[col].astype(str), sort=True)[0]  # [CLASSIF]

    X = X.astype(np.float32)  # [CLASS]
    return X.to_numpy(), y, feature_names, label_target  # [CLASS]

def _partition_indices_iid(n_samples: int, num_partitions: int, seed: int = 42):  # [CLASSIF]
    """[CLASSIF] Particionamento IID por índices (embaralha e divide em partes quase iguais)."""  # [CLASSIF]
    rng = np.random.default_rng(seed)  # [CLASSIF]
    indices = rng.permutation(n_samples)  # [CLASSIF]
    return [arr.astype(int, copy=False) for arr in np.array_split(indices, num_partitions)]  # [CLASSIF]


def _partition_indices_dirichlet(  # [CLASSIF]
    y: np.ndarray, num_partitions: int, alpha: float, seed: int = 42, min_partition_size: int = 1  # [CLASSIF]
):  # [CLASSIF]
    """  # [CLASSIF]
    [CLASSIF] Particionamento Non-IID label-based via Dirichlet(alpha).  # [CLASSIF]
    A cada classe, amostras são distribuídas entre clientes segundo uma probabilidade ~ Dirichlet(alpha).  # [CLASSIF]
    """  # [CLASSIF]
    rng = np.random.default_rng(seed)  # [CLASSIF]
    y = np.asarray(y)  # [CLASSIF]
    classes = np.unique(y)  # [CLASSIF]

    # [CLASSIF] Tenta algumas vezes para evitar partições vazias (equivalente à ideia de min_partition_size).  # [CLASSIF]
    for _ in range(10):  # [CLASSIF]
        parts = [[] for _ in range(num_partitions)]  # [CLASSIF]

        for c in classes:  # [CLASSIF]
            cls_idx = np.flatnonzero(y == c)  # [CLASSIF]
            cls_idx = rng.permutation(cls_idx)  # [CLASSIF]
            n_c = len(cls_idx)  # [CLASSIF]
            if n_c == 0:  # [CLASSIF]
                continue  # [CLASSIF]

            probs = rng.dirichlet(np.repeat(alpha, num_partitions))  # [CLASSIF]
            counts = rng.multinomial(n_c, probs)  # [CLASSIF]

            start = 0  # [CLASSIF]
            for pid, cnt in enumerate(counts):  # [CLASSIF]
                if cnt > 0:  # [CLASSIF]
                    parts[pid].append(cls_idx[start : start + cnt])  # [CLASSIF]
                start += cnt  # [CLASSIF]

        parts = [np.concatenate(p) if len(p) else np.array([], dtype=int) for p in parts]  # [CLASSIF]

        if min_partition_size <= 1 or all(len(p) >= min_partition_size for p in parts):  # [CLASSIF]
            return [rng.permutation(p) if len(p) else p for p in parts]  # [CLASSIF]

    # [CLASSIF] Fallback: retorna a última tentativa mesmo sem satisfazer min_partition_size.  # [CLASSIF]
    return [rng.permutation(p) if len(p) else p for p in parts]  # [CLASSIF]


def _partition_indices_dirichlet_allclasses(  # [CLASSIF]
    y: np.ndarray, num_partitions: int, alpha: float, seed: int = 42, min_samples_per_class: int = 10  # [CLASSIF]
):  # [CLASSIF]
    """  # [CLASSIF]
    [CLASSIF] Particionamento Non-IID via Dirichlet(alpha) com GARANTIA de todas as classes em cada partição.  # [CLASSIF]
    
    Algoritmo híbrido:  # [CLASSIF]
    1. Gera distribuição Dirichlet inicial (mantém heterogeneidade)  # [CLASSIF]
    2. Redistribui amostras para garantir min_samples_per_class de cada classe em cada partição  # [CLASSIF]
    3. Mantém características non-IID mas evita partições sem classes específicas  # [CLASSIF]
    """  # [CLASSIF]
    rng = np.random.default_rng(seed)  # [CLASSIF]
    y = np.asarray(y)  # [CLASSIF]
    classes = np.unique(y)  # [CLASSIF]
    n_classes = len(classes)  # [CLASSIF]
    
    # [CLASSIF] Verificar se há amostras suficientes  # [CLASSIF]
    for c in classes:  # [CLASSIF]
        n_samples_class = np.sum(y == c)  # [CLASSIF]
        min_needed = min_samples_per_class * num_partitions  # [CLASSIF]
        if n_samples_class < min_needed:  # [CLASSIF]
            import warnings  # [CLASSIF]
            warnings.warn(  # [CLASSIF]
                f"Classe {c} tem apenas {n_samples_class} amostras, "  # [CLASSIF]
                f"mas precisa de {min_needed} para garantir {min_samples_per_class} por partição. "  # [CLASSIF]
                f"Reduzindo para {n_samples_class // num_partitions} amostras por partição.",  # [CLASSIF]
                RuntimeWarning  # [CLASSIF]
            )  # [CLASSIF]
            min_samples_per_class = max(1, n_samples_class // num_partitions)  # [CLASSIF]
    
    # [CLASSIF] Fase 1: Distribuição Dirichlet base  # [CLASSIF]
    parts_indices = {pid: [] for pid in range(num_partitions)}  # [CLASSIF]
    
    for c in classes:  # [CLASSIF]
        cls_idx = np.flatnonzero(y == c)  # [CLASSIF]
        cls_idx = rng.permutation(cls_idx)  # [CLASSIF]
        n_c = len(cls_idx)  # [CLASSIF]
        
        # [CLASSIF] Reservar min_samples_per_class para cada partição primeiro  # [CLASSIF]
        reserved_per_partition = min(min_samples_per_class, n_c // num_partitions)  # [CLASSIF]
        reserved_total = reserved_per_partition * num_partitions  # [CLASSIF]
        remaining = n_c - reserved_total  # [CLASSIF]
        
        # [CLASSIF] Distribuir amostras reservadas uniformemente  # [CLASSIF]
        idx = 0  # [CLASSIF]
        for pid in range(num_partitions):  # [CLASSIF]
            parts_indices[pid].append(cls_idx[idx:idx + reserved_per_partition])  # [CLASSIF]
            idx += reserved_per_partition  # [CLASSIF]
        
        # [CLASSIF] Distribuir amostras restantes via Dirichlet (mantém heterogeneidade)  # [CLASSIF]
        if remaining > 0:  # [CLASSIF]
            probs = rng.dirichlet(np.repeat(alpha, num_partitions))  # [CLASSIF]
            counts = rng.multinomial(remaining, probs)  # [CLASSIF]
            
            for pid, cnt in enumerate(counts):  # [CLASSIF]
                if cnt > 0:  # [CLASSIF]
                    parts_indices[pid].append(cls_idx[idx:idx + cnt])  # [CLASSIF]
                    idx += cnt  # [CLASSIF]
    
    # [CLASSIF] Concatenar e embaralhar índices de cada partição  # [CLASSIF]
    parts = [  # [CLASSIF]
        rng.permutation(np.concatenate(parts_indices[pid])) if parts_indices[pid] else np.array([], dtype=int)  # [CLASSIF]
        for pid in range(num_partitions)  # [CLASSIF]
    ]  # [CLASSIF]
    
    return parts  # [CLASSIF]


def load_house_client(client_id: int):  # [CLASSIF]
    """
    [CLASSIF] Carrega a partição do dataset para um cliente específico (iid ou non-iid),
    evitando converter X/y para listas Python (isso estoura memória em datasets grandes).  # [CLASSIF]
    """  # [CLASSIF]
    X, y, feature_names, label_name = load_dataset()  # [CLASSIF]

    if client_id < 0 or client_id >= number_of_clients:  # [CLASSIF]
        raise ValueError(  # [CLASSIF]
            f"client_id {client_id} fora do intervalo [0, {number_of_clients - 1}]"  # [CLASSIF]
        )  # [CLASSIF]

    # [CLASSIF] Seleciona o tipo de particionamento (iid ou non-iid)  # [CLASSIF]
    pt = partition_type.lower() if isinstance(partition_type, str) else "iid"  # [CLASSIF]
    if pt == "iid":  # [CLASSIF]
        partitions = _partition_indices_iid(len(y), number_of_clients, seed=int(partition_seed))  # [CLASSIF]
    elif pt in ("non-iid", "non_iid", "noniid"):  # [CLASSIF]
        partitions = _partition_indices_dirichlet(  # [CLASSIF]
            y=np.asarray(y),  # [CLASSIF]
            num_partitions=number_of_clients,  # [CLASSIF]
            alpha=float(non_iid_alpha),  # [CLASSIF]
            seed=int(partition_seed),  # [CLASSIF]
            min_partition_size=1,  # [CLASSIF]
        )  # [CLASSIF]
    elif pt in ("non-iid-allclasses", "non_iid_allclasses", "noniid_allclasses"):  # [CLASSIF]
        partitions = _partition_indices_dirichlet_allclasses(  # [CLASSIF]
            y=np.asarray(y),  # [CLASSIF]
            num_partitions=number_of_clients,  # [CLASSIF]
            alpha=float(non_iid_alpha),  # [CLASSIF]
            seed=int(partition_seed),  # [CLASSIF]
            min_samples_per_class=int(min_samples_per_class),  # [CLASSIF]
        )  # [CLASSIF]
    else:
        raise ValueError(f"Tipo de particionamento inválido: {partition_type}")  # [CLASSIF]

    client_idx = partitions[client_id]  # [CLASSIF]
    if client_idx.size == 0:  # [CLASSIF]
        raise ValueError(f"Partição vazia para client_id={client_id}. Ajuste alpha/num_clients.")  # [CLASSIF]

    X_client = X[client_idx]  # [CLASSIF]
    y_client = np.asarray(y)[client_idx]  # [CLASSIF]

    # [CLASSIF] Divisão estratificada quando possível; caso contrário, faz split simples para evitar erro.  # [CLASSIF]
    stratify_arg = y_client  # [CLASSIF]
    uniq, cnt = np.unique(y_client, return_counts=True)  # [CLASSIF]
    if uniq.size < 2 or np.min(cnt) < 2:  # [CLASSIF]
        stratify_arg = None  # [CLASSIF]

    X_train, X_test, y_train, y_test = train_test_split(  # [CLASSIF]
        X_client, y_client, test_size=0.2, stratify=stratify_arg, random_state=int(partition_seed)  # [CLASSIF]
    )  # [CLASSIF]
    
    # Salva as partições no disco
    _save_partition(client_id, X_train, y_train, X_test, y_test, pt, feature_names, label_name)
    
    return X_train, y_train, X_test, y_test


def _save_partition(client_id: int, X_train, y_train, X_test, y_test, partition_type_str: str, feature_names: list, label_name: str):
    """
    Salva as partições de treino e teste de um cliente em formato CSV.
    Organiza por: partitions/{dataset_name}/{partition_type}/client_{id}/
    """
    # Extrai o nome do dataset do caminho do arquivo
    dataset_name = Path(dataset_path).stem
    
    # Cria o caminho da pasta: partitions/{dataset_name}/{partition_type}/
    partition_dir = partitions_folder / dataset_name / partition_type_str / f"client_{client_id}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    
    # Salva train e test em arquivos CSV separados
    train_path = partition_dir / "train.csv"
    test_path = partition_dir / "test.csv"
    
    # Cria DataFrames com X e y combinados, usando os nomes das features
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df[label_name] = y_train
    train_df.to_csv(train_path, index=False)
    
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df[label_name] = y_test
    test_df.to_csv(test_path, index=False)
    
    # Salva também metadados em JSON
    import json
    metadata = {
        "client_id": client_id,
        "dataset": dataset_name,
        "partition_type": partition_type_str,
        "partition_seed": int(partition_seed),
        "label_name": label_name,
        "feature_names": feature_names,
        "num_features": len(feature_names),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "num_classes": len(np.unique(y_train)),
        "train_class_distribution": {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        "test_class_distribution": {int(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
    }
    
    metadata_path = partition_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

def load_dataset_for_server() -> list:
    """
    ### Função:
    Carregar um subconjunto mínimo do dataset que contenha todas as classes presentes no problema.  # [CLASSIF]
    Isso inicializa o modelo do servidor de forma compatível com qualquer dataset de classificação,   # [CLASSIF]
    sem assumir um número fixo de amostras ou classes.                                                # [CLASSIF]
    ### Args:
    - None.
    ### Returns:
    - Data Train: As features.
    - Label Train: Os targets. 
    """
    data, label, feature_names, label_name = load_dataset()  # [CLASSIF]

    # [CLASSIF] Converte rótulos para array NumPy para tratamento genérico
    label_array = np.asarray(label)  # [CLASSIF]

    # [CLASSIF] Para cada classe, escolhe um índice representativo (primeira ocorrência)
    unique_classes, first_indices = np.unique(label_array, return_index=True)  # [CLASSIF]

    # [CLASSIF] Suporta tanto DataFrames quanto arrays NumPy
    if hasattr(data, "iloc"):  # DataFrame / Series  # [CLASSIF]
        data_init = data.iloc[first_indices]  # [CLASSIF]
    else:
        data_init = np.asarray(data)[first_indices]  # [CLASSIF]

    label_init = label_array[first_indices]  # [CLASSIF]

    return data_init, label_init  # [CLASSIF]

def load_server_side_validation_data():
    """
    ### Função:
    Carregar o dataset com apenas 1000 amostras, 
    servirá para carregar os dados de validação para testar a performance do modelo.
    ### Args:
    - None.
    ### Returns:
    - Data Valid: As features para validação.
    - Label Valid: Os targets para validação. 
    """
    # [CLASSIF] Dados de validação para classificação com MNIST
    data, label, _, _ = load_dataset()

    _, data_valid, _, label_valid = train_test_split(
        data, label, test_size=0.2, stratify=label
    )  # [CLASSIF]
    return data_valid[-validate_dataset_size:], label_valid[-validate_dataset_size:]

def serialise_tree(tree_model) -> bytes:
    """
    ### Função:
    Serializa um modelo de árvore usando joblib com compressão leve.
    Retorna os bytes resultantes.
    """
    buffer = io.BytesIO()
    joblib.dump(tree_model, buffer, compress=3)  # compress=3 → bom equilíbrio entre tamanho e CPU
    return buffer.getvalue()

def deserialise_tree(serialised_tree_model):
    """
    ### Função:
    Desserializa um modelo de árvore (em bytes) para um objeto Python.
    """
    buffer = io.BytesIO(serialised_tree_model)
    return joblib.load(buffer)

def serialise_several_trees(tree_models):
    """
    ### Função:
    Converter vários modelos de árvore de objeto para bytes.
    ### Args:
    - tree_models: Lista com vários modelos de árvore.
    ### Returns:
    - serialised_trees: Lista de modelos de árvore convertidos em bytes.
    """
    serialised_trees = []
    for tree in tree_models:
        buffer = io.BytesIO()
        joblib.dump(tree, buffer, compress=3)  # compress=3 → equilíbrio entre tamanho e CPU
        serialised_trees.append(buffer.getvalue())
    return serialised_trees

def deserialise_several_trees(serialised_tree_models):
    """
    ### Função:
    Converter vários modelos de árvore em bytes para modelos em formato de objeto.
    ### Args:
    - serialised_tree_models: Lista com vários modelos de árvore em bytes.
    ### Returns:
    - deserialised_trees: Lista com vários modelos de árvore em formato de objeto.
    """
    deserialised_trees = []
    for serialised_tree in serialised_tree_models:
        buffer = io.BytesIO(serialised_tree)
        deserialised_trees.append(joblib.load(buffer))
    return deserialised_trees

def setup_logger(name, log_file, level=logging.INFO):
    """Cria logger colorido que também grava em arquivo."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # --- FORMATADOR COLORIDO PARA O CONSOLE ---
    color_formatter = colorlog.ColoredFormatter(
        "%(asctime_log_color)s%(asctime)s%(reset)s "
        "[%(log_color)s%(levelname)s%(reset)s] "
        "[%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
        secondary_log_colors={
            "asctime": {
                "DEBUG": "bold_purple",
                "INFO": "bold_purple",
                "WARNING": "bold_purple",
                "ERROR": "bold_purple",
                "CRITICAL": "bold_purple",
            }
        },
        style="%",
    )

    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(color_formatter)

    # --- FORMATADOR PADRÃO PARA O ARQUIVO ---
    log_file_path = logs_folder  / log_file
    file_handler = logging.FileHandler(log_file_path)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # --- ADICIONA OS HANDLERS AO LOGGER ---
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def format_time(timestamp):
    return time.strftime('%H:%M:%S', time.gmtime(timestamp))

def get_serialised_size_bytes(serialised) -> int:
    return len(serialised)

def get_size_of_many_serialised_models(serialised_models):
    return sum(len(model) for model in serialised_models)

def create_strategies_result_folder():
    for strategy in aggregation_strategies:
        subpath = results_folder / strategy
        subpath.mkdir(parents=True, exist_ok=True)

def create_specific_result_folder(strategy, base_name):
    subpath = results_folder / strategy / base_name
    subpath.mkdir(parents=True, exist_ok=True)
    return subpath

def create_specific_result_folder_with_dataset(strategy, dataset_name, partition_type, base_name):
    """Cria pasta de resultados com estrutura: results/{strategy}/{dataset_name}/{partition_type}/{base_name}"""
    subpath = results_folder / strategy / dataset_name / partition_type / base_name
    subpath.mkdir(parents=True, exist_ok=True)
    return subpath

def create_specific_logs_folder(strategy, base_name):
    subpath = logs_folder / base_name / strategy
    subpath.mkdir(parents=True, exist_ok=True)
    return subpath

def create_specific_logs_folder_with_dataset(dataset_name, partition_type, strategy, base_name):
    """Cria pasta de logs com estrutura: logs/{base_name}/{dataset_name}/{partition_type}/{strategy}"""
    subpath = logs_folder / base_name / dataset_name / partition_type / strategy
    subpath.mkdir(parents=True, exist_ok=True)
    return subpath

def get_process_cmd(proc):
    """Retorna o comando completo de um processo como string (ou None se inacessível)."""
    try:
        cmdline = proc.info.get('cmdline')
        if not cmdline:
            return None
        return " ".join(cmdline)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None

def find_target_processes(targets):
    """Retorna um dicionário {target_string: [Process, ...]} para cada target encontrado."""
    matches = {t: [] for t in targets}
    for proc in psutil.process_iter(attrs=['pid', 'cmdline']):
        cmd = get_process_cmd(proc)
        if not cmd:
            continue
        for t in targets:
            if t in cmd:
                matches[t].append(proc)
                break
    return matches

def kill_processes(processes, name):
    for target, plist in list(processes.items()):
        for proc in list(plist):
            if proc.name() == name:
                os.kill(proc.pid, signal.SIGINT)


if __name__ == "__main__":
    create_specific_result_folder("Client")