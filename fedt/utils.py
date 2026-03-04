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


def _partition_indices_dominant_client(  # [CLASSIF]
    y: np.ndarray,  # [CLASSIF]
    num_partitions: int,  # [CLASSIF]
    dominant_client_id: int,  # [CLASSIF]
    dominant_percentage: float,  # [CLASSIF]
    alpha: float,  # [CLASSIF]
    seed: int = 42,  # [CLASSIF]
    min_samples_per_class: int = 10  # [CLASSIF]
):  # [CLASSIF]
    """  # [CLASSIF]
    [CLASSIF] Particionamento com um cliente dominante + resto em non_iid_allclasses.  # [CLASSIF]
    
    Um cliente específico (dominant_client_id) recebe dominant_percentage% de TODAS as classes.  # [CLASSIF]
    Os dados remanescentes são distribuídos entre os outros clientes usando non_iid_allclasses.  # [CLASSIF]
    
    Algoritmo:  # [CLASSIF]
    1. Para cada classe, aloca dominant_percentage% para o cliente dominante  # [CLASSIF]
    2. Distribui os dados remanescentes com non_iid_allclasses para os demais clientes  # [CLASSIF]
    3. Garante min_samples_per_class em cada partição (quando possível)  # [CLASSIF]
    """  # [CLASSIF]
    rng = np.random.default_rng(seed)  # [CLASSIF]
    y = np.asarray(y)  # [CLASSIF]
    classes = np.unique(y)  # [CLASSIF]
    
    if dominant_client_id < 0 or dominant_client_id >= num_partitions:  # [CLASSIF]
        raise ValueError(  # [CLASSIF]
            f"dominant_client_id {dominant_client_id} fora do intervalo [0, {num_partitions - 1}]"  # [CLASSIF]
        )  # [CLASSIF]
    
    if dominant_percentage <= 0 or dominant_percentage >= 1.0:  # [CLASSIF]
        raise ValueError(  # [CLASSIF]
            f"dominant_percentage deve estar entre 0 e 1. Recebido: {dominant_percentage}"  # [CLASSIF]
        )  # [CLASSIF]
    
    # [CLASSIF] Inicializa dicionário para índices do cliente dominante e dos restantes  # [CLASSIF]
    dominant_indices = []  # [CLASSIF]
    remaining_y = []  # [CLASSIF]
    remaining_idx_mapping = []  # Mapa de índices originais para restantes  # [CLASSIF]
    
    # [CLASSIF] Para cada classe, aloca dominant_percentage% para o cliente dominante  # [CLASSIF]
    for c in classes:  # [CLASSIF]
        cls_idx = np.flatnonzero(y == c)  # [CLASSIF]
        cls_idx = rng.permutation(cls_idx)  # [CLASSIF]
        n_c = len(cls_idx)  # [CLASSIF]
        
        # [CLASSIF] Calcula quantas amostras da classe vão para o cliente dominante  # [CLASSIF]
        n_dominant = max(1, int(np.round(n_c * dominant_percentage)))  # [CLASSIF]
        n_dominant = min(n_dominant, n_c - 1)  # Garante pelo menos 1 amostra para os demais  # [CLASSIF]
        
        # [CLASSIF] Aloca amostras para o cliente dominante  # [CLASSIF]
        dominant_indices.append(cls_idx[:n_dominant])  # [CLASSIF]
        
        # [CLASSIF] Amostras restantes vão para particionamento não-IID  # [CLASSIF]
        for idx in cls_idx[n_dominant:]:  # [CLASSIF]
            remaining_y.append(int(c))  # [CLASSIF]
            remaining_idx_mapping.append(int(idx))  # [CLASSIF]
    
    # [CLASSIF] Concatena índices do cliente dominante  # [CLASSIF]
    dominant_part = rng.permutation(np.concatenate(dominant_indices)) if dominant_indices else np.array([], dtype=int)  # [CLASSIF]
    
    # [CLASSIF] Particiona os dados remanescentes com non_iid_allclasses  # [CLASSIF]
    remaining_y = np.asarray(remaining_y)  # [CLASSIF]
    remaining_idx_mapping = np.asarray(remaining_idx_mapping, dtype=int)  # [CLASSIF]
    
    # [CLASSIF] Número de partições para os dados remanescentes (excluindo cliente dominante)  # [CLASSIF]
    num_remaining_partitions = num_partitions - 1  # [CLASSIF]
    
    if num_remaining_partitions > 0 and len(remaining_y) > 0:  # [CLASSIF]
        # [CLASSIF] Particiona índices remanescentes com non_iid_allclasses  # [CLASSIF]
        remaining_partitions_local = _partition_indices_dirichlet_allclasses(  # [CLASSIF]
            y=remaining_y,  # [CLASSIF]
            num_partitions=num_remaining_partitions,  # [CLASSIF]
            alpha=alpha,  # [CLASSIF]
            seed=seed + 1,  # [CLASSIF] Seed diferente para evitar duplicação  # [CLASSIF]
            min_samples_per_class=min_samples_per_class,  # [CLASSIF]
        )  # [CLASSIF]
        
        # [CLASSIF] Mapeia índices locais (nos dados remanescentes) para índices globais  # [CLASSIF]
        remaining_partitions_global = [  # [CLASSIF]
            remaining_idx_mapping[part] if len(part) > 0 else np.array([], dtype=int)  # [CLASSIF]
            for part in remaining_partitions_local  # [CLASSIF]
        ]  # [CLASSIF]
    else:  # [CLASSIF]
        remaining_partitions_global = [np.array([], dtype=int) for _ in range(num_remaining_partitions)]  # [CLASSIF]
    
    # [CLASSIF] Monta as partições finais: cliente dominante + partições do resto  # [CLASSIF]
    parts = [None] * num_partitions  # [CLASSIF]
    parts[dominant_client_id] = dominant_part  # [CLASSIF]
    
    # [CLASSIF] Insere as partições restantes nos slots corretos (pulando o cliente dominante)  # [CLASSIF]
    remaining_idx = 0  # [CLASSIF]
    for pid in range(num_partitions):  # [CLASSIF]
        if pid != dominant_client_id:  # [CLASSIF]
            parts[pid] = remaining_partitions_global[remaining_idx]  # [CLASSIF]
            remaining_idx += 1  # [CLASSIF]
    
    return parts  # [CLASSIF]


def load_house_client(client_id: int):  # [CLASSIF]
    """
    [CLASSIF] Carrega a partição do dataset para um cliente específico (iid ou non-iid).
    
    Estratégia corrigida:
    1. Particiona TODOS OS ÍNDICES do dataset em train (80%) e test (20%)
    2. Depois particiona cada subset (train e test) entre os clientes
    3. Isto garante zero sobreposição entre train/test e entre clientes
    """  # [CLASSIF]
    X, y, feature_names, label_name = load_dataset()  # [CLASSIF]

    if client_id < 0 or client_id >= number_of_clients:  # [CLASSIF]
        raise ValueError(  # [CLASSIF]
            f"client_id {client_id} fora do intervalo [0, {number_of_clients - 1}]"  # [CLASSIF]
        )  # [CLASSIF]

    # [CLASSIF] PASSO 1: Dividir ÍNDICES em train/test GLOBALMENTE  # [CLASSIF]
    y_arr = np.asarray(y)  # [CLASSIF]
    n_samples = len(y_arr)  # [CLASSIF]
    
    # [CLASSIF] Criar índices de train/test usando stratificação  # [CLASSIF]
    all_indices = np.arange(n_samples)  # [CLASSIF]
    stratify_arg = y_arr  # [CLASSIF]
    uniq, cnt = np.unique(y_arr, return_counts=True)  # [CLASSIF]
    if uniq.size < 2 or np.min(cnt) < 2:  # [CLASSIF]
        stratify_arg = None  # [CLASSIF]
    
    # [CLASSIF] Split de índices (80% train, 20% test)  # [CLASSIF]
    train_indices_global, test_indices_global = train_test_split(  # [CLASSIF]
        all_indices, test_size=0.2, stratify=stratify_arg, random_state=int(partition_seed)  # [CLASSIF]
    )  # [CLASSIF]
    
    # [CLASSIF] PASSO 2: Particionar cada subset entre os clientes  # [CLASSIF]
    pt = partition_type.lower() if isinstance(partition_type, str) else "iid"  # [CLASSIF]
    
    y_train_global = y_arr[train_indices_global]  # [CLASSIF]
    y_test_global = y_arr[test_indices_global]  # [CLASSIF]
    
    # [CLASSIF] Particiona índices RELATIVOS a cada subset  # [CLASSIF]
    if pt == "iid":  # [CLASSIF]
        train_local_partitions = _partition_indices_iid(len(y_train_global), number_of_clients, seed=int(partition_seed))  # [CLASSIF]
        test_local_partitions = _partition_indices_iid(len(y_test_global), number_of_clients, seed=int(partition_seed) + 999)  # [CLASSIF]
    elif pt in ("non-iid", "non_iid", "noniid"):  # [CLASSIF]
        train_local_partitions = _partition_indices_dirichlet(  # [CLASSIF]
            y=y_train_global, num_partitions=number_of_clients, alpha=float(non_iid_alpha),  # [CLASSIF]
            seed=int(partition_seed), min_partition_size=1  # [CLASSIF]
        )  # [CLASSIF]
        test_local_partitions = _partition_indices_dirichlet(  # [CLASSIF]
            y=y_test_global, num_partitions=number_of_clients, alpha=float(non_iid_alpha),  # [CLASSIF]
            seed=int(partition_seed) + 999, min_partition_size=1  # [CLASSIF]
        )  # [CLASSIF]
    elif pt in ("non-iid-allclasses", "non_iid_allclasses", "noniid_allclasses"):  # [CLASSIF]
        train_local_partitions = _partition_indices_dirichlet_allclasses(  # [CLASSIF]
            y=y_train_global, num_partitions=number_of_clients, alpha=float(non_iid_alpha),  # [CLASSIF]
            seed=int(partition_seed), min_samples_per_class=int(min_samples_per_class)  # [CLASSIF]
        )  # [CLASSIF]
        test_local_partitions = _partition_indices_dirichlet_allclasses(  # [CLASSIF]
            y=y_test_global, num_partitions=number_of_clients, alpha=float(non_iid_alpha),  # [CLASSIF]
            seed=int(partition_seed) + 999, min_samples_per_class=int(min_samples_per_class)  # [CLASSIF]
        )  # [CLASSIF]
    elif pt in ("dominant-client", "dominant_client", "dominantclient"):  # [CLASSIF]
        from fedt.settings import dominant_client_id as dce_id, dominant_client_percentage as dce_pct  # [CLASSIF]
        train_local_partitions = _partition_indices_dominant_client(  # [CLASSIF]
            y=y_train_global, num_partitions=number_of_clients,  # [CLASSIF]
            dominant_client_id=int(dce_id), dominant_percentage=float(dce_pct),  # [CLASSIF]
            alpha=float(non_iid_alpha), seed=int(partition_seed),  # [CLASSIF]
            min_samples_per_class=int(min_samples_per_class)  # [CLASSIF]
        )  # [CLASSIF]
        test_local_partitions = _partition_indices_dominant_client(  # [CLASSIF]
            y=y_test_global, num_partitions=number_of_clients,  # [CLASSIF]
            dominant_client_id=int(dce_id), dominant_percentage=float(dce_pct),  # [CLASSIF]
            alpha=float(non_iid_alpha), seed=int(partition_seed) + 999,  # [CLASSIF]
            min_samples_per_class=int(min_samples_per_class)  # [CLASSIF]
        )  # [CLASSIF]
    else:  # [CLASSIF]
        raise ValueError(f"Tipo de particionamento inválido: {partition_type}")  # [CLASSIF]

    # [CLASSIF] PASSO 3: Mapear índices locais para índices globais  # [CLASSIF]
    train_local_indices_client = train_local_partitions[client_id]  # [CLASSIF]
    test_local_indices_client = test_local_partitions[client_id]  # [CLASSIF]
    
    # [CLASSIF] Converter de índices locais (dentro do subset) para índices globais (dataset completo)  # [CLASSIF]
    train_global_indices_client = train_indices_global[train_local_indices_client]  # [CLASSIF]
    test_global_indices_client = test_indices_global[test_local_indices_client]  # [CLASSIF]
    
    # [CLASSIF] Extrair dados using índices globais  # [CLASSIF]
    X_train = X[train_global_indices_client]  # [CLASSIF]
    y_train = y_arr[train_global_indices_client]  # [CLASSIF]
    X_test = X[test_global_indices_client]  # [CLASSIF]
    y_test = y_arr[test_global_indices_client]  # [CLASSIF]
    
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

def load_dataset_for_server(excluded_clients=None) -> list:
    """
    ### Função:
    Carregar um subconjunto mínimo do dataset que contenha todas as classes presentes no problema.
    Se excluded_clients for especificado, usa apenas dados dos clientes NÃO excluídos.
    
    ### Args:
    - excluded_clients: Set de IDs de clientes a excluir (ex: {0} para unlearning)
    
    ### Returns:
    - Data Train: As features.
    - Label Train: Os targets. 
    """
    # [UNLEARNING] Se há clientes excluídos, carrega dados apenas dos clientes restantes
    if excluded_clients is not None and len(excluded_clients) > 0:
        # Extrai dataset_name e partition_type das configurações
        dataset_name = Path(dataset_path).stem
        pt = partition_type.lower() if isinstance(partition_type, str) else "iid"
        
        partition_base = partitions_folder / dataset_name / pt
        
        # Coleta todos os dados de treino dos clientes NÃO excluídos
        all_train_data = []
        all_train_labels = []
        
        for client_id in range(number_of_clients):
            if client_id in excluded_clients:
                continue  # Pula o cliente excluído
            
            train_path = partition_base / f"client_{client_id}" / "train.csv"
            if train_path.exists():
                df = pd.read_csv(train_path)
                # Separa features e label
                y_col = df.columns[-1]  # Última coluna é o label
                X_client = df.drop(columns=[y_col]).values
                y_client = df[y_col].values
                
                all_train_data.append(X_client)
                all_train_labels.append(y_client)
        
        if all_train_data:
            # Concatena todos os dados
            X_train = np.vstack(all_train_data)
            y_train = np.concatenate(all_train_labels)
            
            # Para cada classe, escolhe um índice representativo (primeira ocorrência)
            unique_classes, first_indices = np.unique(y_train, return_index=True)
            
            return X_train[first_indices], y_train[first_indices]
        else:
            # Fallback: usa dataset completo se não houver partições
            pass
    
    # [CLASSIF] Comportamento padrão: usa dataset completo
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

def load_server_side_validation_data(excluded_clients=None):
    """
    ### Função:
    Carregar dados de validação do servidor. Se excluded_clients for especificado,
    usa apenas os dados dos clientes NÃO excluídos (para machine unlearning).
    
    ### Args:
    - excluded_clients: Set de IDs de clientes a excluir (ex: {0} para unlearning)
    
    ### Returns:
    - Data Valid: As features para validação.
    - Label Valid: Os targets para validação. 
    """
    # [UNLEARNING] Se há clientes excluídos, carrega dados apenas dos clientes restantes
    if excluded_clients is not None and len(excluded_clients) > 0:
        # Extrai dataset_name e partition_type das configurações
        dataset_name = Path(dataset_path).stem
        pt = partition_type.lower() if isinstance(partition_type, str) else "iid"
        
        partition_base = partitions_folder / dataset_name / pt
        
        # Coleta todos os dados de teste dos clientes NÃO excluídos
        all_test_data = []
        all_test_labels = []
        
        for client_id in range(number_of_clients):
            if client_id in excluded_clients:
                continue  # Pula o cliente excluído
            
            test_path = partition_base / f"client_{client_id}" / "test.csv"
            if test_path.exists():
                df = pd.read_csv(test_path)
                # Separa features e label
                y_col = df.columns[-1]  # Última coluna é o label
                X_client = df.drop(columns=[y_col]).values
                y_client = df[y_col].values
                
                all_test_data.append(X_client)
                all_test_labels.append(y_client)
        
        if all_test_data:
            # Concatena todos os dados
            X_valid = np.vstack(all_test_data)
            y_valid = np.concatenate(all_test_labels)
            
            logger = logging.getLogger("SERVER")
            logger.debug(f"[UNLEARNING] Carregadas {len(X_valid)} amostras de validação de {len(all_test_data)} clientes (excluindo {excluded_clients})")
            
            # Limita ao tamanho configurado (se maior)
            if len(X_valid) > validate_dataset_size:
                X_valid = X_valid[-validate_dataset_size:]
                y_valid = y_valid[-validate_dataset_size:]
                logger.debug(f"[UNLEARNING] Truncado para {validate_dataset_size} amostras (limite configurado)")
            
            return X_valid, y_valid
        else:
            # Fallback: usa dataset completo se não houver partições
            logger = logging.getLogger("SERVER")
            logger.warning(f"[UNLEARNING] Fallback: partições não encontradas, usando dataset completo")
            pass
    
    # [CLASSIF] Comportamento padrão: usa dataset completo
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
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')  # [WINDOWS] UTF-8 encoding
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
    """Retorna um dicionário {target_string: [Process, ...]} para cada target encontrado.
    
    Para compatibilidade Windows/Linux, procura pelos componentes da string target
    na linha de comando, ao invés de exigir correspondência exata.
    """
    matches = {t: [] for t in targets}
    for proc in psutil.process_iter(attrs=['pid', 'cmdline']):
        cmd = get_process_cmd(proc)
        if not cmd:
            continue
        for t in targets:
            # Busca exata ou flexível para compatibilidade Windows
            if t in cmd:
                matches[t].append(proc)
                break
            # Para "fedt run server", verifica se os componentes estão presentes
            elif t == "fedt run server":
                cmd_lower = cmd.lower()
                if "run" in cmd_lower and "server" in cmd_lower and "fedt" in cmd_lower:
                    matches[t].append(proc)
                    break
    return matches

def kill_processes(processes, name):
    for target, plist in list(processes.items()):
        for proc in list(plist):
            if proc.name() == name:
                os.kill(proc.pid, signal.SIGINT)


# ===== SHAP FUNCTIONS =====

def calculate_shap_values(model: RandomForestClassifier, X_data: np.ndarray, 
                         max_samples: int = 100) -> tuple:
    """
    Calcula SHAP values para explicabilidade do modelo RandomForest.
    
    Args:
        model: Modelo RandomForestClassifier treinado
        X_data: Dados para calcular SHAP values (features)
        max_samples: Número máximo de amostras para usar (para eficiência)
    
    Returns:
        tuple: (shap_values, explainer, X_sample)
    """
    try:
        import shap
    except ImportError:
        logging.error("SHAP não está instalado. Instale com: pip install shap")
        return None, None, None
    
    # Limitar amostras para eficiência
    if len(X_data) > max_samples:
        sample_indices = np.random.choice(len(X_data), max_samples, replace=False)
        X_sample = X_data[sample_indices]
    else:
        X_sample = X_data
    
    try:
        # Usar TreeExplainer para RandomForest (mais eficiente que KernelExplainer)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        return shap_values, explainer, X_sample
        
    except Exception as e:
        logging.error(f"Erro ao calcular SHAP values: {e}")
        return None, None, None


def save_shap_summary(shap_values, X_sample, feature_names: list, output_path: Path, 
                     plot_type: str = "bar", class_idx: int = None):
    """
    Salva gráfico de resumo SHAP (importância das features).
    
    Args:
        shap_values: SHAP values calculados
        X_sample: Dados de entrada usados para calcular SHAP (para visualizar gradiente)
        feature_names: Nomes das features
        output_path: Caminho para salvar a figura
        plot_type: Tipo de plot ("bar", "beeswarm")
        class_idx: Para multi-classe, qual classe plotar (None = agregado)
    """
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        logging.error("SHAP ou matplotlib não estão instalados")
        return
    
    try:
        plt.figure(figsize=(12, 8))
        
        # Processar SHAP values (multi-classe ou binário)
        if isinstance(shap_values, list):
            # Multi-class case
            if class_idx is not None:
                # Plotar classe específica
                shap_vals_to_plot = shap_values[class_idx]
            else:
                # Para bar plot: agregar (média do abs(SHAP) entre classes)
                if plot_type == "bar":
                    shap_vals_to_plot = np.mean(np.abs(shap_values), axis=0)
                else:
                    # Para beeswarm sem class_idx especificado, usar classe 0
                    shap_vals_to_plot = shap_values[0]
        else:
            shap_vals_to_plot = shap_values
        
        if plot_type == "bar":
            shap.summary_plot(shap_vals_to_plot, feature_names=feature_names, 
                            plot_type="bar", show=False)
        else:
            # Beeswarm com features para visualizar gradiente de cores
            shap.summary_plot(shap_vals_to_plot, features=X_sample, feature_names=feature_names, 
                            show=False)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"SHAP summary plot salvo em: {output_path}")
    except Exception as e:
        logging.error(f"Erro ao salvar SHAP summary plot: {e}")


def save_shap_values_json(shap_values, feature_names: list, output_path: Path):
    """
    Salva SHAP values em formato JSON para análise posterior.
    
    Args:
        shap_values: SHAP values calculados
        feature_names: Nomes das features
        output_path: Caminho para salvar JSON
    """
    try:
        import shap
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Converter para formato salável
        if isinstance(shap_values, list):
            # Multi-class case - salvar para cada classe
            shap_data = {}
            for class_idx, sv in enumerate(shap_values):
                shap_data[f"class_{class_idx}"] = {
                    "shap_values": sv.tolist() if hasattr(sv, 'tolist') else sv,
                    "feature_names": feature_names
                }
        else:
            shap_data = {
                "shap_values": shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
                "feature_names": feature_names
            }
        
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(shap_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"SHAP values salvos em: {output_path}")
    except Exception as e:
        logging.error(f"Erro ao salvar SHAP values JSON: {e}")


def get_feature_names_from_dataset():
    """
    Obtém nomes das features do dataset (excluindo coluna de rótulo).
    
    Returns:
        list: Nomes das features
    """
    try:
        df = pd.read_csv(dataset_path, nrows=0)
        feature_names = [col for col in df.columns if col != str(label_target)]
        return feature_names
    except Exception as e:
        logging.error(f"Erro ao obter nomes das features: {e}")
        return None


if __name__ == "__main__":
    create_specific_result_folder("Client")