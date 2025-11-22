from fedt.settings import (
    dataset_path, percentage_value_of_samples_per_client, 
    validate_dataset_size, aggregation_strategies, 
    results_folder, logs_folder
    )

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

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

def set_initial_params(model: RandomForestRegressor, X_train, y_train):
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
    model: RandomForestRegressor, params: list
) -> RandomForestRegressor:
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

def get_model_parameters(model: RandomForestRegressor):
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
    """
    energy_data_complete = pd.read_csv(dataset_path)
    columns_for_training = []
    temperature_columns = [f"T{i}" for i in range(1, 10)]
    humidity_columns = [f"RH_{i}" for i in range(1, 10)]

    for temperature in temperature_columns:
        columns_for_training.append(temperature)
        
    for humidity in humidity_columns:
        columns_for_training.append(humidity)
        
    columns_for_training.append("T_out")
    columns_for_training.append("RH_out")
    columns_for_training.append("Press_mm_hg")
    columns_for_training.append("Visibility")

    data = energy_data_complete[columns_for_training]
    label = energy_data_complete["Appliances"]
    
    return data, label

def load_house_client():
    rng = np.random.default_rng()

    X, y = load_dataset()

    number_of_samples = int((len(X)*percentage_value_of_samples_per_client)/100)

    idxs = rng.choice(X.shape[0], size=number_of_samples, replace=False)
    X = X.iloc[idxs]
    y = y.iloc[idxs]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, y_train, X_test, y_test

def load_dataset_for_server() -> list:
    """
    ### Função:
    Carregar o dataset com apenas 3 amostras, 
    servirá para inicializar o server e garantir que os parâmetros entre server e cliente serão compativeis.
    ### Args:
    - None.
    ### Returns:
    - Data Train: As features.
    - Label Train: Os targets. 
    """
    data, label  = load_dataset()

    data_train, _, label_train, _ = train_test_split(data, label, test_size=0.2)

    return data_train[0:2], label_train[0:2]

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
    data, label  = load_dataset()

    _, data_valid, _, label_valid = train_test_split(data, label, test_size=0.2)
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

def create_specific_logs_folder(strategy, base_name):
    subpath = logs_folder / base_name / strategy
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