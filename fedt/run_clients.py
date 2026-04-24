from fedt.settings import number_of_clients, client_script_path
from fedt import utils

import subprocess
import time
import os

def run_clients(require_prepared_partitions: bool = True):
    processes = []

    if require_prepared_partitions:
        utils.validate_prepared_partitions(expected_num_clients=number_of_clients)
    
    for i in range(number_of_clients):
        cmd = ["python3", client_script_path, "--client-id", str(i)]
        
        # inicia o processo no diretório especificado
        p = subprocess.Popen(cmd)
        processes.append(p)
        
        # espera 5 segundos antes de iniciar o próximo
        time.sleep(5)

    # espera todos terminarem (equivalente a `wait` no bash)
    for p in processes:
        p.wait()

def run_clients_with_a_specific_strategy(input_aggregation_strategy):
    processes = []

    # Validação obrigatória: sem partições válidas a execução falha antes de iniciar clientes.
    utils.validate_prepared_partitions(expected_num_clients=number_of_clients)
    
    for i in range(number_of_clients):
        cmd = ["python3", client_script_path, "--client-id", str(i), "--strategy", input_aggregation_strategy]
        
        # inicia o processo no diretório especificado
        p = subprocess.Popen(cmd)
        processes.append(p)
        
        # espera 5 segundos antes de iniciar o próximo
        time.sleep(5)

    # espera todos terminarem (equivalente a `wait` no bash)
    for p in processes:
        p.wait()

if __name__ == "__main__":
    run_clients()