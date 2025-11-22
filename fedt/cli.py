import asyncio
import argparse
import time

from fedt.server import run_server
from fedt.run_clients import run_clients, run_clients_with_a_specific_strategy
from fedt.settings import aggregation_strategies, number_of_simulations
from fedt.utils import find_target_processes, kill_processes

import subprocess, signal, os
from multiprocessing import Process

def cmd_server():
    return asyncio.run(run_server())
def cmd_server_with_args(strategy):
    return asyncio.run(run_server(strategy))

def run_server_many_times():
    for strategy in aggregation_strategies:
        for i in range(number_of_simulations):
            print(f"Iniciando o servidor... Simulação: {i}")
            net_proc = subprocess.Popen(
                ["fedt-network", "--strategy", f"{strategy}", "--sim-number", f"{i}", "--user", "server"],
                stdout=subprocess.PIPE,
                text=True
            )
            tcpdump_output = net_proc.stdout.readline().strip()

            time.sleep(3)

            server_proc = Process(
                target=cmd_server_with_args, 
                args=(strategy,)
            )
            server_proc.start()

            cpu_ram_proc = subprocess.Popen([
                "fedt-cpu-ram", 
                "--strategy", f"{strategy}", 
                "--sim-number", f"{i}", 
                "--user", "server", 
                "--pid", f"{server_proc.pid}"])

            server_proc.join()

            tcpdump_processes = find_target_processes([tcpdump_output])
            kill_processes(tcpdump_processes, "tcpdump")

            cpu_ram_proc.wait()
            net_proc.wait()
            print("Server finalizado, pausa de 10 segundos...")
            time.sleep(10)

def run_clients_many_times():
    for strategy in aggregation_strategies:
        for i in range(number_of_simulations):
            print(f"Iniciando os clientes... Simulação: {i}")
            cpu_ram_proc = subprocess.Popen(["fedt-cpu-ram", "--strategy", f"{strategy}", "--sim-number", f"{i}", "--user", "client"])
            net_proc = subprocess.Popen(
                ["fedt-network", "--strategy", f"{strategy}", "--sim-number", f"{i}", "--user", "client"],
                stdout=subprocess.PIPE,
                text=True
                )
            tcpdump_output = net_proc.stdout.readline().strip()

            time.sleep(3)

            run_clients_with_a_specific_strategy(strategy)

            tcpdump_processes = find_target_processes([tcpdump_output])
            kill_processes(tcpdump_processes, "tcpdump")

            cpu_ram_proc.wait()
            net_proc.wait()
            print("Clientes finalizados, pausa de 30 segundos...")
            time.sleep(30)


def run_server_and_clients():
    print("Iniciando servidor...")
    server_proc = subprocess.Popen(["fedt", "run", "server"])

    time.sleep(3)  

    print("Iniciando clientes...")
    clients_proc = subprocess.Popen(["fedt", "run", "clients"])

    server_proc.wait()
    clients_proc.wait()

def main():
    parser = argparse.ArgumentParser(
        description="fedt: Federated Learning for Decision Trees"
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcomandos")

    # Subcomando principal: run
    run_parser = subparsers.add_parser("run", help="Roda a simulação")
    run_subparsers = run_parser.add_subparsers(dest="target", help="")

    # Subcomando: run server
    run_server_parser = run_subparsers.add_parser("server", help="Roda o servidor")
    run_server_parser.set_defaults(func=cmd_server)

    # Subcomando: run clients
    run_clients_parser = run_subparsers.add_parser("clients", help="Roda os clientes")
    run_clients_parser.set_defaults(func=run_clients)

    # Subcomando: run many-serverr
    run_many_server_parser = run_subparsers.add_parser(
        "many-server", help="Roda vários servidores em sequência"
    )
    run_many_server_parser.set_defaults(func=run_server_many_times)

    # Subcomando: run many-clients
    run_many_clients_parser = run_subparsers.add_parser(
        "many-clients", help="Roda vários clientes em sequência"
    )
    run_many_clients_parser.set_defaults(func=run_clients_many_times)


    # Define o comportamento padrão de "run" sem subcomando
    run_parser.set_defaults(func=run_server_and_clients)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()