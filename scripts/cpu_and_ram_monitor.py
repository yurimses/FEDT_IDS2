import psutil
import time
import json

from fedt.utils import create_specific_logs_folder, setup_logger, get_process_cmd, find_target_processes

from pathlib import Path
import logging
import argparse

parse = argparse.ArgumentParser(description="Script para monitorar o consumo de ram e cpu.")
parse.add_argument(
    "--strategy",
    type=str,
    default=None,
    help="É a estrátegia que está rodando no momento."
)
parse.add_argument(
    "--sim-number",
    type=int,
    default=None,
    help="É o número da simulação."
)
parse.add_argument(
    "--user",
    type=str,
    default=None,
    help="Quem está rodando."
)
parse.add_argument(
    "--pid",
    type=int,
    default=None,
    help="PID específico a monitorar. Se definido, ignora TARGET_STRINGS."
)

args = parse.parse_args()
strategy = args.strategy
simulation_number = args.sim_number
user = args.user
specific_pid = args.pid

logger = setup_logger(
    name="CPU_RAM",
    log_file="cpu_ram.log",
    level=logging.INFO
)

logs_folder = create_specific_logs_folder(strategy, "cpu_ram")

# Lista de padrões a monitorar (usado apenas quando pid não é fornecido)
TARGET_STRINGS = ["--client-id", "fedt run server"]
LOG_FILE = logs_folder / f"cpu_and_ram_{user}_{strategy}_{simulation_number}.json"
CHECK_INTERVAL = 0.5
SAVE_INTERVAL = 50


def monitor_specific_pid(pid):
    """Monitorar apenas um PID específico."""
    logger.info(f"Monitorando somente o PID {pid}")

    data = {pid: []}
    iteration_count = 0

    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        logger.error(f"PID {pid} não existe.")
        return

    logger.warning(f"Processo encontrado: PID={pid}, CMD={proc.cmdline()}")

    while proc.is_running():
        try:
            cpu = proc.cpu_percent(interval=None)
            mem = proc.memory_info().rss / (1024 * 1024)
            threads = proc.num_threads()
            timestamp = time.time()

            data[pid].append({
                "timestamp": timestamp,
                "cpu_percent": cpu,
                "memory_mb": mem,
                "num_threads": threads
            })

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break

        iteration_count += 1

        if iteration_count % SAVE_INTERVAL == 0:
            with open(LOG_FILE, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"JSON atualizado ({iteration_count} iterações).")

        time.sleep(CHECK_INTERVAL)

    # Salva final
    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Processo finalizado. Monitoramento encerrado.")
    logger.info(f"Resultados salvos em '{LOG_FILE}'")


def monitor_by_patterns():
    """Versão original que monitora vários processos por TARGET_STRINGS."""
    logger.info(f"Aguardando processos com {TARGET_STRINGS} no comando...")

    processes = {}
    while not any(processes.values()):
        processes = find_target_processes(TARGET_STRINGS)
        if not any(processes.values()):
            time.sleep(CHECK_INTERVAL)

    data = {t: {} for t in TARGET_STRINGS}
    iteration_count = 0

    active_pids = [p.pid for plist in processes.values() for p in plist]
    logger.warning(f"Processos encontrados: {active_pids}")

    while any(processes.values()):
        for target, plist in list(processes.items()):
            for proc in list(plist):
                try:
                    if not proc.is_running():
                        plist.remove(proc)
                        continue

                    pid = proc.pid
                    cpu = proc.cpu_percent(interval=None)
                    mem = proc.memory_info().rss / (1024 * 1024)
                    threads = proc.num_threads()
                    timestamp = time.time()

                    if pid not in data[target]:
                        data[target][pid] = []

                    data[target][pid].append({
                        "timestamp": timestamp,
                        "cpu_percent": cpu,
                        "memory_mb": mem,
                        "num_threads": threads
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    if proc in plist:
                        plist.remove(proc)

        current_pids = {p.pid for plist in processes.values() for p in plist}
        new_matches = find_target_processes(TARGET_STRINGS)

        for target, new_list in new_matches.items():
            for new_proc in new_list:
                if new_proc.pid not in current_pids:
                    processes[target].append(new_proc)
                    logger.warning(f"Novo processo detectado ({target}): PID {new_proc.pid}")

        iteration_count += 1

        if iteration_count % SAVE_INTERVAL == 0:
            with open(LOG_FILE, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"JSON atualizado ({iteration_count} iterações).")

        time.sleep(CHECK_INTERVAL)

    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Todos os processos finalizados. Monitoramento encerrado.")
    logger.info(f"Resultados salvos em '{LOG_FILE}'")


def main():
    if specific_pid is not None:
        monitor_specific_pid(specific_pid)
    else:
        monitor_by_patterns()


if __name__ == "__main__":
    main()
