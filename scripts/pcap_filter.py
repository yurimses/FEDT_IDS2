from fedt.settings import logs_folder, scripts_folder
from fedt.utils import create_specific_logs_folder, setup_logger
import subprocess
import logging
from pathlib import Path
import subprocess

# [x] Ler a pasta resultados e verificar quais pastas estão lá.
# [ ] Ler as pastas de cada estrátegia e ver quais são os dados.
# [ ] Carregar cada arquivo. 
# [ ] Converter o arquivo e salvar.

logger = setup_logger(
    name="NETWORK_CSV",
    log_file="network_csv.log",
    level=logging.INFO
)

script_path = scripts_folder / "pcap_filter"

logs_folder = logs_folder / "network"
strategies_pcap_folder = [path for path in logs_folder.iterdir() if path.is_dir()]

logger.info(f"Pastas encontradas: {[strategy_path.name for strategy_path in strategies_pcap_folder]}")

for strategy_path in strategies_pcap_folder:
    logger.warning(f"Iniciando a conversão para a estrátegia: {strategy_path.name}")

    filtered_folder = create_specific_logs_folder(strategy_path.name, "network_csv")

    pcap_files = list(strategy_path.glob("*.pcap"))

    for file_path in pcap_files:
        logger.info(f"Convertendo o arquivo {file_path} em csv")

        filtered_file_path = filtered_folder / f"{file_path.stem}.csv"

        pcap_filter_proc = subprocess.Popen([
            script_path,
            filtered_file_path
        ])
        pcap_filter_proc.wait()

logger.warning("Fim da execução.")