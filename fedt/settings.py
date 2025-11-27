import tomllib
from pathlib import Path
import importlib.resources as res

def load_config():
    # 1) Tenta via importlib.resources (pacote instalado)
    try:
        pkg_files = res.files("fedt")
        with pkg_files.joinpath("config.toml").open("rb") as f:
            return tomllib.load(f)
    except (FileNotFoundError, NotADirectoryError, ModuleNotFoundError):
        # Cai aqui no caso de erro que você está vendo
        pass

    # 2) Fallback: arquivo ao lado de settings.py
    here = Path(__file__).resolve().parent
    candidates = [
        here / "config.toml",      # fedt/config.toml
        here.parent / "config.toml"  # config.toml na raiz do projeto
    ]
    for c in candidates:
        if c.is_file():
            with c.open("rb") as f:
                return tomllib.load(f)

    raise RuntimeError(
        "config.toml não encontrado. "
        "Coloque o arquivo dentro do pacote 'fedt' ou na raiz do projeto."
    )

config = load_config()

# Usa o base_path definido no config (ou '.' se não existir)
base_path = Path(config["paths"].get("base_path", ".")).expanduser().resolve()

# Usa o base_path definido no config (ou '.' se não existir)
base_path = Path(config["paths"].get("base_path", ".")).expanduser().resolve()

results_folder = (base_path / config["paths"]["results_folder"]).resolve()
logs_folder = (base_path / config["paths"]["logs_folder"]).resolve()
scripts_folder = (base_path / config["paths"]["scripts_path"]).resolve()
client_script_path = (base_path / config["paths"]["client_script_path"]).resolve()
dataset_path = (base_path / config["paths"]["dataset_path"]).resolve()

number_of_jobs = config["settings"]["number_of_jobs"]
number_of_clients = config["settings"]["number_of_clients"]
number_of_rounds = config["settings"]["number_of_rounds"]
imported_aggregation_strategy = config["settings"]["aggregation_strategy"]

many_simulations = config["settings"]["sequence"]["many_simulations"]
number_of_simulations = config["settings"]["sequence"]["number_of_simulations"]
aggregation_strategies = config["settings"]["sequence"]["aggregation_strategies"]

client_timeout = config["settings"]["client"]["timeout"]
client_debug = config["settings"]["client"]["debug"]

server_config = config["settings"]["server"]
server_ip = config["settings"]["server"]["IP"]
server_port = config["settings"]["server"]["port"]
validate_dataset_size = config["settings"]["server"]["validate_dataset_size"]

train_test_split_size = config["dataset"]["train_test_split_size"]
percentage_value_of_samples_per_client = config["dataset"]["percentage_value_of_samples_per_client"]
label_target = config["dataset"].get("label_target", "Attack_label")  # [CLASS]

# [CLASSIF] Hiperparâmetros de poda para as árvores de decisão da RandomForest
max_depth = config["settings"]["classification"]["max_depth"]  # [CLASSIF] profundidade máxima das árvores (pre-pruning)
min_samples_leaf = config["settings"]["classification"]["min_samples_leaf"]  # [CLASSIF] mínimo de amostras por folha (pre-pruning)
min_samples_split = config["settings"]["classification"]["min_samples_split"]  # [CLASSIF] mínimo de amostras para dividir um nó (pre-pruning)
max_features = config["settings"]["classification"]["max_features"]  # [CLASSIF] fração de atributos considerados em cada split
ccp_alpha = config["settings"]["classification"]["ccp_alpha"]  # [CLASSIF] parâmetro de poda por custo-complexidade (post-pruning)


network_interface = config["scripts"]["network_interface"]