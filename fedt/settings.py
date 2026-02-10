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
partitions_folder = (base_path / "partitions").resolve()

number_of_jobs = config["settings"]["number_of_jobs"]
number_of_clients = config["settings"]["number_of_clients"]
number_of_rounds = config["settings"]["number_of_rounds"]
imported_aggregation_strategy = config["settings"]["aggregation_strategy"]

many_simulations = config["settings"]["sequence"]["many_simulations"]
number_of_simulations = config["settings"]["sequence"]["number_of_simulations"]
aggregation_strategies = config["settings"]["sequence"]["aggregation_strategies"]

client_timeout = config["settings"]["client"]["timeout"]
client_debug = config["settings"]["client"]["debug"]
print_class_distribution = config["settings"]["client"].get("print_class_distribution", False)  # [CLASSIF]

server_config = config["settings"]["server"]
server_ip = config["settings"]["server"]["IP"]
server_port = config["settings"]["server"]["port"]
validate_dataset_size = config["settings"]["server"]["validate_dataset_size"]

train_test_split_size = config["dataset"]["train_test_split_size"]
label_target = config["dataset"].get("label_target", "Attack_label")  # [CLASS]
partition_type = config["dataset"].get("partition_type", "iid")  # [CLASSIF]
non_iid_alpha = config["dataset"].get("non_iid_alpha", 0.3)  # [CLASSIF]
min_samples_per_class = config["dataset"].get("min_samples_per_class", 10)  # [CLASSIF]
partition_seed = config["dataset"].get("partition_seed", 42)  # Seed para replicabilidade
# [CLASSIF] Parâmetros para estratégia 'dominant_client'
dominant_client_id = config["dataset"].get("dominant_client_id", 0)  # [CLASSIF]
dominant_client_percentage = config["dataset"].get("dominant_client_percentage", 0.7)  # [CLASSIF]

# [CLASSIF] Hiperparâmetros de poda para as árvores de decisão da RandomForest
max_depth = config["settings"]["classification"]["max_depth"]  # [CLASSIF] profundidade máxima das árvores (pre-pruning)
min_samples_leaf = config["settings"]["classification"]["min_samples_leaf"]  # [CLASSIF] mínimo de amostras por folha (pre-pruning)
min_samples_split = config["settings"]["classification"]["min_samples_split"]  # [CLASSIF] mínimo de amostras para dividir um nó (pre-pruning)
max_features = config["settings"]["classification"]["max_features"]  # [CLASSIF] fração de atributos considerados em cada split
ccp_alpha = config["settings"]["classification"]["ccp_alpha"]  # [CLASSIF] parâmetro de poda por custo-complexidade (post-pruning)


network_interface = config["scripts"]["network_interface"]

# [CLASSIF] Extração automática das classes do dataset
def _get_all_labels():
    """[CLASSIF] Extrai automaticamente todas as classes únicas do dataset.
    
    Retorna uma lista ordenada de labels [0, 1, 2, ..., N-1] onde N é o número
    de classes no dataset. Esta função lê apenas a coluna de labels do dataset
    para minimizar o uso de memória.
    
    O retorno é sempre numérico (range de 0 a N-1) porque o código usa
    LabelEncoder para converter labels string em inteiros sequenciais.
    
    Returns:
        list: Lista ordenada de todos os labels únicos do dataset como inteiros [0, 1, ..., N-1]
    """
    try:
        import pandas as pd
        # Lê apenas a coluna de labels para economizar memória
        df = pd.read_csv(dataset_path, usecols=[label_target])
        
        # Conta o número de classes únicas
        n_classes = df[label_target].nunique()
        
        # Retorna range [0, 1, 2, ..., n_classes-1]
        # Isso corresponde ao que LabelEncoder faz em utils.load_dataset()
        return list(range(n_classes))
    except Exception as e:
        # Fallback: se houver erro, retorna None e deixa o código decidir
        import warnings
        warnings.warn(
            f"Não foi possível extrair ALL_LABELS automaticamente: {e}. "
            f"Considere definir ALL_LABELS manualmente.",
            RuntimeWarning
        )
        return None

# [CLASSIF] ALL_LABELS: conjunto global de labels para métricas comparáveis
# É inferido automaticamente do dataset configurado em config.toml
ALL_LABELS = _get_all_labels()