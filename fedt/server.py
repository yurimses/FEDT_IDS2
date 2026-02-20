import asyncio
import logging
import time
import json
import os
import gc

import grpc
import grpc.aio as grpc_aio

from sklearn.ensemble import RandomForestClassifier #[CLASSIF]

from fedt.settings import (
    server_config, number_of_jobs, number_of_clients, 
    imported_aggregation_strategy, number_of_rounds, many_simulations,
    max_depth, min_samples_leaf, min_samples_split, max_features, ccp_alpha,  # [CLASSIF]
    dominant_client_id, unlearning_enabled, unlearning_round  # [UNLEARNING]
)
from fedt.fedforest import FedForest
from fedt import utils
from fedt.utils import create_specific_result_folder
from fedt import fedT_pb2
from fedt import fedT_pb2_grpc

import warnings
from scipy.stats import ConstantInputWarning

from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore", category=ConstantInputWarning)

# Configuração do log:
log_level = logging.DEBUG if server_config["debug"] else logging.INFO
logger = utils.setup_logger(
    name="SERVER",
    log_file="fedt_server.log",
    level=log_level
)

# TO-DO:
# - [ ] Adicionar o novo método de seleção de número de árvores dos clientes.

def add_end_time(runtime_clients, ID, end_time):
    for i, (client_id, start_time) in enumerate(runtime_clients):
        if client_id == ID:
            runtime_clients[i] = (client_id, (start_time, end_time))
            break
    return runtime_clients

def average_runtime(runtime_clients):
    """Calcula o tempo médio de execução."""
    runtime_list = [(end - start) for (_, (start, end)) in runtime_clients]
    runtime_sum = sum(runtime_list)
    runtime_average = runtime_sum / number_of_clients
    return runtime_average


class FedT(fedT_pb2_grpc.FedTServicer):
    def __init__(self, input_aggregation_strategy=None) -> None:
        super().__init__()

        self.aggregation_strategy = imported_aggregation_strategy
        if many_simulations:
            self.aggregation_strategy = input_aggregation_strategy

        # --- Unlearning config --- [UNLEARNING]
        self.unlearning_enabled = unlearning_enabled
        self.dominant_client_id = dominant_client_id
        self.unlearning_round = unlearning_round
        self.unlearning_done = False
        self.blocked_clients = set()

        # Extrai dataset_name e partition_type das configurações
        from pathlib import Path
        from fedt.settings import dataset_path, partition_type as pt_setting
        
        dataset_name = Path(dataset_path).stem
        partition_type_str = pt_setting.lower() if isinstance(pt_setting, str) else "iid"

        base_file_name = f"{self.aggregation_strategy}_server"
        self.results_folder = utils.create_specific_result_folder_with_dataset(
            self.aggregation_strategy, 
            dataset_name, 
            partition_type_str, 
            "server"
        )
        existing_files = [
            file for file in os.listdir(self.results_folder)
            if file.startswith(base_file_name) and file.endswith(".json")
        ]
        next_file_index = len(existing_files) + 1
        result_file_name = f"{base_file_name}_{next_file_index}.json"
        self.result_file_path = (self.results_folder / result_file_name).resolve()

        logger.warning(f"Result path: {self.result_file_path}")

        self.lock = asyncio.Lock()
        self.aggregation_done = asyncio.Event()

        self.round = 0
        self.aggregation_realised = 0 # 0 waiting, 1 aggregating, 2 done.

        self.clientes_conectados = []
        self.clientes_esperados = number_of_clients
        self.clientes_respondidos = 0
        self.trees_warehouse = []
        self.runtime_clients = []
        self.aggregation_time = 0.0

        self._supervisor_started = False
        self.shutdown_event = None

        self.executor = ThreadPoolExecutor(max_workers=number_of_jobs)

        self.model = RandomForestClassifier( # [CLASSIF]
            n_estimators=self.get_number_of_trees_per_client(),
            max_depth=max_depth,  # [CLASSIF]
            min_samples_leaf=min_samples_leaf,  # [CLASSIF]
            min_samples_split=min_samples_split,  # [CLASSIF]
            max_features=max_features,  # [CLASSIF]
            ccp_alpha=ccp_alpha,  # [CLASSIF]
            class_weight='balanced_subsample',  # [CLASSIF] Balanceia automaticamente classes desbalanceadas
            #warm_start=True  # [CLASSIF] Mantém árvores existentes ao chamar fit() novamente (não crítico aqui, pois sobrescrevemos estimators_)
        )
        # [UNLEARNING] Passa blocked_clients para excluir dados desses clientes
        data_train, label_train = utils.load_dataset_for_server(excluded_clients=self.blocked_clients if self.blocked_clients else None)
        utils.set_initial_params(self.model, data_train, label_train)

        self.global_trees = self.model.estimators_
        # [UNLEARNING] Passa blocked_clients para FedForest evitar vazamento em validação
        self.strategy = FedForest(self.model, excluded_clients=self.blocked_clients if self.blocked_clients else None)

    def attach_shutdown_event(self, event):
        self.shutdown_event = event

    def get_number_of_trees_per_client(self):
        return self.round * server_config["increase_of_trees_per_round"] + server_config["number_of_trees_in_start"]

    def aggregate_strategy(self, best_forests: list[RandomForestClassifier], threshold=server_config["mcc_threshold"]): # [CLASSIF]
        # --- Unlearning: remove trees from dominant client after unlearning_round --- [UNLEARNING]
        if self.unlearning_enabled and self.round >= self.unlearning_round and not self.unlearning_done:
            # Remove all trees from dominant client
            best_forests = [forest for idx, forest in enumerate(best_forests)
                           if self.trees_warehouse[idx][0] != self.dominant_client_id]
            self.blocked_clients.add(self.dominant_client_id)
            # [UNLEARNING] Atualiza excluded_clients em FedForest para evitar vazamento em validação
            self.strategy.excluded_clients = self.blocked_clients
            self.unlearning_done = True
            logger.warning(f"[UNLEARNING] Árvores do cliente dominante (ID={self.dominant_client_id}) removidas após round {self.round}.")

        match self.aggregation_strategy:
            case 'random':
                self.model.estimators_ = self.strategy.aggregate_fit_random_trees_strategy(best_forests)
            case 'best_trees':
                self.model.estimators_ = self.strategy.aggregate_fit_best_trees_strategy(best_forests)
            case 'threshold':
                self.model.estimators_ = self.strategy.aggregate_fit_best_trees_threshold_strategy(best_forests, threshold)
            case 'best_forests':
                self.model.estimators_ = self.strategy.aggregate_fit_best_forest_strategy(best_forests)
            case 'class_coverage':
                # Nova estratégia: garante cobertura por classe
                trees_per_class = server_config.get("trees_per_class", 3)
                total_trees_ratio = server_config.get("total_trees_ratio", 0.5)
                self.model.estimators_ = self.strategy.aggregate_fit_best_trees_with_class_coverage_strategy(
                    best_forests, 
                    trees_per_class=trees_per_class,
                    total_trees_ratio=total_trees_ratio
                )
            case _:
                self.model.estimators_ = self.strategy.aggregate_fit_random_trees_strategy(best_forests)

    async def _supervisor_task(self):
        while True:
            await asyncio.sleep(0.2)

            async with self.lock:
                # --- Unlearning: ajusta clientes_esperados dinamicamente no início do round --- [UNLEARNING]
                clientes_esperados_atual = self.clientes_esperados
                if self.unlearning_enabled and self.round >= self.unlearning_round:
                    clientes_esperados_atual = number_of_clients - 1
                
                enough = ( len(self.trees_warehouse) >= clientes_esperados_atual )
                should_start = ( self.aggregation_realised == 0 and enough )

                if should_start:
                    self.aggregation_realised = 1
                    break

        logger.info(f"Supervisor iniciando agregação, round {self.round}")

        forests = [trees for (_, trees) in self.trees_warehouse]
        start_time = time.time()

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self.executor, self.aggregate_strategy, forests)

            self.aggregation_time = time.time() - start_time
            logger.info(f"Agregação finalizada para o round {self.round}")
        except Exception as error:
            logger.critical(f"Erro na agregação: {error}")

        async with self.lock:
            self.aggregation_realised = 2
            self.aggregation_done.set()


    async def aggregate_trees(self, request_iterator, context):
        client_serialised_trees = []
        client_ID = None

        logger.info(f"Recebendo as árvores dos clientes, Round: {self.round}, Árvores por Cliente: {self.get_number_of_trees_per_client()}")

        async for request in request_iterator:
            client_ID = request.client_ID
            # --- Unlearning: bloqueia comunicação do cliente dominante após round --- [UNLEARNING]
            if self.unlearning_enabled and self.round >= self.unlearning_round and client_ID == self.dominant_client_id:
                logger.warning(f"[UNLEARNING] Ignorando árvores do cliente dominante (ID={client_ID}) após round {self.round}.")
                return  # Não processa mais árvores desse cliente
            client_serialised_trees.append(request.serialised_tree)

        loop = asyncio.get_running_loop()
        client_trees = await loop.run_in_executor(
            self.executor,
            utils.deserialise_several_trees,
            client_serialised_trees
        )
        
        async with self.lock:
            if client_ID not in self.clientes_conectados:
                self.clientes_conectados.append(client_ID)
            self.trees_warehouse.append((client_ID, client_trees))

            logger.debug(f"O cliente {client_ID} enviou {len(client_trees)} árvores.")
            logger.info(f"Clientes conectados {len(self.clientes_conectados)}/{self.clientes_esperados}")

            if not self._supervisor_started:
                self._supervisor_started = True
                asyncio.create_task(self._supervisor_task())

        await self.aggregation_done.wait()

        serialised_global_trees = await loop.run_in_executor(
            self.executor, 
            utils.serialise_several_trees, 
            self.model.estimators_
        )
        number_of_trees = len(serialised_global_trees)
        number_of_sended_trees = 0

        server_reply = fedT_pb2.Forest_Server()
        for tree in serialised_global_trees:
            number_of_sended_trees += 1
            if number_of_sended_trees % server_config["print_every_trees_sent"] == 0:
                logger.info(f"Client ID: {client_ID}. Àrvore {number_of_sended_trees} de {number_of_trees} enviada.")
            server_reply.serialised_tree = tree
            yield server_reply

    async def get_server_model(self, request, context):
        start_time = time.time()

        self.runtime_clients.append([request.client_ID, start_time])
        logger.info(f"Client ID: {request.client_ID}, requisitando o modelo do servidor.")
        
        loop = asyncio.get_running_loop()
        trees = utils.get_model_parameters(self.model)
        serialised_trees = await loop.run_in_executor(
            self.executor, 
            utils.serialise_several_trees, 
            trees
        )
        
        server_message = fedT_pb2.Forest_Server()
        for serialise_tree in serialised_trees:
            server_message.serialised_tree = serialise_tree
            yield server_message

    async def get_server_settings(self, request, context):
        logger.debug(f"Client ID: {request.client_ID}, solicitando as configurações.")
        return fedT_pb2.Server_Settings(
            trees_by_client=self.get_number_of_trees_per_client(), 
            current_round=self.round
        )

    async def end_of_transmission(self, request, context):
        end_time = time.time()
        async with self.lock:
            self.runtime_clients = add_end_time(
                self.runtime_clients, 
                request.client_ID, 
                end_time
            )
            self.clientes_respondidos += 1
            
            # --- Unlearning: ajusta clientes_esperados dinamicamente --- [UNLEARNING]
            clientes_esperados_atual = self.clientes_esperados
            if self.unlearning_enabled and self.round >= self.unlearning_round:
                clientes_esperados_atual = number_of_clients - 1
                
            logger.info(f"O cliente {request.client_ID} finalizou round. Clientes respondidos: {self.clientes_respondidos}/{clientes_esperados_atual}")

            if self.clientes_respondidos == clientes_esperados_atual:
                logger.info("Todos os clientes finalizaram.")

                for i in self.runtime_clients:
                    logger.debug(f"Client ID: {i[0]} → tempo de execução: {utils.format_time(i[1][1] - i[1][0])}")

                logger.info(f"Tempo de Execução Médio: {utils.format_time(average_runtime(self.runtime_clients))}")

                self.metrics = {
                    "trees_by_client": self.get_number_of_trees_per_client(),
                    "aggregation_time": self.aggregation_time,
                    "avg_execution_time": average_runtime(self.runtime_clients)
                }


                if self.result_file_path.exists():
                    with open(self.result_file_path, "r", encoding="utf-8") as file:
                        data = json.load(file)
                else:
                    data = {}

                data[self.round] = self.metrics

                with open(self.result_file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)

                await self._reset_server_async()

                logger.warning(f"Round {self.round} finalizado")
                self.round += 1

                if self.round >= number_of_rounds:
                    logger.warning(f"Encerrando treinamento em 5 segundos...")
                    self.shutdown_event.set()
                    return fedT_pb2.OK(ok=1)
                else: 
                    self.aggregation_realised = 0
                    self.aggregation_done = asyncio.Event()
                    self._supervisor_started = False

                    logger.warning(f"Round {self.round} iniciado")

        return fedT_pb2.OK(ok=1)

    async def _reset_server_async(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.executor, self._reset_server_sync)

    def _reset_server_sync(self):
        logger.warning("Resetando estado do servidor...")
        
        del self.model, self.global_trees, self.strategy
        gc.collect()

        self.model = RandomForestClassifier(  # [CLASSIF]
            n_estimators=self.get_number_of_trees_per_client(),
            max_depth=max_depth,  # [CLASSIF]
            min_samples_leaf=min_samples_leaf,  # [CLASSIF]
            min_samples_split=min_samples_split,  # [CLASSIF]
            max_features=max_features,  # [CLASSIF]
            ccp_alpha=ccp_alpha,  # [CLASSIF]
        )
        # [UNLEARNING] Passa blocked_clients para excluir dados desses clientes
        data_train, label_train = utils.load_dataset_for_server(excluded_clients=self.blocked_clients if self.blocked_clients else None)
        utils.set_initial_params(self.model, data_train, label_train)

        self.global_trees = self.model.estimators_
        self.strategy = FedForest(self.model)

        self.clientes_conectados = []
        self.clientes_respondidos = 0
        self.trees_warehouse = []
        self.aggregation_realised = 0
        self.runtime_clients = []
        self.aggregation_time = 0.0
        
        # --- Unlearning: resetar clientes_esperados dinamicamente --- [UNLEARNING]
        if self.unlearning_enabled and self.round >= self.unlearning_round:
            self.clientes_esperados = number_of_clients - 1
        else:
            self.clientes_esperados = number_of_clients


async def run_server(input_aggregation_strategy=None):
    logger.info("Servidor inicializando...")

    server = grpc_aio.server()
    servicer = FedT(input_aggregation_strategy)

    shutdown_event = asyncio.Event()
    servicer.attach_shutdown_event(shutdown_event)

    fedT_pb2_grpc.add_FedTServicer_to_server(servicer, server)

    server.add_insecure_port(f"{server_config['IP']}:{server_config['port']}")
    
    await server.start()
    logger.info(f"Servidor ativo - {server_config['IP']}:{server_config['port']}")

    await shutdown_event.wait()
    logger.warning("Shutdown event recebido, desligando o servidor...")

    await server.stop(grace=10)
    await server.wait_for_termination()

    servicer.executor.shutdown(wait=True)
    logger.warning("Servidor encerrado.")


if __name__ == "__main__":
    asyncio.run(run_server())