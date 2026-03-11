import asyncio
import time
import os
import json
import gc 

import numpy as np
import pandas as pd

import grpc
import grpc.aio as grpc_aio

from fedt.settings import (
    server_ip, server_port, number_of_rounds, 
    client_timeout, client_debug, 
    imported_aggregation_strategy, many_simulations,
    max_depth, min_samples_leaf, min_samples_split, max_features, ccp_alpha, # [CLASSIF]
    print_class_distribution,  # [CLASSIF]  
    dataset_path, label_target, partition_seed,
    dominant_client_id, unlearning_enabled, unlearning_round,  # [UNLEARNING]
    max_classes_beeswarm, max_display_features  # [SHAP]
)
from fedt import utils
from fedt.utils import create_specific_result_folder
from fedt.utils import format_time
from fedt import fedT_pb2
from fedt import fedT_pb2_grpc

from sklearn.ensemble import RandomForestClassifier #[CLASSIF]
from client_utils import HouseClient

import argparse
import logging
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor


executor = ThreadPoolExecutor(max_workers=None)

parse = argparse.ArgumentParser(description="FedT")
parse.add_argument(
    "--client-id",
    required=True,
    type=int,
    help="Client ID"
)
parse.add_argument(
    "--strategy",
    type=str,
    default=imported_aggregation_strategy,
    help="Nome da estratégia (opcional)"
)
args = parse.parse_args()
ID = args.client_id
aggregation_strategy = args.strategy

log_level = logging.DEBUG if client_debug else logging.INFO
logger = utils.setup_logger(
    name=f"Client {ID}",
    log_file=f"fedt_client_{ID}.log",
    level=log_level
)

_CLASS_NAMES_FOR_CONFUSION = None  # [CLASSIF]


def _get_class_names_for_confusion(num_classes):  # [CLASSIF]
    """[CLASSIF] Retorna nomes de classes na mesma ordem da matriz de confusão."""  # [CLASSIF]
    global _CLASS_NAMES_FOR_CONFUSION  # [CLASSIF]

    if _CLASS_NAMES_FOR_CONFUSION is not None:  # [CLASSIF]
        return _CLASS_NAMES_FOR_CONFUSION  # [CLASSIF]

    try:  # [CLASSIF]
        # [CLASSIF] Lê apenas o cabeçalho (barato) para decidir quais colunas carregar
        header = pd.read_csv(dataset_path, nrows=0)  # [CLASSIF]
        label_col = str(label_target)  # [CLASSIF]
        available_cols = set(header.columns)  # [CLASSIF]

        # [CLASSIF] Carrega o mínimo possível: coluna alvo + possíveis colunas descritivas
        usecols = []  # [CLASSIF]
        if label_col in available_cols:  # [CLASSIF]
            usecols.append(label_col)  # [CLASSIF]

        # [CLASSIF] Para rótulo binário, tenta colunas mais descritivas (se existirem)
        if label_col == "Attack_label":  # [CLASSIF]
            for desc_col in ("Attack_type_6", "Attack_type"):  # [CLASSIF]
                if desc_col in available_cols:  # [CLASSIF]
                    usecols.append(desc_col)  # [CLASSIF]

        # [CLASSIF] Se nenhuma coluna relevante existe, não há como inferir nomes
        if not usecols:  # [CLASSIF]
            _CLASS_NAMES_FOR_CONFUSION = None  # [CLASSIF]
            return None  # [CLASSIF]

        df_small = pd.read_csv(dataset_path, usecols=list(dict.fromkeys(usecols)))  # [CLASSIF]

        candidates = []  # [CLASSIF]
        if label_col == "Attack_label":  # [CLASSIF]
            for desc_col in ("Attack_type_6", "Attack_type"):  # [CLASSIF]
                if desc_col in df_small.columns and df_small[desc_col].nunique(dropna=True) == num_classes:  # [CLASSIF]
                    candidates.append(desc_col)  # [CLASSIF]

        if label_col in df_small.columns and df_small[label_col].nunique(dropna=True) == num_classes:  # [CLASSIF]
            candidates.append(label_col)  # [CLASSIF]

        # [CLASSIF] OBS: removido o scan de "todas as colunas" para evitar carregar o CSV inteiro na memória.

        for col in candidates:  # [CLASSIF]
            y_series = df_small[col].astype(str)  # [CLASSIF]
            uniques = sorted(y_series.dropna().unique())  # [CLASSIF]
            if len(uniques) == num_classes:  # [CLASSIF]
                _CLASS_NAMES_FOR_CONFUSION = list(uniques)  # [CLASSIF]
                return _CLASS_NAMES_FOR_CONFUSION  # [CLASSIF]

        _CLASS_NAMES_FOR_CONFUSION = None  # [CLASSIF]
        return None  # [CLASSIF]
    except Exception as e:  # [CLASSIF]
        logger.warning(f"[CLASSIF] Falha ao obter nomes de classe para matriz de confusão: {e}")  # [CLASSIF]
        return None  # [CLASSIF]


def send_stream_trees(serialise_trees:bytes, client_ID:int):
    async def _gen():
        for tree in serialise_trees:
            msg = fedT_pb2.Forest_CLient()
            msg.client_ID = client_ID
            msg.serialised_tree = tree
            yield msg
            await asyncio.sleep(0)
    return _gen()

async def run():
    # Extrai dataset_name e partition_type das configurações
    from pathlib import Path
    from fedt.settings import partition_type as pt_setting
    
    dataset_name = Path(dataset_path).stem
    partition_type_str = pt_setting.lower() if isinstance(pt_setting, str) else "iid"
    
    base_file_name = f"{aggregation_strategy}_client-id-{ID}"
    results_folder = utils.create_specific_result_folder_with_dataset(
        aggregation_strategy, 
        dataset_name, 
        partition_type_str, 
        f"client-id-{ID}"
    )
    existing_files = [
        file for file in os.listdir(results_folder)
        if file.startswith(base_file_name) and file.endswith(".json")
    ]
    next_file_index = len(existing_files) + 1
    result_file_name = f"{base_file_name}_{next_file_index}.json"
    result_file_path = (results_folder / result_file_name).resolve()

    logger.warning(f"Result path: {result_file_path}")

    async with grpc_aio.insecure_channel(f"{server_ip}:{server_port}") as channel:
        stub = fedT_pb2_grpc.FedTStub(channel)

        dataset = utils.load_house_client(ID)  # [CLASSIF]

         # [CLASSIF] Mostrar desproporção de classes por cliente (train/test),
        # controlado pela flag print_class_distribution no config.toml
        if print_class_distribution:  # [CLASSIF]
            X_train, y_train, X_test, y_test = dataset  # [CLASSIF]

            # [CLASSIF] Usa pandas para obter estatísticas descritivas das classes
            y_train_series = pd.Series(y_train, name="y_train")  # [CLASSIF]
            y_test_series = pd.Series(y_test, name="y_test")  # [CLASSIF]

            logger.info(
                f"[CLASSIF] Cliente {ID} – y_train.describe():\n{y_train_series.describe()}"
            )  # [CLASSIF]
            logger.info(
                f"[CLASSIF] Cliente {ID} – y_test.describe():\n{y_test_series.describe()}"
            )  # [CLASSIF]

            # [CLASSIF] Distribuição básica (código da classe, contagem e proporção)
            classes_train, counts_train = np.unique(y_train, return_counts=True)  # [CLASSIF]
            classes_test, counts_test = np.unique(y_test, return_counts=True)  # [CLASSIF]

            total_train = counts_train.sum()  # [CLASSIF]
            total_test = counts_test.sum()  # [CLASSIF]

            proportions_train = (counts_train / total_train).round(4)  # [CLASSIF]
            proportions_test = (counts_test / total_test).round(4)  # [CLASSIF]

            df_train = pd.DataFrame(  # [CLASSIF]
                {
                    "classe_codigo": classes_train,
                    "contagem": counts_train,
                    "proporcao": proportions_train,
                }
            ).set_index("classe_codigo")  # [CLASSIF]

            df_test = pd.DataFrame(  # [CLASSIF]
                {
                    "classe_codigo": classes_test,
                    "contagem": counts_test,
                    "proporcao": proportions_test,
                }
            ).set_index("classe_codigo")  # [CLASSIF]

            # [CLASSIF] Tenta mapear o código da classe para o nome original (por ex., Attack_type_6)
            try:  # [CLASSIF]
                df_full = pd.read_csv(dataset_path, usecols=[label_target])  # [CLASSIF]
                if label_target in df_full.columns:  # [CLASSIF]
                    all_classes = sorted(  # [CLASSIF]
                        map(str, df_full[label_target].dropna().unique())
                    )  # [CLASSIF]
                    code_to_name = {code: name for code, name in enumerate(all_classes)}  # [CLASSIF]

                    df_train["classe_nome"] = df_train.index.map(code_to_name).astype("string")  # [CLASSIF]
                    df_test["classe_nome"] = df_test.index.map(code_to_name).astype("string")  # [CLASSIF]
            except Exception as e:  # [CLASSIF]
                logger.warning(f"[CLASSIF] Falha ao mapear códigos de classe para nomes: {e}")  # [CLASSIF]

            logger.info(
                f"[CLASSIF] Cliente {ID} – distribuição y_train (classe → contagem/proporção[/nome]):\n{df_train}"
            )  # [CLASSIF]
            logger.info(
                f"[CLASSIF] Cliente {ID} – distribuição y_test (classe → contagem/proporção[/nome]):\n{df_test}"
            )  # [CLASSIF]

        for round_idx in range(number_of_rounds):
            # --- Unlearning: cliente dominante para de se comunicar após unlearning_round --- [UNLEARNING]
            if unlearning_enabled and ID == dominant_client_id and round_idx >= unlearning_round:
                logger.warning(f"[UNLEARNING] Cliente dominante (ID={ID}) encerrado após round {unlearning_round-1}. Não participa mais do treinamento.")
                break
                
            round_start_time = time.time()
            logger.warning(f"Round: {round_idx}")

            request_settings = fedT_pb2.Request_Server(client_ID=ID)
            server_reply_settings = await stub.get_server_settings(request_settings)
            trees_by_client = server_reply_settings.trees_by_client
            server_round = getattr(server_reply_settings, "current_round", None)

            logger.debug(f"Trees by client: {trees_by_client}.")

            wait_start = time.time()
            while server_round is not None and server_round < round_idx:
                logger.info(f"Servidor no round {server_round}, esperando atingir round {round_idx}...")
                await asyncio.sleep(5)
                server_reply_settings = await stub.get_server_settings(request_settings)
                server_round = server_reply_settings.current_round
                trees_by_client = server_reply_settings.trees_by_client
                if time.time() - wait_start > client_timeout:
                    raise RuntimeError(f"[Client {ID}] Timeout esperando servidor avançar do round {server_round} para {round_idx}")

            request_model = fedT_pb2.Request_Server(client_ID=ID)
            server_trees_serialised = []
            async for server_reply in stub.get_server_model(request_model):
                server_trees_serialised.append(server_reply.serialised_tree)

            first_server_serialise_trees_size = utils.get_size_of_many_serialised_models(server_trees_serialised)
            logger.debug(f"Early Server Model in MB: {first_server_serialise_trees_size/(1024**2)}")

            loop = asyncio.get_running_loop()
            server_trees_deserialise = await loop.run_in_executor(
                executor,
                utils.deserialise_several_trees,
                server_trees_serialised
            )
            del server_trees_serialised
            gc.collect()

            server_model = RandomForestClassifier( #[CLASSIF]
                n_estimators=trees_by_client,
                max_depth=max_depth,  # [CLASSIF]
                min_samples_leaf=min_samples_leaf,  # [CLASSIF]
                min_samples_split=min_samples_split,  # [CLASSIF]
                max_features=max_features,  # [CLASSIF]
                ccp_alpha=ccp_alpha,  # [CLASSIF]
                class_weight='balanced',  # [CLASSIF] Balanceia automaticamente classes desbalanceadas
                #warm_start=True  # [CLASSIF] Mantém árvores existentes ao chamar fit() 
            )
            server_model.fit(dataset[0], dataset[1])
            server_model.estimators_ = server_trees_deserialise

            fit_start_time = time.time()
            client = HouseClient(trees_by_client, dataset, ID)
            fit_time = time.time() - fit_start_time

            (f1_value, mcc_value, accuracy_value, precision_value, recall_value, cm_norm, best_trees) = client.evaluate(server_model)  #[CLASSIF]
            logger.info(f"\nModelo Inicial:\nAcurácia: {accuracy_value:.3f}\nRecall: {recall_value:.3f}\nPrecision: {precision_value:.3f}\nF1 Score: {f1_value:.3f}\nMCC: {mcc_value:.3f}")  #[CLASSIF]
            logger.info(f"Matriz de confusão normalizada (cliente {ID}):\n{cm_norm}")  #[CLASSIF]


            serialise_trees = await loop.run_in_executor(
                executor,
                utils.serialise_several_trees,
                client.trees
            )
            client_serialise_trees_size = utils.get_size_of_many_serialised_models(serialise_trees)
            logger.debug(f"Local Model in MB: {client_serialise_trees_size/(1024**2)}")

            server_trees_serialised = []
            async for reply in stub.aggregate_trees(send_stream_trees(serialise_trees, ID)):
                server_trees_serialised.append(reply.serialised_tree)

            del serialise_trees
            gc.collect()

            logger.info("Modelo global recebido")

            request_end = fedT_pb2.Request_Server(client_ID=ID)
            await stub.end_of_transmission(request_end)

            server_trees_deserialised = await loop.run_in_executor(
                executor,
                utils.deserialise_several_trees,
                server_trees_serialised
            )
            server_model.estimators_ = server_trees_deserialised

            final_server_serialise_trees_size = utils.get_size_of_many_serialised_models(server_trees_serialised)
            logger.debug(f"Final Server Model in MB: {final_server_serialise_trees_size/(1024**2)}")


            evaluate_start_time = time.time()
            (f1_value, mcc_value, accuracy_value, precision_value, recall_value, cm_norm, best_trees) = await loop.run_in_executor( #[CLASSIF]
                executor,
                client.evaluate,
                server_model
            )
            evaluate_time = time.time() - evaluate_start_time
            logger.info(f"\nModelo Inicial:\nAcurácia: {accuracy_value:.3f}\nRecall: {recall_value:.3f}\nPrecision: {precision_value:.3f}\nF1 Score: {f1_value:.3f}\nMCC: {mcc_value:.3f}")  #[CLASSIF]
            logger.info(f"Matriz de confusão normalizada (cliente {ID}):\n{cm_norm}")  #[CLASSIF]

            round_end_time = time.time()
            round_time = round_end_time - round_start_time

            start_inference_time = time.time()
            await loop.run_in_executor(
                executor,
                client.evaluate_inference_time,
                100
            )
            inference_time = time.time() - start_inference_time
            logger.debug(f"\nDuração do Round: {format_time(round_time)}\nTempo de treinamento: {format_time(fit_time)}\nTempo de avaliação: {format_time(evaluate_time)}\nTempo de inferência: {format_time(inference_time)}")

            # [SHAP] Calcular SHAP values do cliente no último round (antes de deletar variáveis)
            if round_idx == number_of_rounds - 1:
                logger.info("[SHAP] Último round detectado. Calculando SHAP values para o cliente...")
                
                try:
                    # Carregar nomes das features
                    feature_names = utils.get_feature_names_from_dataset()
                    if feature_names is None:
                        logger.warning("[SHAP] Não foi possível obter nomes das features")
                    else:
                        # Calcular SHAP values para o modelo local
                        shap_values_local, _, X_sample_local = utils.calculate_shap_values(
                            client.local_model, 
                            client.X_test, 
                            max_samples=100,
                            seed=int(partition_seed) + int(ID),
                        )
                        
                        if shap_values_local is not None:
                            # Criar pasta para salvar resultados SHAP
                            shap_folder = result_file_path.parent / "shap"
                            shap_folder.mkdir(parents=True, exist_ok=True)
                            
                            # Salvar bar plot (agregado para multi-classe)
                            summary_bar_path = shap_folder / f"client_{ID}_shap_summary_bar.png"
                            utils.save_shap_summary(shap_values_local, X_sample_local, feature_names, 
                                                   summary_bar_path, plot_type="bar", 
                                                   max_display=max_display_features)
                            
                            # Salvar beeswarm plots (limite configurável de classes)
                            if isinstance(shap_values_local, list):
                                # Multi-classe: salvar um por classe (limite configurável para economizar memória)
                                total_classes = len(shap_values_local)
                                num_classes = total_classes if max_classes_beeswarm == 0 else min(total_classes, max_classes_beeswarm)
                                for class_idx in range(num_classes):
                                    summary_beeswarm_path = shap_folder / f"client_{ID}_shap_summary_beeswarm_class_{class_idx}.png"
                                    utils.save_shap_summary(shap_values_local, X_sample_local, feature_names, 
                                                           summary_beeswarm_path, plot_type="beeswarm", 
                                                           class_idx=class_idx, max_display=max_display_features)
                            else:
                                # Binário: um único beeswarm
                                summary_beeswarm_path = shap_folder / f"client_{ID}_shap_summary_beeswarm.png"
                                utils.save_shap_summary(shap_values_local, X_sample_local, feature_names, 
                                                       summary_beeswarm_path, plot_type="beeswarm", 
                                                       max_display=max_display_features)
                            
                            # Salvar SHAP values em JSON
                            shap_json_path = shap_folder / f"client_{ID}_shap_values.json"
                            utils.save_shap_values_json(shap_values_local, feature_names, shap_json_path)
                            
                            logger.info("[SHAP] SHAP values do cliente calculados e salvos com sucesso!")
                        else:
                            logger.warning("[SHAP] Falha ao calcular SHAP values do cliente")
                except Exception as e:
                    logger.error(f"[SHAP] Erro ao calcular SHAP values do cliente: {e}")

            del server_model, client, server_trees_serialised, server_trees_deserialised

            metrics = {
                "trees_by_client": trees_by_client,
                "first_server_serialise_trees_size": first_server_serialise_trees_size,
                "fit_time": fit_time,
                "client_serialise_trees_size": client_serialise_trees_size,
                "final_server_serialise_trees_size": final_server_serialise_trees_size,
                "f1_score": f1_value,  #[CLASSIF]
                "mcc": mcc_value,  #[CLASSIF]
                "accuracy": accuracy_value,  # [CLASSIF]
                "precision": precision_value,  # [CLASSIF]
                "recall": recall_value,  # [CLASSIF]
                "confusion_matrix": cm_norm.tolist(),  # [CLASSIF]
                "confusion_matrix_labels": _get_class_names_for_confusion(len(cm_norm)),  # [CLASSIF]
                "round_time": round_time,
                "round_start_time": round_start_time,
                "round_end_time": round_end_time,
                "evaluate_time": evaluate_time,
                "inference_time": inference_time
            }
            if result_file_path.exists():
                with open(result_file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
            else:
                data = {}

            data[server_round] = metrics
            with open(result_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            gc.collect()
            await asyncio.sleep(15)


if __name__ == "__main__":
    asyncio.run(run())
    executor.shutdown(wait=True)