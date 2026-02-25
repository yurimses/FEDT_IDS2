from sklearn.ensemble import RandomForestClassifier # [CLASSIF]
from sklearn.tree import DecisionTreeClassifier # [CLASSIF]
from sklearn.metrics import f1_score, matthews_corrcoef  # [CLASSIF]

import numpy as np

import random

import warnings
from scipy.stats import ConstantInputWarning

warnings.filterwarnings("ignore", category=ConstantInputWarning)

from fedt import utils
from fedt.settings import ALL_LABELS  # [CLASSIF]

# [CLASSIF] ALL_LABELS é importado de settings.py e inferido automaticamente do dataset
# configurado em config.toml (dataset_path + label_target)

class FedForest():
    def __init__(self, model: RandomForestClassifier, excluded_clients=None) -> None: # [CLASSIF]
        self.model = model
        self.excluded_clients = excluded_clients  # [UNLEARNING] 

    def _predict_forest_majority(self, forest: list[DecisionTreeClassifier], X):  # [CLASSIF]
        """
        [CLASSIF] Faz predição por votação majoritária a partir de uma lista de árvores,
        sem depender de RandomForestClassifier.predict(), que pressupõe classes_ idênticas.  # [CLASSIF]
        """
        if len(forest) == 0:  # [CLASSIF]
            raise ValueError("Floresta vazia para predição.")  # [CLASSIF]

        # [CLASSIF] all_preds: (#árvores, #amostras)
        all_preds = np.array([tree.predict(X) for tree in forest])  # [CLASSIF]
        n_samples = all_preds.shape[1]  # [CLASSIF]
        majority = np.empty(n_samples, dtype=all_preds.dtype)  # [CLASSIF]

        for j in range(n_samples):  # [CLASSIF]
            values, counts = np.unique(all_preds[:, j], return_counts=True)  # [CLASSIF]
            majority[j] = values[np.argmax(counts)]  # [CLASSIF]

        return majority  # [CLASSIF]

    def aggregate_fit_best_forest_strategy(self, best_forests: list[list[DecisionTreeClassifier]]):  # [CLASSIF]
        # [CLASSIF] Versão de classificação: escolhe a floresta com maior F1-Score macro em um conjunto de validação.
        data_valid, label_valid = utils.load_server_side_validation_data(excluded_clients=self.excluded_clients)  # [CLASSIF]
        best_f1 = float("-inf")  # [CLASSIF]
        best_forest = None  # [CLASSIF]

        for forest in best_forests:  # [CLASSIF]
            if not forest:  # [CLASSIF]
                continue  # [CLASSIF]

            y_pred = self._predict_forest_majority(forest, data_valid)  # [CLASSIF]
            f1 = f1_score(label_valid, y_pred, labels=ALL_LABELS, average="macro", zero_division=0)  # [CLASSIF]
            if f1 > best_f1:  # [CLASSIF]
                best_f1 = f1  # [CLASSIF]
                best_forest = forest  # [CLASSIF]

        if best_forest is not None:  # [CLASSIF]
            utils.set_model_params(self.model, best_forest)  # [CLASSIF]
            return best_forest  # [CLASSIF]
        else:
            # [CLASSIF] Fallback: se nada melhor for encontrado, mantém a floresta atual.
            return self.model.estimators_  # [CLASSIF]
    
    def aggregate_fit_best_trees_strategy(self, best_forests: list[list[DecisionTreeClassifier]]): # [CLASSIF]
        # [CLASSIF] Versão de classificação: ordena as árvores por F1-Score macro em dados de validação.
        X_valid, y_valid = utils.load_server_side_validation_data(excluded_clients=self.excluded_clients)  # [CLASSIF]
        best_trees = []  # [CLASSIF]
        best_trees_ratio = int(len(best_forests[0]) * 0.5)  # numero de melhores arvores por floresta  # [CLASSIF]

        print(f'Numero de melhores arvores por floresta é: {best_trees_ratio}')  # [CLASSIF]

        def _f1_for_tree(tree):  # [CLASSIF]
            y_pred = tree.predict(X_valid)  # [CLASSIF]
            return f1_score(y_valid, y_pred, labels=ALL_LABELS, average="macro", zero_division=0)  # [CLASSIF]

        for forest in best_forests:  # [CLASSIF]
            forest_trees = list(forest)  # [CLASSIF]
            trees_sorted = sorted(  # [CLASSIF]
                forest_trees,
                key=_f1_for_tree,
                reverse=True,
            )  # [CLASSIF]
            best_trees.extend(trees_sorted[:best_trees_ratio])  # [CLASSIF]
            
        return best_trees  # [CLASSIF]

    def aggregate_fit_best_trees_threshold_strategy(self, best_forests: list[list[DecisionTreeClassifier]], threshold: float):  # [CLASSIF]
        """
        Essa estratégia avalia as árvores de cada floresta com métricas de classificação.  # [CLASSIF]
        Seleciona apenas as árvores cujo MCC é maior ou igual ao threshold.  # [CLASSIF]
        """  # [CLASSIF]
        X_valid, y_valid = utils.load_server_side_validation_data(excluded_clients=self.excluded_clients)  # [CLASSIF]
        best_trees = []  # [CLASSIF]

        for forest in best_forests:  # [CLASSIF]
            forest_trees = list(forest)  # [CLASSIF]
            scored_trees = []  # [CLASSIF]

            for tree in forest_trees:  # [CLASSIF]
                y_pred = tree.predict(X_valid)  # [CLASSIF]
                mcc = matthews_corrcoef(y_valid, y_pred)  # [CLASSIF]
                scored_trees.append((tree, mcc))  # [CLASSIF]

            scored_trees.sort(key=lambda item: item[1], reverse=True)  # [CLASSIF]

            selected_trees = [tree for (tree, mcc) in scored_trees if mcc >= threshold]  # [CLASSIF]

            # [CLASSIF] Fallback: se nenhuma árvore atingir o threshold, usa a melhor por MCC
            if not selected_trees and scored_trees:  # [CLASSIF]
                best_tree, best_mcc = scored_trees[0]  # [CLASSIF]
                selected_trees = [best_tree]  # [CLASSIF]

            print(
                f"\n######################\n"
                f"Número de Florestas: {len(best_forests)}\n"
                f"Número de Árvores por Floresta: {len(best_forests[0])}\n"
                f"Threshold MCC: {threshold}\n"
                f"######################\n"
            )  # [CLASSIF]

            best_trees.extend(selected_trees)  # [CLASSIF]

        return best_trees  # [CLASSIF]

    def aggregate_fit_best_trees_with_class_coverage_strategy(
        self, 
        best_forests: list[list[DecisionTreeClassifier]], 
        trees_per_class: int = 3,
        total_trees_ratio: float = 0.5
    ):  # [CLASSIF]
        """
        [CLASSIF] Estratégia que garante cobertura por classe na seleção de árvores.
        
        Para cada classe, seleciona as top-N árvores com melhor F1 para aquela classe.
        Depois completa o ensemble com as melhores árvores por macro-F1 até atingir o 
        número total desejado.
        
        Args:
            best_forests: Lista de florestas (cada floresta é uma lista de árvores)
            trees_per_class: Número de árvores especialistas a selecionar por classe (default: 3)
            total_trees_ratio: Proporção do total de árvores a manter (default: 0.5)
        
        Returns:
            Lista de árvores selecionadas com cobertura garantida por classe
        """
        X_valid, y_valid = utils.load_server_side_validation_data(excluded_clients=self.excluded_clients)  # [CLASSIF]
        
        # Agregar todas as árvores em uma única lista
        all_trees = []  # [CLASSIF]
        for forest in best_forests:  # [CLASSIF]
            all_trees.extend(forest)  # [CLASSIF]
        
        if not all_trees:  # [CLASSIF]
            return []  # [CLASSIF]
        
        # Calcular F1 por classe para cada árvore
        trees_f1_per_class = []  # [CLASSIF]
        trees_f1_macro = []  # [CLASSIF]
        
        print(f"\n[COBERTURA POR CLASSE] Avaliando {len(all_trees)} árvores...")  # [CLASSIF]
        
        for tree in all_trees:  # [CLASSIF]
            y_pred = tree.predict(X_valid)  # [CLASSIF]
            # F1 por classe (vetor com um score para cada classe)
            f1_per_class = f1_score(
                y_valid, y_pred, 
                labels=ALL_LABELS, 
                average=None,  # Retorna array com F1 para cada classe
                zero_division=0
            )  # [CLASSIF]
            # F1 macro (média das classes)
            f1_macro = f1_score(
                y_valid, y_pred,
                labels=ALL_LABELS,
                average="macro",
                zero_division=0
            )  # [CLASSIF]
            trees_f1_per_class.append(f1_per_class)  # [CLASSIF]
            trees_f1_macro.append(f1_macro)  # [CLASSIF]
        
        trees_f1_per_class = np.array(trees_f1_per_class)  # shape: (n_trees, n_classes)
        trees_f1_macro = np.array(trees_f1_macro)  # shape: (n_trees,)
        
        # Conjunto para rastrear índices de árvores já selecionadas
        selected_indices = set()  # [CLASSIF]
        
        # Passo 1: Selecionar top-N árvores para cada classe
        n_classes = len(ALL_LABELS)  # [CLASSIF]
        print(f"[COBERTURA POR CLASSE] Selecionando {trees_per_class} árvores para cada uma das {n_classes} classes...")  # [CLASSIF]
        
        for class_idx in range(n_classes):  # [CLASSIF]
            # Ordenar árvores por F1 nesta classe específica
            class_f1_scores = trees_f1_per_class[:, class_idx]  # [CLASSIF]
            sorted_indices = np.argsort(class_f1_scores)[::-1]  # descendente
            
            # Selecionar top-N para esta classe
            count = 0  # [CLASSIF]
            for idx in sorted_indices:  # [CLASSIF]
                if count >= trees_per_class:  # [CLASSIF]
                    break  # [CLASSIF]
                selected_indices.add(int(idx))  # [CLASSIF]
                count += 1  # [CLASSIF]
            
            best_f1_for_class = class_f1_scores[sorted_indices[0]] if len(sorted_indices) > 0 else 0  # [CLASSIF]
            print(f"  Classe {ALL_LABELS[class_idx]}: melhor F1={best_f1_for_class:.4f}")  # [CLASSIF]
        
        print(f"[COBERTURA POR CLASSE] {len(selected_indices)} árvores únicas selecionadas como especialistas")  # [CLASSIF]
        
        # Passo 2: Completar com as melhores por macro-F1
        target_total = int(len(all_trees) * total_trees_ratio)  # [CLASSIF]
        print(f"[COBERTURA POR CLASSE] Completando até {target_total} árvores com melhores macro-F1...")  # [CLASSIF]
        
        # Ordenar todas as árvores por macro-F1
        sorted_by_macro = np.argsort(trees_f1_macro)[::-1]  # descendente
        
        for idx in sorted_by_macro:  # [CLASSIF]
            if len(selected_indices) >= target_total:  # [CLASSIF]
                break  # [CLASSIF]
            selected_indices.add(int(idx))  # [CLASSIF]
        
        # Criar lista final de árvores selecionadas
        selected_trees = [all_trees[idx] for idx in sorted(selected_indices)]  # [CLASSIF]
        
        print(f"[COBERTURA POR CLASSE] Total final: {len(selected_trees)} árvores selecionadas")  # [CLASSIF]
        print(f"[COBERTURA POR CLASSE] Macro-F1 médio das árvores selecionadas: {np.mean([trees_f1_macro[idx] for idx in selected_indices]):.4f}\n")  # [CLASSIF]
        
        return selected_trees  # [CLASSIF]

    def aggregate_fit_random_trees_strategy(self, best_forests):
        best_trees = []
        best_trees_ratio = int(len(best_forests[0]) * 0.5)

        for forest in best_forests:
            forest_trees = np.array(forest)
            num_trees = len(forest_trees)

            if best_trees_ratio >= num_trees:
                best_trees.extend(forest_trees)
            else:
                selected_indices = np.random.choice(num_trees, best_trees_ratio, replace=False)
                selected_trees = forest_trees[selected_indices]
                best_trees.extend(selected_trees)

        return best_trees
