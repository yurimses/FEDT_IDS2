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
    def __init__(self, model: RandomForestClassifier) -> None: # [CLASSIF]
        self.model = model 

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
        data_valid, label_valid = utils.load_server_side_validation_data()  # [CLASSIF]
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
        X_valid, y_valid = utils.load_server_side_validation_data()  # [CLASSIF]
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
        X_valid, y_valid = utils.load_server_side_validation_data()  # [CLASSIF]
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
