from fedt.settings import results_folder

import numpy as np
from sklearn.ensemble import RandomForestClassifier #[CLASSIF]


from sklearn.metrics import f1_score, matthews_corrcoef  #[CLASSIF]
from scipy.stats import pearsonr

from fedt import utils

import warnings
from scipy.stats import ConstantInputWarning

warnings.filterwarnings("ignore", category=ConstantInputWarning)

class HouseClient():

    def __init__(self, trees_by_client: int, dataset, ID) -> None:
        # Load house data
        self.X_train, self.y_train, self.X_test, self.y_test = dataset

        # Initialize local model and set initial_parameters
        self.local_model = RandomForestClassifier(n_estimators=trees_by_client) #[CLASSIF]
        utils.set_initial_params(self.local_model, self.X_train, self.y_train) 
        self.trees = self.local_model.estimators_
        self.ID = ID

    def get_global_parameters(self, global_model: RandomForestClassifier): #[CLASSIF]
        return utils.get_model_parameters(global_model)

    def evaluate(self, global_model: RandomForestClassifier): #[CLASSIF]
        # [CLASSIF] Avalia modelos local e global usando F1 Score e MCC e retorna o melhor conjunto de árvores
        global_model_trees = self.get_global_parameters(global_model)
        
        y_true = self.y_test  # [CLASSIF]

        # [CLASSIF] Predição do modelo local via votação majoritária nas árvores atuais (self.trees).
        all_local_preds = np.array([tree.predict(self.X_test) for tree in self.trees])  # [CLASSIF]
        n_samples = all_local_preds.shape[1]  # [CLASSIF]
        local_pred = np.empty(n_samples, dtype=all_local_preds.dtype)  # [CLASSIF]
        for j in range(n_samples):  # [CLASSIF]
            values, counts = np.unique(all_local_preds[:, j], return_counts=True)  # [CLASSIF]
            local_pred[j] = values[np.argmax(counts)]  # [CLASSIF]

        # [CLASSIF] Predição do modelo global via votação majoritária das árvores recebidas do servidor.
        if len(global_model_trees) > 0:  # [CLASSIF]
            all_global_preds = np.array([tree.predict(self.X_test) for tree in global_model_trees])  # [CLASSIF]
            n_samples = all_global_preds.shape[1]  # [CLASSIF]
            global_pred = np.empty(n_samples, dtype=all_global_preds.dtype)  # [CLASSIF]
            for j in range(n_samples):  # [CLASSIF]
                values, counts = np.unique(all_global_preds[:, j], return_counts=True)  # [CLASSIF]
                global_pred[j] = values[np.argmax(counts)]  # [CLASSIF]
        else:
            # [CLASSIF] Fallback: se não houver árvores globais, reutiliza a predição local.
            global_pred = local_pred  # [CLASSIF]

        # [CLASSIF] Métricas de classificação
        local_f1 = f1_score(y_true, local_pred, average="macro")  # [CLASSIF]
        global_f1 = f1_score(y_true, global_pred, average="macro")  # [CLASSIF]

        local_mcc = matthews_corrcoef(y_true, local_pred)  # [CLASSIF]
        global_mcc = matthews_corrcoef(y_true, global_pred)  # [CLASSIF]
        
        # [CLASSIF] Seleção: F1 como métrica primária, MCC como critério secundário
        use_global = False  # [CLASSIF]
        if global_f1 > local_f1:  # [CLASSIF]
            use_global = True  # [CLASSIF]
        elif np.isclose(global_f1, local_f1):  # [CLASSIF]
            if global_mcc > local_mcc:  # [CLASSIF]
                use_global = True  # [CLASSIF]

        if use_global:  # [CLASSIF]
            f1_value = global_f1  # [CLASSIF]
            mcc_value = global_mcc  # [CLASSIF]
            self.trees = global_model_trees  # [CLASSIF]
            utils.set_model_params(self.local_model, self.trees)  # [CLASSIF]
        else:  # [CLASSIF]
            f1_value = local_f1  # [CLASSIF]
            mcc_value = local_mcc  # [CLASSIF]

        return f1_value, mcc_value, self.trees  # [CLASSIF]


    def evaluate_inference_time(self, number_of_samples):
        X = self.X_test[-number_of_samples:]
        # [CLASSIF] Usa a floresta atual (self.trees) para simular inferência por votação majoritária.
        if len(self.trees) == 0:  # [CLASSIF]
            return  # [CLASSIF]

        all_preds = np.array([tree.predict(X) for tree in self.trees])  # [CLASSIF]
        n_samples = all_preds.shape[1]  # [CLASSIF]
        majority = np.empty(n_samples, dtype=all_preds.dtype)  # [CLASSIF]
        for j in range(n_samples):  # [CLASSIF]
            values, counts = np.unique(all_preds[:, j], return_counts=True)  # [CLASSIF]
            majority[j] = values[np.argmax(counts)]  # [CLASSIF]