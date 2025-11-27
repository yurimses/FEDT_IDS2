"""
[CLASSIF] Grid Search para RandomForest de classificação com poda.

"""

from fedt.utils import load_dataset  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import GridSearchCV, StratifiedKFold  


def run_grid_search():  # [CLASSIF]
    
    X, y = load_dataset()  # [CLASSIF]

    rf = RandomForestClassifier(
        n_estimators=100,  
        n_jobs=-1,
        random_state=42,
    )

    param_grid = {  # [CLASSIF]
        "n_estimators": [50, 100, 200],        # número de árvores na floresta          
        "max_depth": [None, 4, 6, 8],          # controla profundidade (pre-pruning)    
        "min_samples_split": [2, 4, 8],        # mínimo de amostras para split          
        "min_samples_leaf": [1, 2, 4],         # mínimo de amostras por folha           
        "max_features": ["sqrt", "log2"],      # fração de atributos a cada split       
        "ccp_alpha": [0.0, 1e-4, 1e-3, 5e-3],  # poda por custo-complexidade (post)     
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  

    scoring = {  # [CLASSIF]
        "accuracy": "accuracy",                # acurácia global                        
        "precision_macro": "precision_macro",  # precisão macro (classes equilibradas)  
        "recall_macro": "recall_macro",        # recall macro                           
        "f1_macro": "f1_macro",                # F1 macro (métrica principal)           
    }

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,          # usa o dicionário de métricas                       
        refit="f1_macro",         # escolhe e refaz o fit final usando F1 macro        
        n_jobs=-1,
        verbose=2,
        return_train_score=False, # ajuste para focar nas métricas de validação        
    )

    grid.fit(X, y)  # [CLASSIF]

    print("\n===== RESULTADOS DO GRID SEARCH =====")  
    print("Melhores parâmetros (refit = F1 macro):")  
    print(grid.best_params_)  
    print("\nMelhor F1 macro (média nos folds de validação):", grid.best_score_)  

    best_index = grid.best_index_        # índice da melhor combinação na tabela       
    cv_results = grid.cv_results_        # dicionário com todos os resultados          

    print("\nMétricas de validação (médias nos folds) para os melhores hiperparâmetros:")  
    for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:  
        mean_key = f"mean_test_{metric}"  
        std_key = f"std_test_{metric}"    
        if mean_key in cv_results:
            mean_val = cv_results[mean_key][best_index]  
            std_val = cv_results[std_key][best_index]    
            print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")  




if __name__ == "__main__":  
    run_grid_search()  
