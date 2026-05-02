import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Importiamo tutti i metodi di feature selection
from src.feature_selection.embedded.lasso_feature_selection import LassoFeatureSelector
from src.feature_selection.feature_ranking.pairwise_correlation_ranking import PairwiseCorrelationRanker
from src.feature_selection.feature_ranking.relief_ranking import ReliefRanker
from src.feature_selection.feature_ranking.uncertainty_information_gain_ranking import InformationGainRanker
from src.feature_selection.subset_selection.sfs import SequentialForwardSelector
from src.feature_selection.subset_selection.sbs_subset_selection import SequentialBackwardSelector
from src.feature_selection.subset_selection.bidirectional_subset_selection import StepwiseBidirectionalSelector
from src.feature_selection.subset_selection.max_min_subset_selection import MaxMinSubsetSelector
from src.feature_selection.subset_selection.best_first import BestFirstSelector
from src.feature_selection.feature_ranking.pca import PCAHandler
from src.preprocessing.data_selection import get_balanced_sample


def evaluate_features_with_knn_cv(X_train, y_train, X_val, y_val, features):
    """Valuta le feature usando un KNN ottimizzato tramite GridSearch."""
    if not features:
        return 0.0, None
    
    valid_features = [f for f in features if f in X_train.columns]
    if not valid_features:
        return 0.0, None
    
    X_tr_sub = X_train[valid_features]
    X_v_sub = X_val[valid_features]
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(weights='distance', n_jobs=-1))
    ])
    
    # Grid search su K per trovare il parametro migliore
    param_grid = {'knn__n_neighbors': [3, 5, 9, 15, 21]}
    
    # Usiamo una cv veloce a 3 fold sul train set
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_micro', n_jobs=-1)
    grid_search.fit(X_tr_sub, y_train)
    
    # Valutiamo sul validation set con il miglior modello trovato
    best_model = grid_search.best_estimator_
    best_k = grid_search.best_params_['knn__n_neighbors']
    
    y_pred = best_model.predict(X_v_sub)
    score = f1_score(y_val, y_pred, average='micro')
    
    return score, best_k

def run_evaluation():
    print("="*80)
    print("BENCHMARK RIGOROSO DI FEATURE SELECTION")
    print("="*80)
    print("Caricamento dataset preprocessato...")
    
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "processed" / "train_features_labels_preprocessed.csv"
    
    if not data_path.exists():
        print(f"File non trovato: {data_path}")
        return
        
    df = pd.read_csv(data_path)
    print(f"Dataset originale caricato: {df.shape}")
    
    # 1. Bilanciamento e Campionamento
    target_col = "damage_grade"
    df_sampled = get_balanced_sample(df, target_col, max_per_class=10000)
    print(f"Dataset bilanciato per la validazione: {df_sampled.shape}")
    print("Distribuzione classi:")
    print(df_sampled[target_col].value_counts())
    
    exclude_cols = ["building_id", "geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
    exclude_cols = [c for c in exclude_cols if c in df_sampled.columns]
    
    X = df_sampled.drop(columns=[target_col] + exclude_cols)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
         X = pd.get_dummies(X, columns=categorical_cols, drop_first=False, dtype=float)
         
    y = df_sampled[target_col].astype(int)
    
    # 2. Train-Validation Split Stratificato
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    df_train = X_train.copy()
    df_train[target_col] = y_train
    
    results = []
    MAX_FEATURES = 15
    # Limite interno per subset selector per evitare tempi computationabili esagerati su 24k righe
    MAX_ROWS_SUBSET = 2000 
    
    methods = [
        ("PCA (Elbow Method)", PCAHandler, {}),
        ("Lasso Embedded", LassoFeatureSelector, {}),
        ("Pairwise Correlation", PairwiseCorrelationRanker, {}),
        ("Relief", ReliefRanker, {}),
        ("Information Gain", InformationGainRanker, {}),
        ("Sequential Forward Selection", SequentialForwardSelector, {'estimator_name': 'knn', 'scoring': 'f1_micro'}),
        ("Sequential Backward Selection", SequentialBackwardSelector, {'estimator_name': 'knn', 'scoring': 'f1_micro'}),
        ("Bidirectional Subset Selection", StepwiseBidirectionalSelector, {'estimator_name': 'knn', 'scoring': 'f1_micro'}),
        ("Max-Min Subset Selection", MaxMinSubsetSelector, {}),
        ("Best First Search", BestFirstSelector, {})
    ]
    
    print("\nInizio valutazione dei metodi (con GridSearchCV per ottimizzare il KNN)...")
    
    for name, MethodClass, kwargs in methods:
        print(f"\n-> Valutazione {name}...")
        start_time = time.time()
        selected_features = []
        best_k = None
        try:
            if name == "Information Gain":
                model = MethodClass(log_base=2)
            else:
                model = MethodClass(**kwargs)
                
            if name == "Lasso Embedded":
                res = model.select(X_train, y_train, alpha=0.002)
                selected_features = res["selected_features"]["feature"].head(MAX_FEATURES).tolist()
                score, best_k = evaluate_features_with_knn_cv(X_train, y_train, X_val, y_val, selected_features)
                
            elif name == "PCA (Elbow Method)":
                model.fit(df_train, exclude_columns=[target_col])
                var_table = model.build_variance_table()
                
                y_var = var_table["explained_variance"].values
                x_var = np.arange(1, len(y_var) + 1)
                
                p1 = np.array([x_var[0], y_var[0]])
                p2 = np.array([x_var[-1], y_var[-1]])
                distances = []
                for i in range(len(x_var)):
                    p3 = np.array([x_var[i], y_var[i]])
                    dist = np.abs(np.cross(p2 - p1, p3 - p1)) / np.linalg.norm(p2 - p1)
                    distances.append(dist)
                
                elbow_k = np.argmax(distances) + 1
                
                X_train_pca = model.transform(df_train).iloc[:, :elbow_k]
                X_val_pca = model.transform(X_val).iloc[:, :elbow_k]
                
                # KNN su PCA components con GridSearch
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('knn', KNeighborsClassifier(weights='distance', n_jobs=-1))
                ])
                param_grid = {'knn__n_neighbors': [3, 5, 9, 15, 21]}
                grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_micro', n_jobs=-1)
                grid_search.fit(X_train_pca, y_train)
                
                y_pred = grid_search.best_estimator_.predict(X_val_pca)
                score = f1_score(y_val, y_pred, average='micro')
                best_k = grid_search.best_params_['knn__n_neighbors']
                
                selected_features = [f"PC{i}" for i in range(1, elbow_k + 1)]
                
            elif "Ranker" in MethodClass.__name__:
                res = model.rank(df_train, label_column=target_col)
                if name == "Pairwise Correlation":
                    selected_features = res["supervised_ranking"]["feature"].head(MAX_FEATURES).tolist()
                elif name == "Relief":
                    selected_features = res["relief_ranking"]["feature"].head(MAX_FEATURES).tolist()
                elif name == "Information Gain":
                    selected_features = res["information_gain_ranking"]["feature"].head(MAX_FEATURES).tolist()
                score, best_k = evaluate_features_with_knn_cv(X_train, y_train, X_val, y_val, selected_features)
                
            else:
                # Subset selectors: per sicurezza usiamo il .to_numpy() dove necessario, o serie per MaxMin
                if name == "Sequential Backward Selection":
                    res = model.select(X_train, y_train.to_numpy(), min_features=MAX_FEATURES, max_rows=MAX_ROWS_SUBSET)
                elif name == "Max-Min Subset Selection":
                    res = model.select(X_train, y_train, max_features=MAX_FEATURES)
                elif name == "Best First Search":
                    res = model.select(X_train, y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train, max_rows=MAX_ROWS_SUBSET)
                else:
                    res = model.select(X_train, y_train.to_numpy(), max_features=MAX_FEATURES, max_rows=MAX_ROWS_SUBSET)
                
                selected_features = res["selected_features"]["selected_feature"].tolist()
                score, best_k = evaluate_features_with_knn_cv(X_train, y_train, X_val, y_val, selected_features)
                
            exec_time = time.time() - start_time
            
            print(f"Completato in {exec_time:.2f}s | F1-Micro: {score:.4f} | K Ottimale: {best_k} | Feature: {len(selected_features)}")
            results.append({
                "Method": name,
                "F1_Micro": score,
                "Best_K": best_k,
                "Time_s": exec_time,
                "N_Features": len(selected_features),
                "Selected_Features": selected_features
            })
            
        except Exception as e:
            print(f"Errore in {name}: {str(e)}")
            import traceback
            traceback.print_exc()

    results_df = pd.DataFrame(results).sort_values("F1_Micro", ascending=False).reset_index(drop=True)
    
    print("\n" + "="*80)
    print("CLASSIFICA FINALE (Benchmark Rigoroso: Bilanciato + GridSearch KNN)")
    print("="*80)
    print(results_df[["Method", "F1_Micro", "Best_K", "Time_s", "N_Features"]].to_string())
    
    print("\nI TOP 3 METODI DEFINITIVI SONO:")
    for i in range(min(3, len(results_df))):
        print(f"{i+1}. {results_df.iloc[i]['Method']} (F1: {results_df.iloc[i]['F1_Micro']:.4f})")
        print(f"   Miglior K: {results_df.iloc[i]['Best_K']} | Feature Selezionate: {results_df.iloc[i]['Selected_Features'][:5]}...")
        
    output_path = project_root / "experiments" / "feature_selection_benchmark_rigorous.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nRisultati completi salvati in {output_path}")

if __name__ == "__main__":
    run_evaluation()
