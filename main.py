"""
Script principale per la pipeline ML - Terremoto Nepal 2015

MENU PRINCIPALE:
  FASE 1  - Outlier Detection     : IQR oppure DBSCAN
  FASE 2  - Imputazione           : strategia sui NaN generati dagli outlier
                                    [nota: monum_flag aggiunta DOPO outlier+imputazione]
  FASE 3  - Geo-Level Embedding   : rete neurale per feature geografiche
  FASE 4  - Scelta modello        : KNN / Albero Decisionale / SVM
                                    Multi-Esperto (Random Forest + KNN + Decision Tree)
  FASE 5  - Presentazione risultati (F1-Micro)
"""

import io
import csv
import json
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

# Riconfigura l'encoding di stdout per supportare caratteri Unicode su Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# === Preprocessing ===
from src.preprocessing.clean_ascii import PuliziaASCII, COLONNE_CATEGORICHE
from src.preprocessing.missing_values import MissingValuesHandler
from src.preprocessing.data_cleaning import DataQualityHandler, COLONNE_CONTINUE
from src.preprocessing.validation import DataValidator
from src.preprocessing.imputation_strategies import (
    STRATEGIE_IMPUTAZIONE,
    CODICE_STRATEGIA_DA_NOME_REPORT,
    applica_strategia_imputazione_colonna,
)
from src.preprocessing.outlier_detection.DBSCAN import rileva_outlier_dbscan
from src.preprocessing.geo_features import GeoFeatureEngineer

# === Modelli ===
from src.Modelli.knn import train_knn
from src.Modelli.randomforest import train_randomforest
from src.Modelli.decisiontree import train_decisiontree
from src.Modelli.svm import train_svm

# === Ensemble ===
from src.ensemble.multi_expert_system import MultiExpertSystem, ExpertSpec

# === Feature Selection ===
from src.feature_selection.Hyperparameter_Tuning import (
    esegui_grid_search,
    get_all_configs,
    get_rf_config,
    get_knn_config,
    get_dt_config,
)
from src.feature_selection.feature_ranking.relief_ranking import ReliefRanker
from src.feature_selection.feature_ranking.uncertainty_information_gain_ranking import InformationGainRanker
from src.feature_selection.feature_ranking.pairwise_correlation_ranking import PairwiseCorrelationRanker
from src.feature_selection.feature_ranking.pca import PCAHandler
from src.feature_selection.embedded.lasso_feature_selection import LassoFeatureSelector
from src.feature_selection.subset_selection.sfs import SequentialForwardSelector
from src.feature_selection.subset_selection.sbs_subset_selection import SequentialBackwardSelector
from src.feature_selection.subset_selection.bidirectional_subset_selection import StepwiseBidirectionalSelector
from src.feature_selection.subset_selection.max_min_subset_selection import MaxMinSubsetSelector
from src.feature_selection.subset_selection.best_first import BestFirstSelector

SHOW_DATA_QUALITY_PLOTS = False


# ============================================================
# BANNER
# ============================================================

def _banner(titolo: str) -> None:
    print("\n" + "=" * 80)
    print(titolo)
    print("=" * 80)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def applica_geo_embedding(
    train_values: pd.DataFrame,
    test_values: pd.DataFrame,
    y_train: pd.Series,
    tipo: str = "aggregate",
    smoothing: float = 20.0,
    rare_threshold: int = 10,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applica geo embedding alle feature geografiche usando GeoFeatureEngineer.
    
    Args:
        train_values: dataset di training con colonne geo_level_1/2/3 (senza target)
        test_values: dataset di test con colonne geo_level_1/2/3
        y_train: target di training (damage_grade)
        tipo: tipo di embedding ('aggregate' o 'neural' - attualmente solo aggregate)
        smoothing: parametro di smoothing per l'engineer
        rare_threshold: soglia per considerare un'area rara
        n_splits: fold per OOF sul training
        
    Returns:
        (train_with_geo, test_with_geo): dataset con geo-feature aggiunte
    """
    _banner("GEO-LEVEL EMBEDDING — Applicazione")
    
    print(f"  Tipo: {tipo}")
    print(f"  Smoothing: {smoothing}")
    print(f"  Rare threshold: {rare_threshold}")
    
    # Verifica che le colonne geo siano presenti
    required_geo_cols = ("geo_level_1_id", "geo_level_2_id", "geo_level_3_id")
    for col in required_geo_cols:
        if col not in train_values.columns:
            raise ValueError(f"{col} non trovato in train_values")
    
    geo_engineer = GeoFeatureEngineer(
        geo_columns=required_geo_cols,
        target_col="damage_grade",
        smoothing=smoothing,
        rare_threshold=rare_threshold,
        n_splits=n_splits,
        random_state=42,
        append_original=True,  # Mantiene anche i geo_level_* originali
    )
    
    # OOF sul train per evitare leakage
    print("\n  Costruzione geo-feature su TRAIN set (con OOF anti-leakage)...")
    train_with_geo = geo_engineer.fit_transform_oof(train_values, y_train)
    print(f"  ✓ Features aggiunte a TRAIN: {train_with_geo.shape[1] - train_values.shape[1]} nuove feature")
    
    # Transform su test
    print("  Costruzione geo-feature su TEST set...")
    test_with_geo = geo_engineer.transform(test_values)
    print(f"  ✓ Features aggiunte a TEST: {test_with_geo.shape[1] - test_values.shape[1]} nuove feature")
    
    return train_with_geo, test_with_geo


def menu_feature_selection() -> str:
    """
    Menu per scegliere se usare feature selection.
    
    Ritorna:
        'none' -> nessuna feature selection
        'auto' -> Valutazione dinamica (chiede il modello e valuta il migliore)
    """
    _banner("FEATURE SELECTION — Scelta Metodo")
    print("  Opzioni disponibili:")
    print("  1) Nessuna          — usa tutte le feature")
    print("  2) Automatica       — valuta TUTTI i metodi per il modello scelto")
    print()
    try:
        scelta = input("Seleziona metodo [1-2] (default=1): ").strip()
    except EOFError:
        scelta = ""
    if scelta not in {"1", "2"}:
        scelta = "1"

    mapping = {"1": "none", "2": "auto"}
    metodo = mapping[scelta]
    nomi = {"none": "Nessuna", "auto": "Automatica (TUTTI i metodi)"}
    print(f"\n>> Metodo FS selezionato: {nomi[metodo]}")
    return metodo


def menu_qualita_fs_automatica() -> str:
    """
    Menu per scegliere la qualità della valutazione FS automatica.
    
    Ritorna:
        'fast' -> Valutazione veloce (5k campione, 1 fold, skip subset methods)
        'balanced' -> Equilibrio qualità-velocità (10k campione, 3 fold)
        'thorough' -> Massima qualità (15k campione, 5 fold, TUTTI i metodi)
    """
    _banner("FEATURE SELECTION AUTOMATICA — Modalità di Valutazione")
    print("  Tempo vs Qualità:")
    print("  1) Fast        — veloce (~30 min)      | sample=5k, fold=1, skip subset methods")
    print("  2) Balanced    — medio (~60 min)       | sample=10k, fold=3, TUTTI i metodi")
    print("  3) Thorough    — massima qualità (~120 min) | sample=15k, fold=5, TUTTI i metodi")
    print()
    print()
    try:
        scelta = input("Seleziona modalità [1-3] (default=2): ").strip()
    except EOFError:
        scelta = ""
    if scelta not in {"1", "2", "3"}:
        scelta = "2"
    
    mapping = {"1": "fast", "2": "balanced", "3": "thorough"}
    modalita = mapping[scelta]
    nomi = {"fast": "Fast", "balanced": "Balanced", "thorough": "Thorough"}
    print(f"\n>> Modalità selezionata: {nomi[modalita]}")
    return modalita


def esegui_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    metodo: str = "none",
    top_k: int = 30,
) -> list[str]:
    """
    Esegue feature selection e ritorna la lista di feature selezionate.
    
    Args:
        X_train: feature di training
        y_train: target di training
        metodo: 'none', 'relief', 'info_gain'
        top_k: numero massimo di feature da selezionare
        
    Returns:
        lista di feature selezionate (tutte se metodo='none')
    """
    if metodo == "none":
        print("  Feature Selection: DISABILITATA — usando tutte le feature")
        return X_train.columns.tolist()
    
    _banner(f"FEATURE SELECTION — {metodo.upper()}")
    
    if metodo == "relief":
        print(f"  Metodo: Relief Ranking (top {top_k})")
        ranker = ReliefRanker(random_state=42)
        
        # Prepara DataFrame con y_train
        X_temp = X_train.copy()
        X_temp["__target__"] = y_train.values
        
        result = ranker.rank(X_temp, label_column="__target__")
        
        if "relief_ranking" in result:
            ranking_df = result["relief_ranking"]
        else:
            ranking_df = result.get(list(result.keys())[0])
        
        selected_features = ranking_df["feature"].head(top_k).tolist()
        selected_features = [f for f in selected_features if f in X_train.columns]  # Filtra il target
        print(f"  ✓ Selezionate {len(selected_features)} feature con Relief")
        
    elif metodo == "info_gain":
        print(f"  Metodo: Information Gain Ranking (top {top_k})")
        ranker = InformationGainRanker(log_base=2)
        
        # Creare un dataframe temporaneo con y
        X_temp = X_train.copy()
        X_temp["__target__"] = y_train.values
        
        result = ranker.rank(X_temp, label_column="__target__")
        ranking_df = result["information_gain_ranking"]
        
        selected_features = ranking_df["feature"].head(top_k).tolist()
        selected_features = [f for f in selected_features if f in X_train.columns]  # Filtra il target
        print(f"  ✓ Selezionate {len(selected_features)} feature con Information Gain")
        
    else:
        raise ValueError(f"Metodo FS non riconosciuto: {metodo}")
    
    return selected_features


def _find_optimal_features_from_ranking(ranking_scores: pd.Series, max_features: int = 30) -> int:
    """
    Implementa knee-point detection per trovare il numero ottimale di feature da ranking.
    
    Invece di prendere sempre i top N, trova il punto dove il ranking score 
    diminuisce bruscamente (elbow point).
    
    Args:
        ranking_scores: Series con score per feature (sorted descending)
        max_features: limite massimo di feature
        
    Returns:
        numero ottimale di feature da selezionare (almeno 5, al massimo max_features)
    """
    scores = ranking_scores.values[:max_features]
    
    if len(scores) <= 2:
        return len(scores)
    
    # Calcola le differenze tra punteggi consecutivi (sempre positive)
    diffs = np.abs(np.diff(scores))
    
    # Normalizza le differenze
    max_diff = np.max(diffs)
    if max_diff > 0:
        diffs_norm = diffs / max_diff
    else:
        return len(scores)
    
    # Trova il "gomito" - dove il drop è maggiore della media
    mean_diff = np.mean(diffs_norm)
    threshold = 1.5 * mean_diff
    
    # Seleziona il primo punto dove il drop è significativo (> 1.5x media)
    elbow_indices = np.where(diffs_norm > threshold)[0]
    
    if len(elbow_indices) > 0:
        optimal_k = elbow_indices[0] + 1
    else:
        # Se non trova un elbow chiaro, usa 60% di max_features
        optimal_k = max(1, int(max_features * 0.6))
    
    # Almeno 5 feature, massimo max_features
    return max(5, min(optimal_k, max_features))


def _is_pca_component_selection(selected_features: list[str]) -> bool:
    """
    Riconosce una selezione composta da componenti principali nominate PC1..PCn.
    """
    if not selected_features:
        return False

    return all(
        isinstance(feature, str) and feature.startswith("PC") and feature[2:].isdigit()
        for feature in selected_features
    )


def _should_use_pca_feature_space(
    metodo_fs: str | None,
    selected_features: list[str],
    available_columns: pd.Index,
) -> bool:
    """
    Determina se il training finale deve ricostruire uno spazio PCA.

    La FS automatica può restituire componenti PCA (PC1..PCn) che non esistono
    come colonne fisiche nel dataframe originale. In quel caso serve rifare il
    transform prima dell'addestramento finale.
    """
    if not _is_pca_component_selection(selected_features):
        return False

    pca_method_names = {"PCA (Elbow)", "PCA (Elbow Method)", "pca"}
    if metodo_fs in pca_method_names:
        return True

    return not any(feature in available_columns for feature in selected_features)


def valuta_fs_dinamica_per_modello(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scelta_modello: str,
    modalita: str = "balanced",
    sample_size: int = None,
    max_features: int = None,
    max_rows_subset: int = None,
    subset_cv_folds: int = None,
) -> dict:
    """
    Valuta TUTTI i metodi di Feature Selection (come evaluate_feature_selection.py)
    su un campione e seleziona il migliore in base al modello scelto.

    Metodi testati:
    1. PCA (Elbow Method)
    2. Lasso Embedded
    3. Pairwise Correlation
    4. Relief
    5. Information Gain
    6. Sequential Forward Selection
    7. Sequential Backward Selection
    8. Bidirectional Subset Selection
    9. Max-Min Subset Selection
    10. Best First Search

    Args:
        X_train: feature di training completo (con geo-embedding)
        y_train: target di training completo
        scelta_modello: '1' (KNN), '2' (DT), '3' (SVM), '4' (Multi-Esperto → usa KNN come proxy)
        modalita: 'fast', 'balanced', 'thorough' (determina sample_size e CV fold)
        sample_size: numero di righe da usare (se None, viene calcolato dalla modalità)
        max_features: numero massimo di feature da selezionare
        max_rows_subset: righe max per subset selection (velocità)
        subset_cv_folds: fold per CV nei subset selector

    Returns:
        dict con:
        - 'metodo': migliore metodo FS
        - 'f1_best': F1-Micro del migliore
        - 'features': feature selezionate dal metodo migliore
        - 'risultati': dict completo con F1 per ogni metodo
    """
    # Determina parametri in base alla modalità
    if modalita == "fast":
        sample_size = sample_size or 5000
        max_features = max_features or 20
        max_rows_subset = max_rows_subset or 1000
        subset_cv_folds = subset_cv_folds or 1
        skip_subset_methods = True
    elif modalita == "balanced":
        sample_size = sample_size or 10000
        max_features = max_features or 30
        max_rows_subset = max_rows_subset or 2000
        subset_cv_folds = subset_cv_folds or 3
        skip_subset_methods = False
    elif modalita == "thorough":
        sample_size = sample_size or 15000
        max_features = max_features or 100
        max_rows_subset = max_rows_subset or 3000
        subset_cv_folds = subset_cv_folds or 5
        skip_subset_methods = False
    else:
        sample_size = sample_size or 10000
        max_features = max_features or 30
        max_rows_subset = max_rows_subset or 2000
        subset_cv_folds = subset_cv_folds or 3
        skip_subset_methods = False
    
    _banner("VALUTAZIONE DINAMICA FEATURE SELECTION (TUTTI I METODI)")
    print(f"  Modalità: {modalita.upper()}")
    print(f"  Parametri: sample={sample_size}, max_features={max_features}, CV_fold={subset_cv_folds}")
    if skip_subset_methods:
        print(f"  ⚠️  (modalità 'fast' skip subset methods per velocità)")
    
    # Campionamento stratificato per velocità
    print(f"\n  Campionamento {sample_size} righe per valutazione...")
    if len(X_train) > sample_size:
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train,
            train_size=sample_size,
            random_state=42,
            stratify=y_train
        )
    else:
        X_sample, y_sample = X_train.copy(), y_train.copy()
    
    # Filtra feature con varianza zero per evitare warning in correlazione
    print(f"  Filtraggio feature con varianza zero...")
    variance = X_sample.var()
    features_nonzero_var = variance[variance > 1e-10].index.tolist()
    X_sample_filtered = X_sample[features_nonzero_var].copy()
    print(f"    ✓ Feature con varianza > 0: {len(features_nonzero_var)}/{len(X_sample.columns)}")
    
    # Definisci i 10 metodi
    fs_methods = [
        ("PCA (Elbow)", PCAHandler, {}),
        ("Lasso Embedded", LassoFeatureSelector, {}),
        ("Pairwise Correlation", PairwiseCorrelationRanker, {}),
        ("Relief", ReliefRanker, {}),
        ("Information Gain", InformationGainRanker, {}),
        ("SFS", SequentialForwardSelector, {"estimator_name": "knn", "scoring": "f1_micro"}),
        ("SBS", SequentialBackwardSelector, {"estimator_name": "knn", "scoring": "f1_micro"}),
        ("Bidirectional", StepwiseBidirectionalSelector, {"estimator_name": "knn", "scoring": "f1_micro"}),
        ("Max-Min", MaxMinSubsetSelector, {}),
        ("Best First", BestFirstSelector, {}),
    ]
    
    risultati = {}
    metodo_features = {}  # Salva le feature per ogni metodo
    
    for method_name, MethodClass, kwargs in fs_methods:
        print(f"\n  Test metodo: {method_name}...")
        
        # Skip subset methods in "fast" mode
        subset_methods = {"SFS", "SBS", "Bidirectional"}
        if skip_subset_methods and method_name in subset_methods:
            print(f"    ⏭️  Skipped (modalità fast)")
            continue
        
        try:
            selected_features = None
            pca_model = None
            
            # ── PCA ────────────────────────────────────
            if method_name == "PCA (Elbow)":
                model = MethodClass()
                model.fit(X_sample_filtered, exclude_columns=[])
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
                selected_features = [f"PC{i}" for i in range(1, elbow_k + 1)]
                X_sample_transformed = model.transform(X_sample_filtered).iloc[:, :elbow_k]
                pca_model = (X_sample_transformed, model)
            
            # ── Lasso Embedded ────────────────────────
            elif method_name == "Lasso Embedded":
                model = MethodClass()
                result = model.select(X_sample_filtered, y_sample)
                ranking_df = result["selected_features"].copy()
                # Usa knee-point detection per trovare numero ottimale di feature
                if len(ranking_df) > 0 and "score" in ranking_df.columns:
                    optimal_k = _find_optimal_features_from_ranking(ranking_df["score"], max_features)
                else:
                    optimal_k = min(max_features, len(ranking_df))
                selected_features = ranking_df["feature"].head(optimal_k).tolist()
                selected_features = [f for f in selected_features if f in X_sample.columns]
            
            # ── Pairwise Correlation ──────────────────
            elif method_name == "Pairwise Correlation":
                model = MethodClass(**kwargs)
                X_temp = X_sample_filtered.copy()
                X_temp["__target__"] = y_sample.values
                result = model.rank(X_temp, label_column="__target__")
                ranking_key = "combined_ranking" if "combined_ranking" in result else "supervised_ranking"
                ranking_df = result[ranking_key].copy()
                # Usa knee-point detection per trovare numero ottimale di feature
                if len(ranking_df) > 0 and "score" in ranking_df.columns:
                    optimal_k = _find_optimal_features_from_ranking(ranking_df["score"], max_features)
                else:
                    optimal_k = min(max_features, len(ranking_df))
                selected_features = ranking_df["feature"].head(optimal_k).tolist()
                selected_features = [f for f in selected_features if f in X_sample_filtered.columns and f != "__target__"]
            
            # ── Relief ────────────────────────────────
            elif method_name == "Relief":
                model = MethodClass(random_state=42)
                X_temp = X_sample_filtered.copy()
                X_temp["__target__"] = y_sample.values
                result = model.rank(X_temp, label_column="__target__")
                ranking_df = result["relief_ranking"].copy()
                # Usa knee-point detection per trovare numero ottimale di feature
                if len(ranking_df) > 0 and "score" in ranking_df.columns:
                    optimal_k = _find_optimal_features_from_ranking(ranking_df["score"], max_features)
                else:
                    optimal_k = min(max_features, len(ranking_df))
                selected_features = ranking_df["feature"].head(optimal_k).tolist()
                selected_features = [f for f in selected_features if f in X_sample_filtered.columns]
            
            # ── Information Gain ──────────────────────
            elif method_name == "Information Gain":
                model = InformationGainRanker(log_base=2)
                X_temp = X_sample_filtered.copy()
                X_temp["__target__"] = y_sample.values
                result = model.rank(X_temp, label_column="__target__")
                ranking_df = result["information_gain_ranking"].copy()
                # Usa knee-point detection per trovare numero ottimale di feature
                if len(ranking_df) > 0 and "score" in ranking_df.columns:
                    optimal_k = _find_optimal_features_from_ranking(ranking_df["score"], max_features)
                else:
                    optimal_k = min(max_features, len(ranking_df))
                selected_features = ranking_df["feature"].head(optimal_k).tolist()
                selected_features = [f for f in selected_features if f in X_sample_filtered.columns]
            
            # ── Subset Selection methods ───────────────
            elif method_name == "SFS":
                model = MethodClass(**kwargs)
                # SFS cresce fino al limite naturale con early stopping intelligente
                # min_improvement=0.001 ferma quando il miglioramento scende sotto 0.1%
                res = model.select(X_sample_filtered, y_sample.to_numpy(),
                                   max_features=None,  # Nessun limite fisso
                                   max_rows=max_rows_subset,
                                   cv_folds=subset_cv_folds,
                                   min_improvement=0.001)  # Early stopping intelligente
                selected_features = res["selected_features"]["selected_feature"].tolist()
                selected_features = [f for f in selected_features if f in X_sample.columns]
            
            elif method_name == "SBS":
                model = MethodClass(**kwargs)
                # SBS reduce feature a partire da tutte, fino al limite naturale
                # min_improvement=0.001 ferma quando il miglioramento scende sotto 0.1%
                res = model.select(X_sample_filtered, y_sample.to_numpy(),
                                   min_features=5,  # Non scendere sotto 5 feature
                                   max_rows=max_rows_subset,
                                   cv_folds=subset_cv_folds,
                                   min_improvement=0.001)  # Early stopping intelligente
                selected_features = res["selected_features"]["selected_feature"].tolist()
                selected_features = [f for f in selected_features if f in X_sample.columns]
            
            elif method_name == "Bidirectional":
                model = MethodClass(**kwargs)
                # Bidirectional cresce e shrink liberamente verso l'ottimo
                # min_improvement=0.001 ferma quando il miglioramento scende sotto 0.1%
                res = model.select(X_sample_filtered, y_sample.to_numpy(),
                                   min_features=5,  # Non scendere sotto 5 feature
                                   max_features=None,  # Nessun limite massimo fisso
                                   max_rows=max_rows_subset,
                                   cv_folds=subset_cv_folds,
                                   min_improvement=0.001)  # Early stopping intelligente
                selected_features = res["selected_features"]["selected_feature"].tolist()
                selected_features = [f for f in selected_features if f in X_sample.columns]
            
            elif method_name == "Max-Min":
                model = MethodClass(**kwargs)
                res = model.select(X_sample_filtered, y_sample)
                selected_features = res["selected_features"]["selected_feature"].tolist()
                selected_features = [f for f in selected_features if f in X_sample.columns]
            
            elif method_name == "Best First":
                model = MethodClass(**kwargs)
                y_arr = y_sample.to_numpy() if isinstance(y_sample, pd.Series) else y_sample
                res = model.select(X_sample_filtered, y_arr, max_rows=max_rows_subset)
                selected_features = res["selected_features"]["selected_feature"].tolist()
                selected_features = [f for f in selected_features if f in X_sample.columns]
            
            if not selected_features:
                print(f"    ⚠️  {method_name} non ha selezionato feature, skip")
                risultati[method_name] = {"f1": 0.0, "n_features": 0}
                continue
            
            metodo_features[method_name] = selected_features
            
            # Split per valutazione
            if pca_model:
                X_fs_train = pca_model[0]
                X_fs_sample = X_fs_train.copy()
            else:
                X_fs_sample = X_sample[selected_features].copy()
            
            X_train_fs, X_val_fs, y_train_fs, y_val_fs = train_test_split(
                X_fs_sample, y_sample,
                test_size=0.2,
                random_state=42,
                stratify=y_sample
            )
            
            # Addestra modello scelto con verbosità ridotta
            print(f"    Addestramento modello...")
            if scelta_modello == "1":
                modello_info = train_knn(X_train_fs, y_train_fs, X_val_fs, y_val_fs, verbose=True)
            elif scelta_modello == "2":
                modello_info = train_decisiontree(X_train_fs, y_train_fs, X_val_fs, y_val_fs, verbose=True)
            elif scelta_modello == "3":
                modello_info = train_svm(X_train_fs, y_train_fs, X_val_fs, y_val_fs, verbose=False)
            elif scelta_modello == "4":
                # Multi-Esperto: usa Random Forest come proxy durante la FS (il sistema
                # completo sarebbe troppo lento su un campione iterato per ogni metodo FS)
                print(f"    (Multi-Esperto: uso Random Forest come proxy per la valutazione FS)")
                modello_info = train_randomforest(X_train_fs, y_train_fs, X_val_fs, y_val_fs, verbose=False)
            else:
                print(f"    ⚠️  Modello {scelta_modello} non supportato")
                continue
            
            if modello_info is None:
                print(f"    ⚠️  {method_name}: addestramento fallito")
                risultati[method_name] = {"f1": 0.0, "n_features": len(selected_features)}
                continue
            
            # Estrai F1-Micro
            f1_val = modello_info.get("metrics", {}).get("f1_micro", 0.0)
            risultati[method_name] = {"f1": f1_val, "n_features": len(selected_features)}
            print(f"    ✓ {method_name}: F1-Micro = {f1_val:.4f} ({len(selected_features)} feature)")
            
        except Exception as e:
            print(f"    ❌ Errore con {method_name}: {str(e)}")
            risultati[method_name] = {"f1": 0.0, "n_features": 0}
            continue
    
    # Trova il migliore
    if not risultati:
        print("  ⚠️  Nessun metodo FS riuscito, userò Relief di default")
        return {"metodo": "relief", "f1_best": 0.0, "features": [], "risultati": {}}
    
    metodo_best = max(risultati.items(), key=lambda x: x[1]["f1"])
    best_method_name = metodo_best[0]
    best_features = metodo_features.get(best_method_name, [])
    
    print(f"\n  🏆 MIGLIORE: {best_method_name} con F1-Micro = {metodo_best[1]['f1']:.4f}")
    
    print(f"\n  Ranking metodi:")
    for name, metrics in sorted(risultati.items(), key=lambda x: x[1]["f1"], reverse=True):
        print(f"    {name:25s} → F1 = {metrics['f1']:.4f} ({metrics['n_features']:2d} feature)")
    
    return {
        "metodo": best_method_name,
        "f1_best": metodo_best[1]["f1"],
        "features": best_features,
        "risultati": risultati
    }


# ============================================================
# FASE 1 — OUTLIER DETECTION
# ============================================================

def menu_outlier_detection() -> str:
    """
    Fase 1: scelta del metodo di outlier detection.

    Ritorna:
        '1' -> IQR  (Interquartile Range)
        '2' -> DBSCAN (multivariato)
    """
    _banner("FASE 1 — OUTLIER DETECTION")
    print("  1) IQR    — Interquartile Range (per colonna, soglia classica)")
    print("  2) DBSCAN — outlier detection multivariato con hyperparameter tuning")
    print()
    try:
        scelta = input("Seleziona metodo [1-2] (default=1): ").strip()
    except EOFError:
        scelta = ""
    if scelta not in {"1", "2"}:
        scelta = "1"
    print(f"\n>> Metodo selezionato: {'IQR' if scelta == '1' else 'DBSCAN'}")
    return scelta


# ============================================================
# FASE 2 — IMPUTAZIONE
# ============================================================

def menu_strategia_imputazione_outlier_numerici() -> str:
    """
    Fase 2: scelta della strategia di imputazione per i NaN creati
    dalla rimozione degli outlier.

    Ritorna codice stringa '1'-'4'.
    """
    _banner("FASE 2 — IMPUTAZIONE FEATURE NUMERICHE (outlier → NaN)")
    for strategia in STRATEGIE_IMPUTAZIONE.values():
        print(f"  {strategia.codice_menu}) {strategia.nome_menu}")
    print()
    try:
        scelta = input("Seleziona opzione [1-4] (default=4): ").strip()
    except EOFError:
        scelta = ""
    if scelta not in {"1", "2", "3", "4"}:
        scelta = "4"
    return scelta


# ============================================================
# FASE 3 — GEO EMBEDDING
# ============================================================

def menu_geo_embedding_tipo() -> str:
    """
    Fase 3a: scelta tra embedding statico o rete neurale.

    Ritorna:
        'embedding' -> embedding statico
        'neural' -> rete neurale con n_hidden layer
    """
    _banner("FASE 3A — GEO-LEVEL EMBEDDING (Scelta Tipo)")
    print("  Gestione dei geo_level_1/2/3:")
    print("  1) Embedding statico      — approccio discreto/categorico")
    print("  2) Rete neurale           — autoencoder per rappresentazioni dense")
    print()
    try:
        scelta = input("Seleziona tipo embedding [1-2] (default=2): ").strip()
    except EOFError:
        scelta = ""
    if scelta not in {"1", "2"}:
        scelta = "2"

    tipo = "embedding" if scelta == "1" else "neural"
    print(f"\n>> Tipo embedding selezionato: {tipo}")
    return tipo


def menu_geo_embedding_neural() -> str:
    """
    Fase 3b: scelta del numero di hidden layer per la rete neurale
    (solo se tipo == 'neural').

    Ritorna stringa con numero hidden layer ('1', '2', ...).
    Default = 2.
    """
    _banner("FASE 3B — GEO-LEVEL EMBEDDING (Rete Neurale - Hidden Layer)")
    print("  Numero di hidden layer per l'autoencoder:")
    print("  1) 1 hidden layer  — rete leggera")
    print("  2) 2 hidden layer  — default consigliato")
    print("  3) 3 hidden layer  — rete più espressiva")
    print()
    try:
        scelta = input("Seleziona numero hidden layer [1-3] (default=2): ").strip()
    except EOFError:
        scelta = ""
    if scelta not in {"1", "2", "3"}:
        scelta = "2"
    print(f"\n>> Rete neurale con {scelta} hidden layer selezionato.")
    return scelta


# ============================================================
# FASE 4 — SCELTA MODELLO  [parzialmente DA IMPLEMENTARE]
# ============================================================

def menu_scelta_modello() -> str:
    """
    Fase 4: scelta del modello di classificazione principale.

    Modelli disponibili:
        1 -> KNN
        2 -> Albero Decisionale
        3 -> SVM (Support Vector Machine)
        4 -> Random Forest

    Ritorna stringa '1'-'4'.
    """
    _banner("FASE 4 — SCELTA MODELLO")
    print("  1) KNN                — K-Nearest Neighbors")
    print("  2) Albero Decisionale — Decision Tree")
    print("  3) SVM                — Support Vector Machine (LinearSVC, One-vs-Rest)")
    print("  4) Random Forest      — ensemble di alberi decisionali")
    print()
    try:
        scelta = input("Seleziona modello [1-4] (default=1): ").strip()
    except EOFError:
        scelta = ""
    if scelta not in {"1", "2", "3", "4"}:
        scelta = "1"

    nomi = {
        "1": "KNN",
        "2": "Albero Decisionale",
        "3": "SVM",
        "4": "Random Forest",
    }
    print(f"\n>> Modello selezionato: {nomi[scelta]}")

    return scelta


# ============================================================
# UTILITY
# ============================================================

def sostituisci_outlier_con_nan(df, colonna, lower_bound, upper_bound, metodo="iqr", valori_anomali=None):
    """Sostituisce con NaN gli outlier rilevati con IQR o rarita se IQR=0."""
    if colonna not in df.columns:
        return df, 0
    df_out = df.copy()
    if metodo == "rarity_iqr_zero" and valori_anomali:
        mask_outlier = df_out[colonna].isin(valori_anomali)
    else:
        mask_outlier = (df_out[colonna] < lower_bound) | (df_out[colonna] > upper_bound)

    n_sostituiti = int(mask_outlier.sum())
    df_out.loc[mask_outlier, colonna] = pd.NA
    return df_out, n_sostituiti


def esegui_silenzioso(funzione, *args, **kwargs):
    """Esegue una funzione sopprimendo le stampe su stdout."""
    with redirect_stdout(io.StringIO()):
        return funzione(*args, **kwargs)


def salva_dataset_preprocessati(train_values, train_labels, test_values, output_dir="Data/preprocessed"):
    """Crea la cartella e salva i dataset preprocessati in CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_train = output_path / "train_values_preprocessed.csv"
    file_test = output_path / "test_values_preprocessed.csv"
    file_train_con_label = output_path / "train_features_labels_preprocessed.csv"

    train_values.to_csv(file_train, index=False)
    test_values.to_csv(file_test, index=False)
    pd.merge(train_values, train_labels, on="building_id").to_csv(file_train_con_label, index=False)

    _banner("SALVATAGGIO DATASET PREPROCESSATI")
    print(f"File salvati in: {output_path.resolve()}")
    print(f"  - {file_train.name}")
    print(f"  - {file_test.name}")
    print(f"  - {file_train_con_label.name}")


# ============================================================
# STUB MODELLI  [fasi 4-5, parte DA IMPLEMENTARE]
# ============================================================

def _stub_da_implementare(nome_sezione: str) -> None:
    _banner(f"[DA IMPLEMENTARE] — {nome_sezione}")
    print(f"  La sezione '{nome_sezione}' non è ancora implementata.")
    print("  Verrà integrata nelle prossime iterazioni dello sviluppo.")


# ============================================================
# STUB MODELLI  [fasi 4-5, parte DA IMPLEMENTARE]
# ============================================================

def _stub_da_implementare(nome_sezione: str) -> None:
    _banner(f"[DA IMPLEMENTARE] — {nome_sezione}")
    print(f"  La sezione '{nome_sezione}' non è ancora implementata.")
    print("  Verrà integrata nelle prossime iterazioni dello sviluppo.")


def train_multi_expert_system(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    verbose: bool = True,
) -> dict:
    """
    Addestra un sistema multi-esperto con Random Forest, KNN e Decision Tree.

    Il Random Forest è integrato in questa sezione come esperto del sistema
    combinato. Gli esperti vengono addestrati singolarmente, poi aggregati
    con la regola ``weighted_mean`` del decision profile.

    Args:
        X_train: feature di training (DataFrame)
        y_train: target di training (Series)
        X_val: feature di validation (DataFrame)
        y_val: target di validation (Series)
        verbose: stampe a console

    Returns:
        dict con chiavi:
        - "model": modello MultiExpertSystem addestrato
        - "best_params": dict con parametri degli esperti
        - "metrics": metriche di valutazione
        - "n_features": numero di feature utilizzate
    """
    if verbose:
        print(f"\n{'='*80}")
        print("MODELLO MULTI-ESPERTO (Random Forest + KNN + Decision Tree)")
        print(f"{'='*80}\n")

    # Addestra gli esperti singolarmente per estrarre i modelli migliori
    print("  Addestramento esperti singoli...")

    try:
        rf_result = train_randomforest(X_train, y_train, X_val, y_val, verbose=True)
        rf_f1 = rf_result.get('metrics', {}).get('f1_micro', 0)
        print(f"    ✓ Random Forest addestrato (F1: {rf_f1:.4f})")
    except Exception as e:
        print(f"    ❌ Errore RF: {e}")
        rf_result = None

    try:
        knn_result = train_knn(X_train, y_train, X_val, y_val, verbose=False)
        knn_f1 = knn_result.get('metrics', {}).get('f1_micro', 0)
        print(f"    ✓ KNN addestrato (F1: {knn_f1:.4f})")
    except Exception as e:
        print(f"    ❌ Errore KNN: {e}")
        knn_result = None

    try:
        dt_result = train_decisiontree(X_train, y_train, X_val, y_val, verbose=False)
        dt_f1 = dt_result.get('metrics', {}).get('f1_micro', 0)
        print(f"    ✓ Decision Tree addestrato (F1: {dt_f1:.4f})")
    except Exception as e:
        print(f"    ❌ Errore DT: {e}")
        dt_result = None

    # Crea il sistema multi-esperto
    if not any([rf_result, knn_result, dt_result]):
        raise ValueError("Nessun esperto è stato addestrato con successo.")

    experts = []
    best_params = {}

    if rf_result:
        experts.append(ExpertSpec(
            name="Random Forest",
            estimator=rf_result["model"],
            weight=1.0
        ))
        best_params["rf"] = rf_result.get("best_params", {})

    if knn_result:
        experts.append(ExpertSpec(
            name="KNN",
            estimator=knn_result["model"],
            weight=1.0
        ))
        best_params["knn"] = knn_result.get("best_params", {})

    if dt_result:
        experts.append(ExpertSpec(
            name="Decision Tree",
            estimator=dt_result["model"],
            weight=1.0
        ))
        best_params["dt"] = dt_result.get("best_params", {})

    print(f"\n  Creazione sistema multi-esperto ({len(experts)} esperti)...")
    multi_expert_system = MultiExpertSystem(experts, aggregation="weighted_mean")
    multi_expert_system.fit(X_train, y_train)
    print(f"  ✓ Sistema multi-esperto creato e addestrato")

    # Valutazione sul validation set
    y_pred = multi_expert_system.predict(X_val)
    f1_micro_val = f1_score(y_val, y_pred, average="micro", zero_division=0)
    accuracy_val = accuracy_score(y_val, y_pred)

    if verbose:
        print(f"\n  📊 Metriche su VALIDATION set:")
        print(f"    - F1 Micro:              {f1_micro_val:.4f}")
        print(f"    - Accuracy:              {accuracy_val:.4f}")

        # Mostra contributi esperti
        diversity_df = multi_expert_system.diversity_report(X_val, y_val)
        if not diversity_df.empty:
            print(f"\n  🤝 Diversità tra esperti:")
            print(f"    - Disagreement (media):  {diversity_df['disagreement'].mean():.4f}")
            print(f"    - Double Fault (media):  {diversity_df['double_fault'].mean():.4f}")

    return {
        "model": multi_expert_system,
        "best_params": best_params,
        "metrics": {
            "f1_micro": f1_micro_val,
            "accuracy": accuracy_val,
        },
        "n_features": X_train.shape[1],
    }


def esegui_modello(
    scelta_modello: str,
    train_values: pd.DataFrame,
    train_labels: pd.Series,
    selected_features: list[str],
    metodo_fs: str | None = None,
) -> dict:
    """
    Fase 4: addestramento del modello scelto con hyperparameter tuning.

    La pipeline:
    1. Estrae le feature selezionate dai dati
    2. Fa split train/validation (80/20) stratificato
    3. Addestra il modello su TRAIN
    4. Valuta su VALIDATION
    5. Ritorna metriche e modello

    Args:
        scelta_modello: '1' (KNN), '2' (DT), '3' (SVM), '4' (Multi-Esperto)
        train_values: features di training (con geo-feature già aggiunte)
        train_labels: target di training
        selected_features: lista di feature selezionate
        metodo_fs: nome del metodo di feature selection usato

    Returns:
        dizionario con risultati, metriche, modello e feature usate
    """
    _banner("FASE 4 — ADDESTRAMENTO MODELLO")

    # === Preparazione dati ===
    print(f"\n  Preparazione dati...")
    print(f"  Feature selezionate: {len(selected_features)}")
    print(f"  Prime 5 feature: {selected_features[:5]}")

    y = train_labels.copy()

    # === Split train/validation ===
    print(f"\n  Split stratificato train/validation (80/20)...")
    usa_spazio_pca = _should_use_pca_feature_space(
        metodo_fs=metodo_fs,
        selected_features=selected_features,
        available_columns=train_values.columns,
    )

    if usa_spazio_pca:
        print("  Rilevata selezione PCA: ricostruisco le componenti sul train split.")

        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            train_values, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        pca_handler = PCAHandler()
        pca_handler.fit(X_train_raw, exclude_columns=[])

        n_components = len(selected_features)
        X_train = pca_handler.transform(X_train_raw).iloc[:, :n_components].copy()
        X_val = pca_handler.transform(X_val_raw).iloc[:, :n_components].copy()
        selected_features = X_train.columns.tolist()

        print(f"  ✓ Componenti PCA disponibili: {len(selected_features)}")
        print(f"  ✓ Prime 5 componenti: {selected_features[:5]}")
    else:
        # Verifica che le feature selezionate esistano nei dati
        feature_mancanti = [f for f in selected_features if f not in train_values.columns]
        if feature_mancanti:
            print(f"  ⚠ Feature non trovate: {feature_mancanti}")
            selected_features = [f for f in selected_features if f in train_values.columns]
            print(f"  ✓ Usando {len(selected_features)} feature disponibili")

        # Estrai X (solo feature selezionate) e y
        X = train_values[selected_features].copy()
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

    print(f"  Train: {X_train.shape[0]} righe")
    print(f"  Validation: {X_val.shape[0]} righe")
    print(f"  Feature usate: {X_train.shape[1]}")

    # === Selezione e addestramento modello ===
    nome_modello = None
    modello_info = None

    if scelta_modello == "1":
        nome_modello = "K-Nearest Neighbors (KNN)"
        print(f"\n  Addestramento: {nome_modello}")
        modello_info = train_knn(X_train, y_train, X_val, y_val, verbose=True)

    elif scelta_modello == "2":
        nome_modello = "Decision Tree"
        print(f"\n  Addestramento: {nome_modello}")
        modello_info = train_decisiontree(X_train, y_train, X_val, y_val, verbose=True)

    elif scelta_modello == "3":
        nome_modello = "SVM (LinearSVC, One-vs-Rest)"
        print(f"\n  Addestramento: {nome_modello}")
        modello_info = train_svm(X_train, y_train, X_val, y_val, verbose=True)

    elif scelta_modello == "4":
        nome_modello = "Random Forest"
        print(f"\n  Addestramento: {nome_modello}")
        modello_info = train_randomforest(X_train, y_train, X_val, y_val, verbose=True)

    else:
        print("  ERRORE: modello non riconosciuto.")
        return None

    if modello_info is None:
        print("  ❌ ERRORE: Modello non addestrato correttamente.")
        return None

    # === Preparazione risultati ===
    print(f"\n  Preparazione risultati...")

    risultati = {
        "scelta_modello": scelta_modello,
        "nome_modello": nome_modello,
        "modello": modello_info.get("model"),
        "best_params": modello_info.get("best_params", {}),
        "n_features": len(selected_features),
        "selected_features": selected_features,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
    }

    # Aggiungi metriche se disponibili
    if "metrics" in modello_info:
        risultati.update(modello_info["metrics"])

    # === Diagnostica Overfitting ===
    # Calcola F1-Micro sul TRAINING set per confrontarlo con il VALIDATION set.
    # Un gap elevato (train >> val) indica overfitting.
    modello_addestrato = modello_info.get("model")
    if modello_addestrato is not None:
        try:
            from sklearn.base import clone
            from sklearn.model_selection import StratifiedKFold, cross_val_score

            print(f"\n  {'─'*76}")
            print(f"  📊 DIAGNOSTICA OVERFITTING")
            print(f"  {'─'*76}")

            # ── 1. Train vs Validation ──────────────────────────────────────
            y_pred_train = modello_addestrato.predict(X_train)
            f1_train = f1_score(y_train, y_pred_train, average="micro", zero_division=0)

            f1_val = modello_info.get("metrics", {}).get("f1_micro")
            if f1_val is None:
                y_pred_val = modello_addestrato.predict(X_val)
                f1_val = f1_score(y_val, y_pred_val, average="micro", zero_division=0)

            gap = f1_train - f1_val
            gap_pct = gap * 100

            print(f"\n  [1] Train vs Validation")
            print(f"      F1-Micro TRAIN:      {f1_train:.4f}")
            print(f"      F1-Micro VALIDATION: {f1_val:.4f}")
            print(f"      Gap (train - val):   {gap:+.4f}  ({gap_pct:+.1f}%)")

            if gap > 0.10:
                print(f"      ⚠️  POSSIBILE OVERFITTING — gap > 10%")
                print(f"         Considera: regolarizzazione più forte, più dati, pruning.")
            elif gap > 0.05:
                print(f"      ⚡ Leggero overfitting — gap tra 5% e 10%.")
            elif gap < -0.02:
                print(f"      ℹ️  Gap negativo (val > train): validation set favorevole")
                print(f"         o class_weight=balanced che aiuta le classi rare.")
            else:
                print(f"      ✅ Nessun overfitting significativo — gap ≤ 5%.")

            risultati["f1_micro_train"] = round(f1_train, 4)
            risultati["overfitting_gap"] = round(gap, 4)

            # ── 2. Cross-Validation 5-fold ──────────────────────────────────
            print(f"\n  [2] Cross-Validation 5-fold stratificata (su training set)")

            # Campiona max 20k righe per mantenere tempi ragionevoli
            CV_MAX_ROWS = 20_000
            if len(X_train) > CV_MAX_ROWS:
                X_cv, _, y_cv, _ = train_test_split(
                    X_train, y_train,
                    train_size=CV_MAX_ROWS,
                    random_state=42,
                    stratify=y_train,
                )
                print(f"      (campione {CV_MAX_ROWS} righe su {len(X_train)} per velocità)")
            else:
                X_cv, y_cv = X_train, y_train

            try:
                estimatore_cv = clone(modello_addestrato)
            except Exception:
                # Se clone non riesce (es. MultiExpertSystem), salta la CV
                raise RuntimeError("clone() non supportato per questo modello.")

            cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                estimatore_cv, X_cv, y_cv,
                cv=cv_strategy,
                scoring="f1_micro",
                n_jobs=1,
            )

            cv_mean = cv_scores.mean()
            cv_std  = cv_scores.std()

            print(f"      Scores per fold: {' | '.join(f'{s:.4f}' for s in cv_scores)}")
            print(f"      Media:  {cv_mean:.4f}")
            print(f"      Std:    {cv_std:.4f}")

            # Confronto CV mean vs validation holdout
            cv_val_diff = cv_mean - f1_val
            print(f"      CV media vs Validation: {cv_val_diff:+.4f}")
            if abs(cv_val_diff) <= 0.02:
                print(f"      ✅ CV ≈ Validation — stima holdout affidabile.")
            elif cv_val_diff > 0.02:
                print(f"      ⚡ CV > Validation — il validation set è leggermente difficile.")
            else:
                print(f"      ⚡ CV < Validation — il validation set è leggermente facile.")

            # Stabilità: std alta = modello instabile tra fold
            if cv_std > 0.03:
                print(f"      ⚠️  Alta varianza tra fold (std > 3%) — modello instabile.")
                print(f"         Prova più dati o regolarizzazione più forte.")
            else:
                print(f"      ✅ Bassa varianza tra fold (std ≤ 3%) — modello stabile.")

            risultati["cv_f1_mean"] = round(cv_mean, 4)
            risultati["cv_f1_std"]  = round(cv_std, 4)

        except Exception as e:
            print(f"  ⚠️  Impossibile calcolare diagnostica overfitting: {e}")

    return risultati


def presenta_risultati(
    scelta_modello: str,
    scelta_outlier: str,
    strategia_imputazione: str,
    tipo_geo_embedding: str,
    metodo_fs: str,
    risultati_modello,
) -> None:
    """
    Fase 5: presentazione dettagliata dei risultati e delle scelte effettuate.
    
    Args:
        scelta_modello: '1'-'4' dal menu modelli
        scelta_outlier: '1' (IQR) o '2' (DBSCAN)
        strategia_imputazione: nome della strategia di imputazione
        tipo_geo_embedding: 'embedding' o 'neural'
        metodo_fs: 'none', 'relief', 'info_gain'
        risultati_modello: dizionario ritornato da esegui_modello()
    """
    if risultati_modello is None:
        _banner("FASE 5 — RISULTATI (Non disponibili)")
        print("  Nessun modello è stato addestrato correttamente.")
        return

    _banner("FASE 5 — RISULTATI FINALI")

    def _format_imputation_strategy(s):
        # Accetta oggetto Strategy, codice o nome_report e ritorna etichetta umana
        try:
            from src.preprocessing.imputation_strategies import STRATEGIE_IMPUTAZIONE
        except Exception:
            STRATEGIE_IMPUTAZIONE = {}

        if s is None:
            return "Sconosciuta"
        # se è un oggetto strategy con nome_menu
        if hasattr(s, 'nome_menu'):
            return getattr(s, 'nome_menu')
        # se è un codice del registry (es. '1','2') trova l'oggetto
        if isinstance(s, str) and s in STRATEGIE_IMPUTAZIONE:
            return STRATEGIE_IMPUTAZIONE[s].nome_menu
        # se è il nome_report (es. 'univariata_media'), cerca nell'oggetto
        if isinstance(s, str):
            for strat in STRATEGIE_IMPUTAZIONE.values():
                if getattr(strat, 'nome_report', None) == s:
                    return getattr(strat, 'nome_menu')
        return str(s)

    # === Riepilogo scelte ===
    print("\n📋 RIEPILOGO SCELTE EFFETTUATE:\n")
    
    print(f"  🔹 Outlier Detection:      {'IQR' if scelta_outlier == '1' else 'DBSCAN'}")
    print(f"  🔹 Imputazione:            {_format_imputation_strategy(strategia_imputazione)}")
    print(f"  🔹 Geo Embedding:          {'Embedding statico' if tipo_geo_embedding == 'embedding' else 'Rete neurale'}")
    print(f"  🔹 Feature Selection:      {metodo_fs.replace('_', ' ').upper()}")
    print(f"  🔹 Modello:                {risultati_modello.get('nome_modello', '?')}")
    print(f"  🔹 Feature usate:          {risultati_modello.get('n_features', '?')}")

    # === Risultati modello ===
    print("\n📊 RISULTATI MODELLO:\n")
    
    best_params = risultati_modello.get("best_params", {})
    if best_params:
        print("  Miglior configurazione iperparametri:")
        # Mappa nomi parametri in etichette umane quando possibile
        param_display_map = {
            'metric': 'Distanza',
            'n_neighbors': 'Numero vicini',
            'weights': 'Pesi',
            'max_depth': 'Profondità Max',
            'n_estimators': 'N. Alberi',
            'min_samples_leaf': 'Camp. Min Foglia',
            'min_samples_split': 'Camp. Min Split',
            'max_features': 'Feature Per Split',
        }
        for param, value in list(best_params.items())[:5]:
            display = param_display_map.get(param, param.replace('_', ' ').capitalize())
            print(f"    - {display}: {value}")
        if len(best_params) > 5:
            print(f"    ... e {len(best_params) - 5} altri parametri")
    
    # Metriche di validazione
    if "f1_micro_val" in risultati_modello:
        print(f"\n  📈 Metriche su VALIDATION set:")
        print(f"    - F1 Micro:              {risultati_modello.get('f1_micro_val', 'N/A'):.4f}")
    
    if "accuracy_val" in risultati_modello:
        print(f"    - Accuracy:              {risultati_modello.get('accuracy_val', 'N/A'):.4f}")
    
    print("\n  ℹ️  Nota: Le metriche sopra sono calcolate su un validation set separato")
    print("     (80% training, 20% validation) per evitare overfitting.")

    # === Feature selection details ===
    selected_features = risultati_modello.get("selected_features", [])
    if selected_features and metodo_fs != "none":
        print(f"\n🎯 FEATURE SELECTION ({len(selected_features)} feature selezionate):\n")
        # Mostra prime 10 feature
        for i, feat in enumerate(selected_features[:10], 1):
            print(f"    {i:2d}. {feat}")
        if len(selected_features) > 10:
            print(f"    ... e {len(selected_features) - 10} altre feature")
    
    print("\n" + "=" * 80)


# ============================================================
# PERFORMANCE CSV
# ============================================================

def _sanitize_csv_column_name(name: str) -> str:
    """Normalizza un nome colonna per l'export CSV."""
    sanitized = str(name).strip().lower()
    for old, new in (
        ("__", "_"),
        (" ", "_"),
        ("-", "_"),
        ("/", "_"),
        ("(", ""),
        (")", ""),
        (".", "_"),
        (":", "_"),
        (",", "_"),
    ):
        sanitized = sanitized.replace(old, new)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")


def _coerce_csv_value(value):
    """Converte valori arbitrari in un formato stabile per il CSV."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " | ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return value


def _build_performance_record(
    scelta_modello: str,
    scelta_outlier: str,
    strategia_imputazione: str | None,
    tipo_geo_embedding: str,
    metodo_fs: str,
    modalita_fs: str | None,
    risultati_modello: dict | None,
) -> dict[str, object]:
    """Costruisce un record leggibile e piatto per performances.csv."""
    now_local = datetime.now().astimezone()

    best_params = {}
    selected_features: list[str] = []
    if risultati_modello:
        best_params = risultati_modello.get("best_params", {}) or {}
        selected_features = list(risultati_modello.get("selected_features", []) or [])

    record: dict[str, object] = {
        "run_date": now_local.date().isoformat(),
        "run_time": now_local.strftime("%H:%M:%S"),
        "timezone": now_local.strftime("%z"),
        "timestamp_iso": now_local.isoformat(timespec="seconds"),
        "model_code": scelta_modello,
        "model_name": risultati_modello.get("nome_modello") if risultati_modello else "",
        "outlier_method": "IQR" if scelta_outlier == "1" else "DBSCAN",
        "imputation_strategy": strategia_imputazione or "",
        "geo_embedding": (
            "Embedding statico" if tipo_geo_embedding == "embedding" else "Rete neurale"
        ),
        "feature_selection_method": metodo_fs.replace("_", " ").upper(),
        "feature_selection_mode": modalita_fs or "",
        "n_features_used": risultati_modello.get("n_features") if risultati_modello else "",
        "n_selected_features": len(selected_features),
        "f1_micro_val": risultati_modello.get("f1_micro") if risultati_modello else "",
        "accuracy_val": risultati_modello.get("accuracy") if risultati_modello else "",
        "f1_macro_val": risultati_modello.get("f1_macro") if risultati_modello else "",
        "balanced_accuracy_val": (
            risultati_modello.get("balanced_accuracy") if risultati_modello else ""
        ),
        "cv_best_score": risultati_modello.get("cv_best_score") if risultati_modello else "",
        "fit_seconds": risultati_modello.get("fit_seconds") if risultati_modello else "",
        "tuning_seconds": risultati_modello.get("tuning_seconds") if risultati_modello else "",
        "selected_features_preview": ", ".join(selected_features[:10]),
        "selected_features_all": " | ".join(selected_features),
        "best_params_json": json.dumps(best_params, ensure_ascii=False, sort_keys=True, default=str),
    }

    max_feature_columns = 10
    for idx in range(1, max_feature_columns + 1):
        record[f"feature_{idx:02d}"] = selected_features[idx - 1] if idx <= len(selected_features) else ""

    for param_name, param_value in sorted(best_params.items()):
        column_name = f"param_{_sanitize_csv_column_name(param_name)}"
        record[column_name] = _coerce_csv_value(param_value)

    return record


def _append_performance_record(out_file: str, record: dict[str, object]) -> None:
    """Scrive il record nel CSV mantenendo uno schema leggibile e stabile."""
    existing_rows: list[dict[str, str]] = []
    existing_columns: list[str] = []

    if os.path.exists(out_file):
        with open(out_file, "r", newline="", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            existing_columns = list(reader.fieldnames or [])
            existing_rows = list(reader)

    preferred_columns = [
        "run_date",
        "run_time",
        "timezone",
        "timestamp_iso",
        "model_code",
        "model_name",
        "outlier_method",
        "imputation_strategy",
        "geo_embedding",
        "feature_selection_method",
        "feature_selection_mode",
        "n_features_used",
        "n_selected_features",
        "f1_micro_val",
        "accuracy_val",
        "f1_macro_val",
        "balanced_accuracy_val",
        "cv_best_score",
        "fit_seconds",
        "tuning_seconds",
        "selected_features_preview",
        "selected_features_all",
    ]
    preferred_columns.extend([f"feature_{idx:02d}" for idx in range(1, 11)])

    dynamic_param_columns = sorted(col for col in record.keys() if col.startswith("param_"))
    trailing_columns = ["best_params_json"]

    ordered_columns: list[str] = []
    for column in (
        preferred_columns
        + dynamic_param_columns
        + trailing_columns
        + existing_columns
        + list(record.keys())
    ):
        if column not in ordered_columns:
            ordered_columns.append(column)

    normalized_rows: list[dict[str, object]] = []
    for row in existing_rows:
        normalized_rows.append({column: row.get(column, "") for column in ordered_columns})

    normalized_record = {
        column: _coerce_csv_value(record.get(column, "")) for column in ordered_columns
    }
    normalized_rows.append(normalized_record)

    with open(out_file, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=ordered_columns)
        writer.writeheader()
        writer.writerows(normalized_rows)


# ============================================================
# MAIN
# ============================================================

def main():
    _banner("PIPELINE ML — TERREMOTO NEPAL 2015")
    print("  Fasi:")
    print("    1) Outlier Detection  (IQR / DBSCAN)")
    print("    2) Imputazione        (varie strategie)")
    print("    3) Geo Embedding      (rete neurale)")
    print("    4) Scelta Modello     (KNN / DT / RF / Multi-Esperto)")
    print("    5) Risultati          (F1-Micro)")

    # Track whether pipeline reached completion (Fase 5)
    pipeline_completed = False

    # ── FASE 1: Outlier Detection ──────────────────────────
    scelta_outlier = menu_outlier_detection()

    # ── CARICAMENTO DATI ───────────────────────────────────
    _banner("CARICAMENTO DATI")
    pulizia = PuliziaASCII()
    train_values, train_labels, test_values = pulizia.processa(
        colonne_categoriche=COLONNE_CATEGORICHE
    )

    # ── DATA QUALITY TRAIN ─────────────────────────────────
    train_quality_handler = DataQualityHandler(train_values)
    train_quality_report = train_quality_handler.esegui_controlli(plot=SHOW_DATA_QUALITY_PLOTS)
    train_values = train_quality_handler.data

    # Salvo l'upper bound di 'age' calcolato sul train (serve per monum_flag)
    age_upper_bound_train = train_quality_report["outliers"].loc["age", "upper_bound"]

    # ── DATA QUALITY TEST ──────────────────────────────────
    test_quality_handler = DataQualityHandler(test_values)
    test_quality_handler.pulisci_nomi_colonne()
    test_quality_handler.controlla_duplicati_building_id()
    test_quality_report = test_quality_handler.report
    test_values = test_quality_handler.data

    # ── SNAPSHOT età ORIGINALE ─────────────────────────────
    # Salvo la colonna 'age' PRIMA di qualsiasi modifica da outlier detection.
    # Serve per costruire monum_flag sul valore reale dell'edificio,
    # non sul valore imputato (che sarebbe sempre ≤ soglia dopo la pulizia).
    age_originale_train = train_values["age"].copy() if "age" in train_values.columns else None
    age_originale_test  = test_values["age"].copy()  if "age" in test_values.columns  else None

    # ── OUTLIER DETECTION E SOSTITUZIONE (solo TRAIN) ──────
    missing_handler = MissingValuesHandler(null_threshold=70)
    colonne_numeriche = [
        col for col in COLONNE_CONTINUE
        if col in train_values.columns and col in test_values.columns
    ]
    colonne_outlier = []
    outlier_replacement_counts = {}

    if scelta_outlier == "1":
        # --- IQR ---
        _banner("REPORT OUTLIER — TRAINING SET")
        print(train_quality_report["outliers"])
        
        _banner("OUTLIER DETECTION — IQR (solo training set)")
        outliers_df = train_quality_report.get("outliers")
        if outliers_df is not None:
            for col in colonne_numeriche:
                if col in outliers_df.index and float(outliers_df.loc[col, "n_outliers"]) > 0:
                    colonne_outlier.append(col)

        for col in colonne_outlier:
            lower_bound = float(outliers_df.loc[col, "lower_bound"])
            upper_bound = float(outliers_df.loc[col, "upper_bound"])
            metodo_outlier = outliers_df.loc[col, "metodo"] if "metodo" in outliers_df.columns else "iqr"
            valori_anomali = outliers_df.loc[col, "valori_anomali"] if "valori_anomali" in outliers_df.columns else []
            if not isinstance(valori_anomali, list):
                valori_anomali = []

            train_values, n_sost_train = sostituisci_outlier_con_nan(
                train_values,
                colonna=col,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                metodo=metodo_outlier,
                valori_anomali=valori_anomali,
            )
            outlier_replacement_counts[col] = {
                "lower_bound_train": lower_bound,
                "upper_bound_train": upper_bound,
                "metodo_outlier": metodo_outlier,
                "valori_anomali_train": valori_anomali,
                "n_valori_sostituiti_train": n_sost_train,
            }
            print(
                f"  [{col}] metodo={metodo_outlier} "
                f"bound=[{lower_bound:.2f}, {upper_bound:.2f}]  "
                f"outlier->NaN: {n_sost_train}"
            )

    else:
        # --- DBSCAN ---
        _banner("OUTLIER DETECTION — DBSCAN (solo training set)")

        # (l'IQR è già stato calcolato per il report, qui usiamo solo DBSCAN)
        outliers_df = train_quality_report.get("outliers")

        # Passo 2: DBSCAN multivariato
        print("\n  Avvio DBSCAN per raffinamento multivariato...")
        mask_outlier, info_dbscan = rileva_outlier_dbscan(train_values, COLONNE_CONTINUE)
        n_outlier_totali = int(mask_outlier.sum())

        print(f"\n  DBSCAN — parametri usati: {info_dbscan}")
        print(f"  DBSCAN — righe outlier individuate: {n_outlier_totali}")

        if n_outlier_totali > 0:
            colonne_outlier = list(colonne_numeriche)
            for col in colonne_outlier:
                n_nan_prima = int(train_values[col].isna().sum())
                train_values.loc[mask_outlier, col] = pd.NA
                n_nan_dopo = int(train_values[col].isna().sum())
                n_sostituiti = n_nan_dopo - n_nan_prima
                outlier_replacement_counts[col] = {
                    "lower_bound_train": float("nan"),
                    "upper_bound_train": float("nan"),
                    "n_valori_sostituiti_train": n_sostituiti,
                }
                print(f"  [{col}]  outlier→NaN (DBSCAN): {n_sostituiti}")
        else:
            print("  DBSCAN: nessun outlier individuato.")

    if not colonne_outlier:
        print("\n  Nessuna feature numerica con outlier: nessuna imputazione necessaria.")

    # ── FASE 2: Imputazione ────────────────────────────────
    scelta_imputazione = menu_strategia_imputazione_outlier_numerici()
    risultati_knn = None
    risultati_knn_per_colonna = {}

    imputation_reports = {}
    for col in colonne_outlier:
        train_values, test_values, col_report = applica_strategia_imputazione_colonna(
            missing_handler=missing_handler,
            train_values=train_values,
            test_values=test_values,
            scelta=scelta_imputazione,
            colonna=col,
        )
        col_report.update(outlier_replacement_counts[col])
        imputation_reports[col] = col_report

        _banner(f"IMPUTAZIONE '{col}' — {col_report['strategia'].upper()}")
        if scelta_outlier == "1":
            print(
                f"  Bound train [{col_report['lower_bound_train']:.2f}, "
                f"{col_report['upper_bound_train']:.2f}] → NaN: "
                f"{col_report['n_valori_sostituiti_train']}"
            )
        else:
            print(f"  Outlier DBSCAN → NaN: {col_report['n_valori_sostituiti_train']}")
        print(
            f"  Missing {col} train: "
            f"{col_report['n_missing_train_prima']} → {col_report['n_missing_train_dopo']}"
        )
        print(
            f"  Missing {col} test:  "
            f"{col_report['n_missing_test_prima']} → {col_report['n_missing_test_dopo']}"
        )
        # Se la strategia è univariata (media/mediana), mostriamo il valore calcolato sul train
        if col_report.get('strategia') in ('univariata_media', 'univariata_mediana'):
            valore = col_report.get('valore_imputazione_train')
            if valore is not None:
                print(f"  Valore usato per imputazione (train): {valore:.2f}")
        # Se la strategia è multivariata (regressione o knn), mostriamo statistiche dei valori imputati
        if col_report.get('strategia') in ('multivariata_regressione_lineare', 'knn_predictor'):
            # statistiche_imputazione possono contenere media/mediana/min/max per train/test
            media_train = col_report.get('train_media_imputata')
            mediana_train = col_report.get('train_mediana_imputata')
            min_train = col_report.get('train_min_imputato')
            max_train = col_report.get('train_max_imputato')

            media_test = col_report.get('test_media_imputata')
            mediana_test = col_report.get('test_mediana_imputata')
            min_test = col_report.get('test_min_imputato')
            max_test = col_report.get('test_max_imputato')

            if media_train is not None:
                print(f"  Valori imputati TRAIN — media: {media_train:.2f}, mediana: {mediana_train:.2f}, min: {min_train:.2f}, max: {max_train:.2f}")
            if media_test is not None:
                print(f"  Valori imputati TEST  — media: {media_test:.2f}, mediana: {mediana_test:.2f}, min: {min_test:.2f}, max: {max_test:.2f}")

    # ── NUOVA FEATURE: monum_flag (age_flag) ───────────────
    # La flag viene calcolata sull'età ORIGINALE dell'edificio (pre-outlier),
    # non sul valore imputato. Motivo:
    #   - IQR:   age > 90 viene messo a NaN poi imputato (≤ 90) → flag sempre 0
    #   - DBSCAN: l'intera riga va a NaN, age imputata ≈ media → flag sempre 0
    # Soluzione: sostituisco temporaneamente age con il valore originale,
    # calcolo il flag, poi ripristino i valori imputati nel dataset.

    # TRAIN
    _banner("NUOVA FEATURE: monum_flag (TRAIN)")
    if age_originale_train is not None:
        age_imputata_train = train_values["age"].copy()          # salvo age imputata
        train_values["age"] = age_originale_train.values         # ripristino age originale
        train_quality_handler.data = train_values
        train_values = train_quality_handler.aggiungi_feature_age_flag(upper_bound=age_upper_bound_train)
        train_values["age"] = age_imputata_train.values          # riporto age imputata
    else:
        train_quality_handler.data = train_values
        train_values = train_quality_handler.aggiungi_feature_age_flag(upper_bound=age_upper_bound_train)

    # TEST (usa age originale, non modificata dall'outlier detection)
    _banner("NUOVA FEATURE: monum_flag (TEST)")
    if age_originale_test is not None:
        test_quality_handler.data = test_values
        # Sul test non applichiamo outlier detection, quindi age_originale_test == age corrente.
        # Assegniamo comunque per coerenza col flusso TRAIN.
        test_values["age"] = age_originale_test.values
        test_values = test_quality_handler.aggiungi_feature_age_flag(upper_bound=age_upper_bound_train)
    else:
        test_quality_handler.data = test_values
        test_values = test_quality_handler.aggiungi_feature_age_flag(upper_bound=age_upper_bound_train)

    # ── VALIDAZIONE ────────────────────────────────────────
    validator_train = DataValidator(train_values)
    validation_report_train = esegui_silenzioso(validator_train.esegui_validazione, verbose=True)
    validator_test = DataValidator(test_values)
    validation_report_test = esegui_silenzioso(validator_test.esegui_validazione, verbose=True)

    # ── STANDARDIZZAZIONE ──────────────────────────────────
    train_quality_handler.data = train_values
    scaler = esegui_silenzioso(train_quality_handler.fit_standardizzazione)
    train_values = esegui_silenzioso(train_quality_handler.applica_standardizzazione, scaler)

    test_quality_handler.data = test_values
    test_values = esegui_silenzioso(test_quality_handler.applica_standardizzazione, scaler)

    # ── ONE-HOT ENCODING ───────────────────────────────────
    train_quality_handler.data = train_values
    ohe_encoder = esegui_silenzioso(train_quality_handler.fit_one_hot_encoding, COLONNE_CATEGORICHE)
    train_values = esegui_silenzioso(
        train_quality_handler.applica_one_hot_encoding, ohe_encoder, COLONNE_CATEGORICHE
    )
    test_quality_handler.data = test_values
    test_values = esegui_silenzioso(
        test_quality_handler.applica_one_hot_encoding, ohe_encoder, COLONNE_CATEGORICHE
    )

    # ── MISSING VALUES (log) ───────────────────────────────
    df_merged = pd.merge(train_values, train_labels, on="building_id")
    handler = MissingValuesHandler(null_threshold=70)
    report = esegui_silenzioso(handler.analizza, df_merged, target_col="damage_grade")
    report["numeric_outlier_imputation"] = imputation_reports

    # ── FASE 3: Geo Embedding ──────────────────────────────
    # Il menu viene mostrato PRIMA del salvataggio perché le geo-feature
    # verranno aggiunte al dataset
    tipo_geo_embedding = menu_geo_embedding_tipo()
    n_hidden = None
    if tipo_geo_embedding == "neural":
        n_hidden = int(menu_geo_embedding_neural())

    _banner("FASE 3 — GEO EMBEDDING (Esecuzione)")
    
    try:
        train_values, test_values = applica_geo_embedding(
            train_values=train_values,
            test_values=test_values,
            y_train=train_labels["damage_grade"],
            tipo=tipo_geo_embedding,
            smoothing=20.0,
            rare_threshold=10,
            n_splits=5,
        )
        print("  ✓ Geo-feature applicate con successo")
    except Exception as e:
        print(f"  ❌ ERRORE applicando geo-feature: {e}")
        import traceback
        traceback.print_exc()
        return None

    # ── SALVATAGGIO (dopo preprocessing + geo embedding) ──
    salva_dataset_preprocessati(train_values, train_labels, test_values)

    # ── FASE 3.5: Feature Selection ────────────────────────
    metodo_fs = menu_feature_selection()
    # Modalità di valutazione usata per la FS automatica (fast/balanced/thorough)
    modalita_fs = None
    
    _banner("FASE 3.5 — FEATURE SELECTION")
    
    # Prepara X (tutte le feature tranne target e building_id)
    X_train_fs = train_values.drop(columns=["building_id"], errors="ignore")
    y_train_fs = train_labels["damage_grade"]
    
    # Se automatica, valuta per il modello scelto
    if metodo_fs == "auto":
        print(f"  Scegli il modello per cui valutare il migliore FS...")
        scelta_modello_per_fs = menu_scelta_modello()
        
        # Menu qualità di valutazione
        modalita_fs = menu_qualita_fs_automatica()
        
        # Valuta TUTTI i metodi dinamicamente
        eval_result = valuta_fs_dinamica_per_modello(
            X_train=X_train_fs,
            y_train=y_train_fs,
            scelta_modello=scelta_modello_per_fs,
            modalita=modalita_fs,
        )
        
        metodo_fs = eval_result["metodo"]
        selected_features = eval_result["features"]
        
        print(f"\n  ✓ Metodo FS selezionato: {metodo_fs}")
        print(f"  ✓ Feature selezionate: {len(selected_features)}")
        
    else:
        print(f"  Metodo: {metodo_fs.upper()}")
        # Esegui feature selection
        selected_features = esegui_feature_selection(
            X_train=X_train_fs,
            y_train=y_train_fs,
            metodo=metodo_fs,
            top_k=50,  # Seleziona top 50 feature se usando FS
        )
    
    if not selected_features:
        print("  ⚠️  Warning: Nessuna feature selezionata, usando tutte")
        selected_features = X_train_fs.columns.tolist()
    
    print(f"  ✓ Selezionate {len(selected_features)} feature")

    # ── FASE 4: Scelta Modello ─────────────────────────────
    scelta_modello = menu_scelta_modello()
    
    risultati_modello = esegui_modello(
        scelta_modello=scelta_modello,
        train_values=train_values,
        train_labels=train_labels["damage_grade"],
        selected_features=selected_features,
        metodo_fs=metodo_fs,
    )

    # ── FASE 5: Risultati ──────────────────────────────────
    presenta_risultati(
        scelta_modello=scelta_modello,
        scelta_outlier=scelta_outlier,
        strategia_imputazione=STRATEGIE_IMPUTAZIONE.get(scelta_imputazione, "Sconosciuta"),
        tipo_geo_embedding=tipo_geo_embedding,
        metodo_fs=metodo_fs,
        risultati_modello=risultati_modello,
    )
    # Segna che abbiamo raggiunto la Fase 5 (pipeline completata)
    pipeline_completed = True

    # Salva le performance finali SOLO se la pipeline è arrivata alla Fase 5.
    if pipeline_completed:
        temp_json = None
        try:
            out_dir = os.path.join("outputs", "performances")
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, "performances.csv")
            temp_json = os.path.join(out_dir, "record_tmp.json")

            # Prova a ricavare una etichetta leggibile per la strategia di imputazione
            try:
                from src.preprocessing.imputation_strategies import STRATEGIE_IMPUTAZIONE as STRATEGIE_IMPUTAZIONE_REGISTRY
            except Exception:
                STRATEGIE_IMPUTAZIONE_REGISTRY = {}

            strategia_usata = None
            if imputation_reports:
                strategia_usata = next(iter(imputation_reports.values()))["strategia"]

            strategia_readable = strategia_usata
            if isinstance(strategia_usata, str):
                if strategia_usata in STRATEGIE_IMPUTAZIONE_REGISTRY:
                    strategia_readable = STRATEGIE_IMPUTAZIONE_REGISTRY[strategia_usata].nome_menu
                else:
                    for strat in STRATEGIE_IMPUTAZIONE_REGISTRY.values():
                        if getattr(strat, "nome_report", None) == strategia_usata:
                            strategia_readable = getattr(strat, "nome_menu")
                            break

            record = _build_performance_record(
                scelta_modello=scelta_modello,
                scelta_outlier=scelta_outlier,
                strategia_imputazione=strategia_readable,
                tipo_geo_embedding=tipo_geo_embedding,
                metodo_fs=metodo_fs,
                modalita_fs=modalita_fs,
                risultati_modello=risultati_modello,
            )

            # Scrivi record temporaneo (atomicità: se l'utente interrompe, non eseguiamo l'append finale)
            with open(temp_json, "w", encoding="utf-8") as tf:
                json.dump(record, tf, ensure_ascii=False)

            # Append al CSV con schema leggibile e colonne stabili
            _append_performance_record(out_file, record)

            # Rimuovi temporaneo
            try:
                os.remove(temp_json)
            except Exception:
                pass

            print(f"  ✓ Performance salvate in: {out_file}")

        except KeyboardInterrupt:
            # Utente ha premuto Ctrl+C: assicurati di non lasciare artefatti
            try:
                if temp_json and os.path.exists(temp_json):
                    os.remove(temp_json)
            except Exception:
                pass
            print("  ⚠️ Salvataggio annullato: interruzione utente (Ctrl+C)")
        except Exception as e:
            print(f"  ⚠️ Impossibile salvare le performance: {e}")

    # ── RIEPILOGO FINALE ───────────────────────────────────
    _banner("PIPELINE COMPLETATA")
    print(f"  ✅ Outlier detection:       {'IQR' if scelta_outlier == '1' else 'DBSCAN'}")
    if imputation_reports:
        strategia_usata = next(iter(imputation_reports.values()))["strategia"]
        # Proviamo a ottenere una etichetta leggibile dalla registry delle strategie
        try:
            from src.preprocessing.imputation_strategies import STRATEGIE_IMPUTAZIONE as STRATEGIE_IMPUTAZIONE_REGISTRY
        except Exception:
            STRATEGIE_IMPUTAZIONE_REGISTRY = {}

        strategia_readable = strategia_usata
        if isinstance(strategia_usata, str):
            if strategia_usata in STRATEGIE_IMPUTAZIONE_REGISTRY:
                strategia_readable = STRATEGIE_IMPUTAZIONE_REGISTRY[strategia_usata].nome_menu
            else:
                for strat in STRATEGIE_IMPUTAZIONE_REGISTRY.values():
                    if getattr(strat, 'nome_report', None) == strategia_usata:
                        strategia_readable = getattr(strat, 'nome_menu')
                        break

        print(f"  ✅ Strategia imputazione:   {strategia_readable}")
        print(f"     Colonne imputate:       {list(imputation_reports.keys())}")
    else:
        print("  ✅ Nessuna colonna numerica da imputare")
    
    print(f"  ✅ Geo embedding:           {'Embedding statico' if tipo_geo_embedding == 'embedding' else f'Rete neurale ({n_hidden} layer)'}")
    print(f"  ✅ Feature selection:       {metodo_fs.upper()}")
    print(f"     Feature usate:           {len(selected_features)}")
    
    nomi_modelli = {
        "1": "KNN",
        "2": "Albero Decisionale",
        "3": "SVM",
        "4": "Random Forest",
    }
    print(f"  ✅ Modello selezionato:     {nomi_modelli.get(scelta_modello, '?')}")

    return (
        train_values,
        train_labels,
        test_values,
        report,
        train_quality_report,
        test_quality_report,
        validation_report_train,
        validation_report_test,
    )


if __name__ == "__main__":
    risultato = main()
    if risultato is not None:
        (
            train_values,
            train_labels,
            test_values,
            report,
            train_quality_report,
            test_quality_report,
            validation_report_train,
            validation_report_test,
        ) = risultato
