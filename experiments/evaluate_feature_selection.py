"""
Benchmark rigoroso di Feature Selection con Hyperparameter Tuning integrato.

Questo script valuta diversi metodi di feature selection confrontando
le feature selezionate tramite il modulo Hyperparameter_Tuning con
GridSearchCV completa (Random Forest di default, oppure KNN + RF + DT
con --full-tuning).

Pipeline eseguita:
    1. Caricamento del dataset preprocessato (DataPreprocessed).
    2. Campionamento bilanciato per rendere equa la valutazione tra classi.
    3. Split stratificato train / validation (80 / 20).
    4. Per ciascun metodo di Feature Selection:
       a. Selezione delle feature secondo il metodo.
       b. Hyperparameter tuning sulle feature selezionate via modulo modulare.
    5. Report comparativo finale con classifica e top 3 metodi.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configurazione path di progetto
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Import moduli del progetto
# ---------------------------------------------------------------------------
from src.feature_selection.Hyperparameter_Tuning import (
    esegui_grid_search,
    get_all_configs,
    get_rf_config,
)
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
from src.preprocessing.data_selection import get_balanced_sample, get_stratified_sample

# ---------------------------------------------------------------------------
# Costanti globali (non modificabili via CLI)
# ---------------------------------------------------------------------------
TARGET_COL = "damage_grade"
EXCLUDE_COLS = ["building_id", "geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
RANDOM_STATE = 42

# Valori di default per i parametri CLI
DEFAULT_SAMPLE_MODE = "balanced"
DEFAULT_MAX_PER_CLASS = 10000
DEFAULT_N_SAMPLES = 20000
DEFAULT_MAX_FEATURES = 15
DEFAULT_MAX_ROWS_SUBSET = 2000
DEFAULT_TEST_SIZE = 0.20


# ---------------------------------------------------------------------------
# Parsing argomenti CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Analizza gli argomenti da riga di comando."""
    parser = argparse.ArgumentParser(
        description="Benchmark Feature Selection con Hyperparameter Tuning integrato."
    )
    parser.add_argument(
        "--sample-mode",
        choices=["balanced", "stratified"],
        default=DEFAULT_SAMPLE_MODE,
        help="Tipo di campionamento: 'balanced' (ugual numero per classe) o "
             "'stratified' (proporzioni originali). Default: balanced.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=DEFAULT_MAX_PER_CLASS,
        help="Campioni massimi per classe con sample-mode=balanced. Default: 10000.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Campioni totali con sample-mode=stratified. Default: 20000.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=DEFAULT_MAX_FEATURES,
        help="Numero massimo di feature da selezionare per i ranking. Default: 15.",
    )
    parser.add_argument(
        "--max-rows-subset",
        type=int,
        default=DEFAULT_MAX_ROWS_SUBSET,
        help="Righe massime per i subset selector (SBS, BestFirst). Default: 2000.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Quota di validation set. Default: 0.20.",
    )
    parser.add_argument(
        "--full-tuning",
        action="store_true",
        help="Usa KNN + Random Forest + Decision Tree (piu' lento). Default: solo Random Forest.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Valutazione feature con Hyperparameter Tuning modulare
# ---------------------------------------------------------------------------

def evaluate_features_with_tuning(X_train, y_train, X_val, y_val,
                                  features, full_tuning=False):
    """
    Valuta un set di feature tramite il modulo di Hyperparameter Tuning.

    Sostituisce il vecchio grid search inline con una chiamata al modulo
    modulare Hyperparameter_Tuning.py che esegue GridSearchCV completa.

    Args:
        X_train, y_train: dati di training.
        X_val, y_val: dati di validazione.
        features: lista di nomi delle feature da valutare.
        full_tuning: se True usa KNN + RF + DT, altrimenti solo Random Forest.

    Returns:
        best_result: dizionario con il miglior risultato (o None).
        all_results: lista completa dei risultati per ogni algoritmo.
    """
    if not features:
        return None, []

    valid_features = [f for f in features if f in X_train.columns]
    if not valid_features:
        return None, []

    X_tr_sub = X_train[valid_features]
    X_v_sub = X_val[valid_features]

    configs = get_all_configs() if full_tuning else get_rf_config()
    risultati = esegui_grid_search(
        X_tr_sub, y_train, X_v_sub, y_val, configs, verbose=False
    )

    best = max(risultati, key=lambda r: r["F1_Micro_Val"])
    return best, risultati


# ---------------------------------------------------------------------------
# Definizione metodi di Feature Selection
# ---------------------------------------------------------------------------

def _get_fs_methods():
    """Restituisce la lista dei metodi di feature selection da valutare."""
    return [
        ("PCA (Elbow Method)", PCAHandler, {}),
        ("Lasso Embedded", LassoFeatureSelector, {}),
        ("Pairwise Correlation", PairwiseCorrelationRanker, {}),
        ("Relief", ReliefRanker, {}),
        ("Information Gain", InformationGainRanker, {}),
        ("Sequential Forward Selection", SequentialForwardSelector,
         {"estimator_name": "knn", "scoring": "f1_micro"}),
        ("Sequential Backward Selection", SequentialBackwardSelector,
         {"estimator_name": "knn", "scoring": "f1_micro"}),
        ("Bidirectional Subset Selection", StepwiseBidirectionalSelector,
         {"estimator_name": "knn", "scoring": "f1_micro"}),
        ("Max-Min Subset Selection", MaxMinSubsetSelector, {}),
        ("Best First Search", BestFirstSelector, {}),
    ]


# ---------------------------------------------------------------------------
# Logica di selezione feature per ciascun metodo
# ---------------------------------------------------------------------------

def _select_features(name, MethodClass, kwargs, X_train, y_train, df_train,
                     max_features=DEFAULT_MAX_FEATURES,
                     max_rows_subset=DEFAULT_MAX_ROWS_SUBSET):
    """
    Esegue la fase di selezione delle feature per un singolo metodo FS.

    Returns:
        selected_features: lista di nomi feature selezionate.
        pca_data: tuple (X_train_pca, X_val_pca) se PCA, altrimenti None.
    """
    if name == "Information Gain":
        model = MethodClass(log_base=2)
    else:
        model = MethodClass(**kwargs)

    # ── Lasso Embedded ─────────────────────────────────────────────────
    if name == "Lasso Embedded":
        res = model.select(X_train, y_train, alpha=0.002)
        return res["selected_features"]["feature"].head(max_features).tolist(), None

    # ── PCA (Elbow Method) ─────────────────────────────────────────────
    if name == "PCA (Elbow Method)":
        model.fit(df_train, exclude_columns=[TARGET_COL])
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
        selected = [f"PC{i}" for i in range(1, elbow_k + 1)]

        # Restituiamo i dati trasformati per la valutazione successiva
        X_train_pca = model.transform(df_train).iloc[:, :elbow_k]
        return selected, (X_train_pca, model)

    # ── Ranking methods ────────────────────────────────────────────────
    if "Ranker" in MethodClass.__name__:
        res = model.rank(df_train, label_column=TARGET_COL)
        if name == "Pairwise Correlation":
            return res["supervised_ranking"]["feature"].head(max_features).tolist(), None
        elif name == "Relief":
            return res["relief_ranking"]["feature"].head(max_features).tolist(), None
        elif name == "Information Gain":
            return res["information_gain_ranking"]["feature"].head(max_features).tolist(), None

    # ── Subset selectors ───────────────────────────────────────────────
    if name == "Sequential Backward Selection":
        res = model.select(X_train, y_train.to_numpy(),
                           min_features=max_features, max_rows=max_rows_subset)
    elif name == "Max-Min Subset Selection":
        res = model.select(X_train, y_train, max_features=max_features)
    elif name == "Best First Search":
        y_arr = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train
        res = model.select(X_train, y_arr, max_rows=max_rows_subset)
    else:
        res = model.select(X_train, y_train.to_numpy(),
                           max_features=max_features, max_rows=max_rows_subset)

    return res["selected_features"]["selected_feature"].tolist(), None


# ---------------------------------------------------------------------------
# Stampa risultato singolo metodo
# ---------------------------------------------------------------------------

def _print_method_result(best, exec_time, n_features):
    """Stampa il riepilogo del risultato per un singolo metodo FS."""
    print(f"    Completato in {exec_time:.2f}s")
    print(f"    Algoritmo migliore : {best['Algoritmo']}")
    print(f"    F1-Micro Val       : {best['F1_Micro_Val']:.4f}")
    print(f"    F1-Micro CV        : {best['F1_Micro_CV']:.4f}")
    print(f"    Accuracy Val       : {best['Accuracy_Val']:.4f}")
    print(f"    Iperparametri      : {best['Migliori_Iperparametri']}")
    print(f"    N. Feature         : {n_features}")


# ---------------------------------------------------------------------------
# Entry point principale
# ---------------------------------------------------------------------------

def run_evaluation(args=None):
    """
    Esegue il benchmark completo di tutti i metodi di Feature Selection.

    Args:
        args: namespace con i parametri CLI (da parse_args()). Se None,
              usa i valori di default.
    """
    if args is None:
        args = parse_args()

    full_tuning = args.full_tuning
    tuning_mode = "COMPLETO (KNN + RF + DT)" if full_tuning else "Random Forest"
    sample_label = (
        f"bilanciato (max {args.max_per_class}/classe)"
        if args.sample_mode == "balanced"
        else f"stratificato ({args.n_samples} totali)"
    )

    print("=" * 80)
    print("BENCHMARK RIGOROSO DI FEATURE SELECTION")
    print(f"Hyperparameter Tuning : {tuning_mode}")
    print(f"Campionamento         : {sample_label}")
    print(f"Max feature ranking   : {args.max_features}")
    print(f"Max righe subset sel. : {args.max_rows_subset}")
    print(f"Test size             : {args.test_size}")
    print("=" * 80)

    # ── 1. Caricamento dataset ─────────────────────────────────────────
    print("\n1. Caricamento dataset preprocessato...")
    data_path = project_root / "Data" / "preprocessed" / "train_features_labels_preprocessed.csv"

    if not data_path.exists():
        print(f"   ERRORE: File non trovato: {data_path}")
        print("   Esegui prima la pipeline di preprocessing (main.py).")
        return

    df = pd.read_csv(data_path)
    print(f"   Dataset caricato: {df.shape[0]} righe x {df.shape[1]} colonne")

    # ── 2. Campionamento ───────────────────────────────────────────────
    print(f"\n2. Campionamento ({args.sample_mode})...")
    if args.sample_mode == "balanced":
        df_sampled = get_balanced_sample(
            df, TARGET_COL, max_per_class=args.max_per_class
        )
    else:
        df_sampled = get_stratified_sample(
            df, TARGET_COL, n_samples=args.n_samples
        )
    print(f"   Dataset campionato: {df_sampled.shape[0]} righe")
    print(f"   Distribuzione classi:\n{df_sampled[TARGET_COL].value_counts().to_string()}")

    # ── 3. Preparazione feature e target ───────────────────────────────
    print("\n3. Preparazione feature e target...")
    exclude = [c for c in EXCLUDE_COLS if c in df_sampled.columns]
    X = df_sampled.drop(columns=[TARGET_COL] + exclude)

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=False, dtype=float)

    y = df_sampled[TARGET_COL].astype(int)
    print(f"   Feature totali: {X.shape[1]}")

    # ── 4. Split stratificato ──────────────────────────────────────────
    print("\n4. Split stratificato train / validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Train: {X_train.shape[0]} righe | Validation: {X_val.shape[0]} righe")

    df_train = X_train.copy()
    df_train[TARGET_COL] = y_train

    # ── 5. Valutazione metodi FS ───────────────────────────────────────
    methods = _get_fs_methods()
    results = []

    print(f"\n5. Inizio valutazione dei {len(methods)} metodi di Feature Selection...")

    for name, MethodClass, kwargs in methods:
        print(f"\n{'-' * 80}")
        print(f"  > {name}")
        print(f"{'-' * 80}")

        start_time = time.time()

        try:
            # Fase A: selezione feature
            selected_features, pca_data = _select_features(
                name, MethodClass, kwargs, X_train, y_train, df_train,
                max_features=args.max_features,
                max_rows_subset=args.max_rows_subset,
            )

            if not selected_features:
                print("    Nessuna feature selezionata - metodo saltato.")
                continue

            # Fase B: hyperparameter tuning sulle feature selezionate
            if pca_data is not None:
                # Caso PCA: usiamo i dati trasformati
                X_train_pca, pca_model = pca_data
                X_val_pca = pca_model.transform(X_val).iloc[:, :len(selected_features)]

                configs = get_all_configs() if full_tuning else get_rf_config()
                tuning_results = esegui_grid_search(
                    X_train_pca, y_train, X_val_pca, y_val, configs, verbose=False
                )
                best = max(tuning_results, key=lambda r: r["F1_Micro_Val"])
            else:
                # Caso standard: usiamo le feature originali
                best, tuning_results = evaluate_features_with_tuning(
                    X_train, y_train, X_val, y_val, selected_features,
                    full_tuning=full_tuning
                )

            exec_time = time.time() - start_time

            if best is None:
                print("    Nessuna feature valida trovata per la valutazione.")
                continue

            _print_method_result(best, exec_time, len(selected_features))

            results.append({
                "Method": name,
                "Algoritmo": best["Algoritmo"],
                "F1_Micro_CV": best["F1_Micro_CV"],
                "F1_Micro_Val": best["F1_Micro_Val"],
                "Accuracy_Val": best["Accuracy_Val"],
                "Best_Params": str(best["Migliori_Iperparametri"]),
                "Time_s": round(exec_time, 2),
                "N_Features": len(selected_features),
                "Selected_Features": selected_features,
            })

        except Exception as e:
            print(f"    ERRORE in {name}: {e}")
            traceback.print_exc()

    # ── 6. Report finale ───────────────────────────────────────────────
    if not results:
        print("\nNessun risultato ottenuto.")
        return

    results_df = (
        pd.DataFrame(results)
        .sort_values("F1_Micro_Val", ascending=False)
        .reset_index(drop=True)
    )

    display_cols = [
        "Method", "Algoritmo", "F1_Micro_CV", "F1_Micro_Val",
        "Accuracy_Val", "Time_s", "N_Features",
    ]

    print("\n" + "=" * 80)
    print("CLASSIFICA FINALE - BENCHMARK FEATURE SELECTION")
    print(f"Hyperparameter Tuning: {tuning_mode}")
    print("=" * 80)
    print(results_df[display_cols].to_string(index=False))

    print("\n\nI TOP 3 METODI DI FEATURE SELECTION:")
    for i in range(min(3, len(results_df))):
        row = results_df.iloc[i]
        features_preview = row["Selected_Features"][:5]
        suffix = "..." if len(row["Selected_Features"]) > 5 else ""
        print(f"\n  {i + 1}. {row['Method']}")
        print(f"     Valutato con: {row['Algoritmo']}")
        print(f"     F1-Micro Val: {row['F1_Micro_Val']:.4f} | CV: {row['F1_Micro_CV']:.4f}")
        print(f"     Accuracy: {row['Accuracy_Val']:.4f}")
        print(f"     Iperparametri: {row['Best_Params']}")
        print(f"     Feature ({row['N_Features']}): {features_preview}{suffix}")

    # ── 7. Salvataggio ─────────────────────────────────────────────────
    output_path = project_root / "experiments" / "feature_selection_benchmark_rigorous.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nRisultati completi salvati in: {output_path}")

    return results_df


if __name__ == "__main__":
    run_evaluation(parse_args())
