"""
Modello Random Forest per il dataset Terremoto Nepal con Hyperparameter Tuning.

La pipeline usa:
1. imputazione mediana di sicurezza;
2. Random Forest con GridSearchCV per tuning degli iperparametri;
3. classificatore RandomForestClassifier.

Gli iperparametri tuned sono:
- n_estimators: numero di alberi nella foresta
- max_depth: profondità massima di ciascun albero
- min_samples_split: campioni minimi per dividere un nodo
- min_samples_leaf: campioni minimi in una foglia
- max_features: feature considerate per ciascun split

Le geo feature aggregate sono attive di default: gli ID geografici grezzi
vengono prima trasformati in statistiche supervisionate anti-leakage, poi
rimossi dalle feature finali per evitare bias.

Random Forest non richiede standardizzazione (tree-based), ma usa imputazione
mediana per robustezza.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.geo_features import GeoFeatureEngineer
from src.feature_selection.Hyperparameter_Tuning import esegui_grid_search, get_rf_config


# ---------------------------------------------------------------------------
# Costanti di progetto
# ---------------------------------------------------------------------------

TARGET_COL = "damage_grade"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "Data" / "preprocessed" / "train_features_labels_preprocessed.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modelli" / "randomforest"
DEFAULT_EXCLUDE_COLUMNS = ["building_id", TARGET_COL]
GEO_ID_COLUMNS = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    """Legge i parametri da riga di comando per rendere lo script riusabile."""
    parser = argparse.ArgumentParser(
        description="Addestra e valuta un Random Forest multiclasse con hyperparameter tuning."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="CSV con feature e damage_grade. Default: dataset preprocessato.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Cartella dove salvare metriche e feature usate.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=30000,
        help="Numero massimo di righe usate per il test rapido. Usa 0 per tutte.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Quota di validation holdout. Default: 0.2.",
    )
    parser.add_argument(
        "--no-geo-aggregate",
        action="store_true",
        help="Disattiva le geo feature aggregate e usa solo le feature preprocessate originali.",
    )
    parser.add_argument(
        "--use-raw-geo-ids",
        action="store_true",
        help="Mantiene anche gli ID geografici grezzi dopo la generazione delle geo aggregate.",
    )
    parser.add_argument(
        "--geo-smoothing",
        type=float,
        default=20.0,
        help="Smoothing per target mean/probabilita geografiche. Default: 20.0.",
    )
    parser.add_argument(
        "--geo-rare-threshold",
        type=int,
        default=10,
        help="Soglia sotto cui una zona geografica viene marcata come rara. Default: 10.",
    )
    parser.add_argument(
        "--geo-n-splits",
        type=int,
        default=5,
        help="Numero fold per geo encoding out-of-fold anti-leakage. Default: 5.",
    )
    return parser.parse_args()


def load_dataset(dataset_path: Path, max_rows: int) -> pd.DataFrame:
    """Carica il dataset e, se richiesto, usa un campione stratificato veloce."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset non trovato: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Colonna target mancante: {TARGET_COL}")

    if max_rows and max_rows > 0 and len(df) > max_rows:
        # Campioniamo mantenendo circa la stessa distribuzione delle classi.
        df = (
            df.groupby(TARGET_COL, group_keys=False)
            .sample(frac=max_rows / len(df), random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa feature e target rimuovendo ID tecnico e target."""
    X = df.drop(columns=DEFAULT_EXCLUDE_COLUMNS)
    y = df[TARGET_COL]
    return X, y


def add_geo_aggregate_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    use_geo_aggregate: bool = True,
    keep_raw_geo_ids: bool = False,
    smoothing: float = 20.0,
    rare_threshold: int = 10,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggiunge geo feature aggregate evitando leakage sul train.

    Sul train usa fit_transform_oof: ogni riga riceve statistiche calcolate
    senza usare il proprio target. Sulla validation usa solo statistiche apprese
    dal train. Questo rende il confronto corretto.
    """
    if not use_geo_aggregate:
        return _drop_raw_geo_ids_if_needed(X_train, keep_raw_geo_ids), _drop_raw_geo_ids_if_needed(
            X_val,
            keep_raw_geo_ids,
        )

    geo_engineer = GeoFeatureEngineer(
        geo_columns=tuple(GEO_ID_COLUMNS),
        target_col=TARGET_COL,
        smoothing=smoothing,
        rare_threshold=rare_threshold,
        n_splits=n_splits,
        random_state=RANDOM_STATE,
        append_original=True,
    )

    X_train_geo = geo_engineer.fit_transform_oof(X_train, y_train)
    X_val_geo = geo_engineer.transform(X_val)

    return _drop_raw_geo_ids_if_needed(X_train_geo, keep_raw_geo_ids), _drop_raw_geo_ids_if_needed(
        X_val_geo,
        keep_raw_geo_ids,
    )


def _drop_raw_geo_ids_if_needed(df: pd.DataFrame, keep_raw_geo_ids: bool) -> pd.DataFrame:
    """Rimuove gli ID geografici grezzi se si vogliono usare solo le aggregate."""
    if keep_raw_geo_ids:
        return df
    return df.drop(columns=[col for col in GEO_ID_COLUMNS if col in df.columns])


def validate_numeric_features(X: pd.DataFrame) -> list[str]:
    """Controlla che tutte le feature passate al Random Forest siano numeriche."""
    feature_columns = X.columns.tolist()
    if not feature_columns:
        raise ValueError("Nessuna feature disponibile per addestrare il Random Forest.")

    non_numeric = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    if non_numeric:
        raise ValueError(f"Random Forest richiede feature numeriche. Colonne non numeriche: {non_numeric}")

    return feature_columns


def run_hyperparameter_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    verbose: bool = True,
) -> dict:
    """Esegue il grid search sui parametri Random Forest usando il modulo Hyperparameter_Tuning.

    Ritorna un dizionario con:
    - best_estimator: il modello addestrato con i migliori iperparametri
    - best_params: dizionario dei migliori iperparametri
    - results: metriche di valutazione
    """
    # Ottieni la configurazione Random Forest dal modulo di hyperparameter tuning
    rf_configs = get_rf_config()
    if not rf_configs:
        raise ValueError("Impossibile caricare la configurazione Random Forest.")

    if verbose:
        print(f"\n{'='*80}")
        print("HYPERPARAMETER TUNING - RANDOM FOREST")
        print(f"{'='*80}")

    # Esegui grid search usando la funzione del modulo Hyperparameter_Tuning
    risultati = esegui_grid_search(X_train, y_train, X_val, y_val, configs=rf_configs, verbose=verbose)

    if not risultati:
        raise ValueError("Grid search non ha prodotto risultati.")

    # Estrai i risultati
    risultato = risultati[0]
    best_estimator = risultato["grid_search_obj"].best_estimator_

    return {
        "best_estimator": best_estimator,
        "best_params": risultato["Migliori_Iperparametri"],
        "results": {
            "f1_micro": risultato["F1_Micro_Val"],
            "accuracy": risultato["Accuracy_Val"],
            "f1_macro": 0.0,  # Non calcolato da esegui_grid_search
            "balanced_accuracy": 0.0,  # Non calcolato da esegui_grid_search
            "cv_best_score": risultato["F1_Micro_CV"],
            "tuning_seconds": risultato["Tempo_s"],
        },
    }





def save_outputs(output_dir: Path, tuning_result: dict, feature_columns: list[str], metrics: dict) -> None:
    """Placeholder - non salva output."""
    pass


def main() -> dict:
    """Esegue il workflow completo: caricamento dati, tuning, training e report."""
    args = parse_args()

    print("\n" + "=" * 80)
    print("MODELLO RANDOM FOREST")
    print("=" * 80)

    df = load_dataset(args.dataset, max_rows=args.max_rows)
    X, y = split_features_target(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_val = add_geo_aggregate_features(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        use_geo_aggregate=not args.no_geo_aggregate,
        keep_raw_geo_ids=args.use_raw_geo_ids,
        smoothing=args.geo_smoothing,
        rare_threshold=args.geo_rare_threshold,
        n_splits=args.geo_n_splits,
    )
    feature_columns = validate_numeric_features(X_train)

    # Esegui hyperparameter tuning
    tuning_result = run_hyperparameter_tuning(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        verbose=True,
    )

    # Compila il report finale
    metrics = {
        "dataset": str(args.dataset),
        "rows_used": int(len(df)),
        "n_features_input": int(len(feature_columns)),
        "test_size": float(args.test_size),
        "scoring_metric": "f1_micro",
        "geo_aggregate_enabled": not args.no_geo_aggregate,
        "raw_geo_ids_kept": bool(args.use_raw_geo_ids),
        "geo_smoothing": float(args.geo_smoothing),
        "geo_rare_threshold": int(args.geo_rare_threshold),
        "geo_n_splits": int(args.geo_n_splits),
        "best_params": tuning_result["best_params"],
        "scores": tuning_result["results"],
    }

    save_outputs(args.output_dir, tuning_result, feature_columns, metrics)

    print(f"\n{'='*80}")
    print("RIEPILOGO FINALE RANDOM FOREST")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Righe usate: {len(df)}")
    print(f"Feature usate: {len(feature_columns)}")
    print(f"Geo aggregate: {not args.no_geo_aggregate}")
    print(f"ID geo grezzi mantenuti: {args.use_raw_geo_ids}")

    return metrics


if __name__ == "__main__":
    main()
