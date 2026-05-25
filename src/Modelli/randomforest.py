"""
Modello Random Forest per il dataset Terremoto Nepal con Hyperparameter Tuning.

Utilizzo come funzione importabile:
    from src.Modelli.randomforest import train_randomforest
    results = train_randomforest(X_train, y_train, X_val, y_val)
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_selection.Hyperparameter_Tuning import esegui_grid_search, get_rf_config


def validate_numeric_features(X: pd.DataFrame) -> list[str]:
    """Controlla che tutte le feature passate al Random Forest siano numeriche."""
    feature_columns = X.columns.tolist()
    if not feature_columns:
        raise ValueError("Nessuna feature disponibile per addestrare il Random Forest.")

    non_numeric = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    if non_numeric:
        raise ValueError(f"Random Forest richiede feature numeriche. Colonne non numeriche: {non_numeric}")

    return feature_columns


def train_randomforest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    verbose: bool = True,
) -> dict:
    """
    Addestra un modello Random Forest con hyperparameter tuning su dati preprocessati.

    Args:
        X_train: feature di training (DataFrame)
        y_train: target di training (Series)
        X_val: feature di validation (DataFrame)
        y_val: target di validation (Series)
        verbose: stampe a console

    Returns:
        dict con chiavi:
        - "model": modello addestrato
        - "best_params": iperparametri migliori
        - "metrics": metriche di valutazione
        - "n_features": numero di feature utilizzate
    """
    if verbose:
        print(f"\n{'='*80}")
        print("MODELLO RANDOM FOREST")
        print(f"{'='*80}")

    # Validazione
    feature_columns = validate_numeric_features(X_train)

    # Ottieni configurazione Random Forest
    rf_configs = get_rf_config()
    if not rf_configs:
        raise ValueError("Impossibile caricare la configurazione Random Forest.")

    # Esegui grid search
    risultati = esegui_grid_search(X_train, y_train, X_val, y_val, configs=rf_configs, verbose=verbose)

    if not risultati:
        raise ValueError("Grid search non ha prodotto risultati.")

    risultato = risultati[0]
    best_estimator = risultato["grid_search_obj"].best_estimator_

    metrics = {
        "f1_micro": risultato["F1_Micro_Val"],
        "accuracy": risultato["Accuracy_Val"],
        "f1_macro": 0.0,
        "balanced_accuracy": 0.0,
        "cv_best_score": risultato["F1_Micro_CV"],
        "tuning_seconds": risultato["Tempo_s"],
    }

    if verbose:
        print(f"\n{'='*80}")
        print("RIEPILOGO FINALE RANDOM FOREST")
        print(f"{'='*80}")
        print(f"Feature usate: {len(feature_columns)}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Micro: {metrics['f1_micro']:.4f}")

    return {
        "model": best_estimator,
        "best_params": risultato["Migliori_Iperparametri"],
        "metrics": metrics,
        "n_features": len(feature_columns),
    }
