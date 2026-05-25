"""
Modello Decision Tree per il dataset Terremoto Nepal con Hyperparameter Tuning.

Utilizzo come funzione importabile:
    from src.Modelli.decisiontree import train_decisiontree
    results = train_decisiontree(X_train, y_train, X_val, y_val)
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_selection.Hyperparameter_Tuning import esegui_grid_search, get_dt_config


def validate_numeric_features(X: pd.DataFrame) -> list[str]:
    """Controlla che tutte le feature passate al Decision Tree siano numeriche."""
    feature_columns = X.columns.tolist()
    if not feature_columns:
        raise ValueError("Nessuna feature disponibile per addestrare il Decision Tree.")

    non_numeric = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    if non_numeric:
        raise ValueError(f"Decision Tree richiede feature numeriche. Colonne non numeriche: {non_numeric}")

    return feature_columns


def train_decisiontree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    verbose: bool = True,
) -> dict:
    """
    Addestra un modello Decision Tree con hyperparameter tuning su dati preprocessati.

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
        print("MODELLO DECISION TREE")
        print(f"{'='*80}")

    # Validazione
    feature_columns = validate_numeric_features(X_train)

    # Ottieni configurazione Decision Tree
    dt_configs = get_dt_config()
    if not dt_configs:
        raise ValueError("Impossibile caricare la configurazione Decision Tree.")

    # Esegui grid search
    risultati = esegui_grid_search(X_train, y_train, X_val, y_val, configs=dt_configs, verbose=verbose)

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
        print("RIEPILOGO FINALE DECISION TREE")
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
