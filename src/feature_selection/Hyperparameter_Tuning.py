"""
Hyperparameter Tuning tramite Grid Search per KNN, Decision Tree e Random Forest.

Questo modulo confronta due classificatori cercando per ognuno la combinazione
ottimale di iperparametri tramite GridSearchCV con cross-validation stratificata.

Pipeline eseguita:
    1. Caricamento del dataset preprocessato (DataPreprocessed).
    2. Campionamento bilanciato per rendere equa la valutazione tra classi.
    3. Split stratificato train / validation (80 / 20).
    4. Grid Search con 5-fold CV stratificata per ogni algoritmo.
    5. Valutazione sul validation set con il modello ottimale trovato.
    6. Report comparativo con F1-micro, accuracy, tempo di esecuzione e
       migliori iperparametri.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configurazione del path di progetto
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.data_selection import get_balanced_sample

# ---------------------------------------------------------------------------
# Configurazione globale
# ---------------------------------------------------------------------------
TARGET_COL = "damage_grade"
EXCLUDE_COLS = ["building_id", "geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
MAX_PER_CLASS = 10000          # campioni per classe nel bilanciamento
TEST_SIZE = 0.20               # percentuale di validation set
CV_FOLDS = 5                   # fold per la cross-validation
SCORING = "f1_micro"           # metrica primaria della Grid Search
RANDOM_STATE = 42
N_JOBS = -1                    # usa tutti i core disponibili


# ---------------------------------------------------------------------------
# Definizione delle griglie di iperparametri
# ---------------------------------------------------------------------------

def _get_algorithm_configs():
    """
    Restituisce la lista delle configurazioni per KNN, Decision Tree e Random Forest.

    Ogni elemento contiene:
        - name          : nome descrittivo dell'algoritmo
        - pipeline      : Pipeline sklearn
        - param_grid    : dizionario dei parametri da esplorare
    """
    configs = []

    # ── 1. K-Nearest Neighbors ─────────────────────────────────────────────
    #   - n_neighbors : numero di vicini (da pochi a molti per capire il trade-off)
    #   - weights     : 'uniform' (tutti uguali) vs 'distance' (peso inverso alla distanza)
    #   - metric      : tipo di distanza (Euclidea, Manhattan, Minkowski con p=3)
    configs.append({
        "name": "K-Nearest Neighbors (KNN)",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),          # scaling fondamentale per KNN
            ("clf", KNeighborsClassifier()),
        ]),
        "param_grid": {
            "clf__n_neighbors": [3, 5, 7, 9, 15, 21, 31],
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan", "minkowski"],
        },
    })

    # ── 2. Random Forest ───────────────────────────────────────────────────
    #   - n_estimators      : numero di alberi nella foresta
    #   - max_depth          : profondita' massima di ciascun albero
    #   - min_samples_split  : campioni minimi per dividere un nodo
    #   - min_samples_leaf   : campioni minimi in una foglia
    #   - max_features       : feature considerate per ciascun split
    configs.append({
        "name": "Random Forest",
        "pipeline": Pipeline([
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS)),
        ]),
        "param_grid": {
            "clf__n_estimators": [100, 200, 500],
            "clf__max_depth": [None, 10, 20, 30],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2"],
        },
    })

    # ── 3. Decision Tree ───────────────────────────────────────────────────
    #   - criterion          : funzione per misurare la qualità dello split
    #   - max_depth           : profondità massima dell'albero
    #   - min_samples_split   : campioni minimi per dividere un nodo
    #   - min_samples_leaf    : campioni minimi in una foglia
    #   - max_features        : feature considerate per ciascun split
    configs.append({
        "name": "Decision Tree",
        "pipeline": Pipeline([
            ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),
        "param_grid": {
            "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [None, 5, 10, 15, 20, 30],
            "clf__min_samples_split": [2, 5, 10, 20],
            "clf__min_samples_leaf": [1, 2, 4, 8],
            "clf__max_features": ["sqrt", "log2", None],
        },
    })

    return configs


def get_knn_config():
    """Restituisce solo la configurazione KNN per uso esterno rapido."""
    return [c for c in _get_algorithm_configs() if "KNN" in c["name"]]


def get_rf_config():
    """Restituisce solo la configurazione Random Forest per uso esterno."""
    return [c for c in _get_algorithm_configs() if "Random Forest" in c["name"]]


def get_dt_config():
    """Restituisce solo la configurazione Decision Tree per uso esterno."""
    return [c for c in _get_algorithm_configs() if "Decision Tree" in c["name"]]


def get_all_configs():
    """Restituisce tutte le configurazioni (KNN, Random Forest, Decision Tree)."""
    return _get_algorithm_configs()


# ---------------------------------------------------------------------------
# Funzioni principali
# ---------------------------------------------------------------------------

def carica_dataset(data_path=None):
    """
    Carica il dataset preprocessato.

    Se data_path e' None, cerca il file
    DataPreprocessed/train_features_labels_preprocessed.csv
    nella root di progetto.
    """
    if data_path is None:
        data_path = PROJECT_ROOT / "DataPreprocessed" / "train_features_labels_preprocessed.csv"
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset preprocessato non trovato: {data_path}\n"
            "Esegui prima la pipeline di preprocessing (main.py)."
        )

    df = pd.read_csv(data_path)
    print(f"Dataset caricato: {df.shape[0]} righe x {df.shape[1]} colonne")
    return df


def prepara_dati(df):
    """
    Prepara X e y dal DataFrame completo:
      - rimuove colonne da escludere e la colonna target
      - one-hot encode di eventuali colonne categoriche residue
    """
    exclude = [c for c in EXCLUDE_COLS if c in df.columns]
    X = df.drop(columns=[TARGET_COL] + exclude)

    # Encoding di sicurezza per colonne categoriche eventualmente rimaste
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=float)

    y = df[TARGET_COL].astype(int)
    return X, y


def esegui_grid_search(X_train, y_train, X_val, y_val, configs=None, verbose=True):
    """
    Per ciascun algoritmo in configs esegue GridSearchCV, addestra il
    modello ottimale e lo valuta sul validation set.

    Restituisce una lista di dizionari con i risultati.
    """
    if configs is None:
        configs = _get_algorithm_configs()

    cv_strategy = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    risultati = []

    for cfg in configs:
        name = cfg["name"]
        pipeline = cfg["pipeline"]
        param_grid = cfg["param_grid"]

        # Calcolo combinazioni totali
        n_combinazioni = 1
        for values in param_grid.values():
            n_combinazioni *= len(values)

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"  {name}")
            print(f"  Combinazioni da valutare: {n_combinazioni} x {CV_FOLDS} fold = "
                  f"{n_combinazioni * CV_FOLDS} fit totali")
            print(f"{'=' * 80}")

        start = time.time()

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring=SCORING,
            n_jobs=N_JOBS,
            refit=True,
            verbose=1 if verbose else 0,
            error_score="raise",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid.fit(X_train, y_train)

        tempo = time.time() - start

        # Predizione sul validation set
        y_pred = grid.best_estimator_.predict(X_val)
        f1_val = f1_score(y_val, y_pred, average="micro")
        acc_val = accuracy_score(y_val, y_pred)

        # Parametri migliori (senza prefisso pipeline)
        best_params_clean = {
            k.replace("clf__", "").replace("scaler__", ""): v
            for k, v in grid.best_params_.items()
        }

        if verbose:
            print(f"\n  Completato in {tempo:.1f}s")
            print(f"    Miglior score CV ({SCORING}): {grid.best_score_:.4f}")
            print(f"    Score Validation  (F1-micro): {f1_val:.4f}")
            print(f"    Accuracy Validation:          {acc_val:.4f}")
            print(f"    Migliori iperparametri:       {best_params_clean}")

        risultati.append({
            "Algoritmo": name,
            "F1_Micro_CV": round(grid.best_score_, 4),
            "F1_Micro_Val": round(f1_val, 4),
            "Accuracy_Val": round(acc_val, 4),
            "Tempo_s": round(tempo, 1),
            "Migliori_Iperparametri": best_params_clean,
            "grid_search_obj": grid,
        })

    return risultati


def stampa_report_finale(risultati, X_val, y_val):
    """Stampa il riepilogo comparativo e il classification report del vincitore."""

    risultati_ordinati = sorted(
        risultati, key=lambda r: r["F1_Micro_Val"], reverse=True
    )

    print("\n" + "=" * 80)
    print("CLASSIFICA FINALE - HYPERPARAMETER TUNING")
    print("=" * 80)

    df_report = pd.DataFrame([
        {k: v for k, v in r.items() if k != "grid_search_obj"}
        for r in risultati_ordinati
    ])
    print(df_report.to_string(index=False))

    # Classification report dettagliato del miglior modello
    best = risultati_ordinati[0]
    print(f"\n{'=' * 80}")
    print(f"DETTAGLIO MIGLIOR MODELLO: {best['Algoritmo']}")
    print(f"{'=' * 80}")
    y_pred_best = best["grid_search_obj"].best_estimator_.predict(X_val)
    print(classification_report(y_val, y_pred_best, digits=4))

    return risultati_ordinati


def salva_risultati(risultati, output_dir=None):
    """Salva i risultati in un CSV nella cartella experiments."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "experiments"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in risultati:
        row = {k: v for k, v in r.items() if k != "grid_search_obj"}
        row["Migliori_Iperparametri"] = str(row["Migliori_Iperparametri"])
        rows.append(row)

    df = pd.DataFrame(rows)
    out_file = output_dir / "hyperparameter_tuning_results.csv"
    df.to_csv(out_file, index=False)
    print(f"\nRisultati salvati in: {out_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("HYPERPARAMETER TUNING - GRID SEARCH (KNN + DECISION TREE + RANDOM FOREST)")
    print("=" * 80)

    # 1. Caricamento
    print("\n1. Caricamento dataset preprocessato...")
    df = carica_dataset()

    # 2. Campionamento bilanciato
    print("\n2. Campionamento bilanciato...")
    df_bal = get_balanced_sample(df, TARGET_COL, max_per_class=MAX_PER_CLASS)
    print(f"   Dataset bilanciato: {df_bal.shape[0]} righe")
    print(f"   Distribuzione classi:\n{df_bal[TARGET_COL].value_counts().to_string()}")

    # 3. Preparazione feature / target
    print("\n3. Preparazione feature e target...")
    X, y = prepara_dati(df_bal)
    print(f"   Feature totali: {X.shape[1]}")

    # 4. Split stratificato
    print("\n4. Split stratificato train / validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Train: {X_train.shape[0]} righe | Validation: {X_val.shape[0]} righe")

    # 5. Grid Search
    print("\n5. Avvio Grid Search...")
    risultati = esegui_grid_search(X_train, y_train, X_val, y_val)

    # 6. Report
    risultati_ordinati = stampa_report_finale(risultati, X_val, y_val)

    # 7. Salvataggio
    salva_risultati(risultati_ordinati)

    return risultati_ordinati


if __name__ == "__main__":
    main()
