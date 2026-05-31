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
import threading
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

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

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
N_JOBS = 1                     # disabilita parallelizzazione (evita deadlock su Windows)


# ---------------------------------------------------------------------------
# Classe Spinner per mostrare il progresso in tempo reale
# ---------------------------------------------------------------------------

class ProgressSpinner:
    """Spinner ASCII che mostra il progresso durante il fitting di GridSearchCV."""
    
    SPINNERS = ["|", "/", "-", "\\"]
    
    def __init__(self, total_fits, verbose=True):
        self.total_fits = total_fits
        self.verbose = verbose
        self.running = False
        self.thread = None
        self.start_time = time.time()
        self.spinner_idx = 0
    
    def start(self):
        """Avvia lo spinner in un thread separato."""
        if not self.verbose:
            return
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Ferma lo spinner."""
        if not self.verbose:
            return
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        # Pulisci la riga dello spinner
        try:
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()
        except:
            pass
    
    def _spin(self):
        """Loop dello spinner che gira mentre il fitting è in corso."""
        while self.running:
            try:
                elapsed = time.time() - self.start_time
                spinner_char = self.SPINNERS[self.spinner_idx % len(self.SPINNERS)]
                msg = f"  {spinner_char} Tuning in corso... ({elapsed:.0f}s)"
                sys.stdout.write("\r" + msg)
                sys.stdout.flush()
                self.spinner_idx += 1
                time.sleep(0.5)
            except:
                # Ignora qualsiasi errore nel thread dello spinner
                pass


# ---------------------------------------------------------------------------
# Callback per monitorare il progresso di GridSearchCV
# ---------------------------------------------------------------------------
class GridSearchLogger:
    """Logger personalizzato per GridSearchCV con output in italiano."""
    
    # Traduzione dei nomi dei parametri
    PARAM_NAMES_IT = {
        "n_estimators": "N. Alberi",
        "max_depth": "Prof. Massima",
        "min_samples_split": "Camp. Min Split",
        "min_samples_leaf": "Camp. Min Foglia",
        "max_features": "Feature Per Split",
        "n_neighbors": "N. Vicini",
        "weights": "Pesi",
        "metric": "Metrica Distanza",
        "max_depth": "Profondità Max",
    }
    
    PARAM_DESCRIPTIONS = {
        "n_estimators": "numero di alberi decisionali nella foresta",
        "max_depth": "profondità massima di ogni albero",
        "min_samples_split": "campioni minimi per dividere un nodo",
        "min_samples_leaf": "campioni minimi in una foglia",
        "max_features": "numero di feature considerate per ogni split",
        "n_neighbors": "numero di vicini da considerare",
        "weights": "come pesare i vicini (uniforme o per distanza)",
        "metric": "metrica di distanza (euclidea, manhattan, etc.)",
    }
    
    def __init__(self, total_fits, n_params, verbose=True):
        self.total_fits = total_fits
        self.n_params = n_params
        self.verbose = verbose
        self.fit_count = 0
        self.best_score = 0.0
        self.best_params = None
        self.start_time = time.time()
        self.pbar = None
        
        if verbose and TQDM_AVAILABLE:
            self.pbar = tqdm(
                total=total_fits,
                desc="⏳ Tuning Iperparametri",
                unit=" fit",
                leave=True,
                position=0,
                bar_format='{desc} |{bar}| {n_fmt}/{total_fmt} [{percentage:.0f}%] {postfix}'
            )
    
    def _format_params_italian(self, params_dict: dict) -> str:
        """Formatta i parametri in italiano con descrizioni."""
        output = []
        for k, v in params_dict.items():
            k_clean = k.replace("clf__", "").replace("scaler__", "")
            k_it = self.PARAM_NAMES_IT.get(k_clean, k_clean)
            output.append(f"{k_it}={v}")
        return " • ".join(output)
    
    def finalize(self, best_score, best_params_dict):
        """Finalizza il logger e stampa i risultati in italiano."""
        self.best_score = best_score
        self.best_params = best_params_dict
        
        if self.pbar:
            self.pbar.close()
        
        elapsed = time.time() - self.start_time
        
        if self.verbose:
            print(f"\n  {'─' * 76}")
            print(f"  RIEPILOGO TUNING IPERPARAMETRI")
            print(f"  {'─' * 76}")
            print(f"  Tempo totale: {elapsed:.1f}s")
            print(f"  Miglior score CV: {best_score:.4f}")
            print(f"  Parametri migliori:")
            sys.stdout.flush()
            
            for k, v in best_params_dict.items():
                k_clean = k.replace("clf__", "").replace("scaler__", "")
                k_it = self.PARAM_NAMES_IT.get(k_clean, k_clean)
                desc = self.PARAM_DESCRIPTIONS.get(k_clean, "")
                
                # Formatta il valore
                if isinstance(v, float):
                    v_str = f"{v:.4f}" if v < 1 else f"{v:.0f}"
                else:
                    v_str = str(v)
                
                if desc:
                    print(f"    {k_it} = {v_str}  ({desc})")
                else:
                    print(f"    {k_it} = {v_str}")
            sys.stdout.flush()




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
    #                         Range empirico [50, 150, 300, 500] per trovare plateauing della
    #                         performance e evitare overfitting oltre 300.
    #   - max_depth          : profondita' massima di ciascun albero
    #   - min_samples_split  : campioni minimi per dividere un nodo
    #   - min_samples_leaf   : campioni minimi in una foglia
    #   - max_features       : feature considerate per ciascun split
    configs.append({
        "name": "Random Forest",
        "pipeline": Pipeline([
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "param_grid": {
            "clf__n_estimators": [50, 150, 300, 500],
            "clf__max_depth": [10, 20],
            "clf__min_samples_split": [5, 10],
            "clf__min_samples_leaf": [2, 4],
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
            "clf__max_depth": [10, 20, 30],
            "clf__min_samples_split": [5, 10],
            "clf__min_samples_leaf": [2, 4],
            "clf__max_features": ["sqrt", "log2"],
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
    DataPreprocessed/preprocessed/train_features_labels_preprocessed.csv
    nella root di progetto.
    """
    if data_path is None:
        data_path = PROJECT_ROOT / "Data" / "preprocessed" / "train_features_labels_preprocessed.csv"
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

    risultati = []

    for cfg in configs:
        name = cfg["name"]
        pipeline = cfg["pipeline"]
        param_grid = cfg["param_grid"]

        # Calcolo combinazioni totali
        n_combinazioni = 1
        for values in param_grid.values():
            n_combinazioni *= len(values)
        
        total_fits = n_combinazioni * CV_FOLDS

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"  {name}")
            print(f"  Combinazioni da valutare: {n_combinazioni} x {CV_FOLDS} fold = "
                  f"{total_fits} fit totali")
            print(f"{'=' * 80}")
            sys.stdout.flush()

        start = time.time()

        # Logger personalizzato per grid search
        logger = GridSearchLogger(total_fits, n_combinazioni, verbose=verbose)

        if verbose:
            print(f"  Inizio tuning... (visualizza progressi candidato per candidato)\n")
            sys.stdout.flush()

        # Crea lo spinner per mostrare il progresso
        spinner = ProgressSpinner(total_fits, verbose=verbose)
        spinner.start()

        # Crea la GridSearchCV con verbose=0 (lo spinner mostrerà il progresso)
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            scoring=SCORING,
            n_jobs=N_JOBS,
            refit=True,
            verbose=0,  # Niente messaggi [CV] - usa solo lo spinner
            error_score="raise",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid.fit(X_train, y_train)
        
        # Ferma lo spinner
        spinner.stop()
        
        if verbose:
            print("\n  ✓ Tuning completato")
            sys.stdout.flush()

        tempo = time.time() - start
        
        # Finalizza il logger con i risultati
        logger.finalize(grid.best_score_, grid.best_params_)

        # Predizione sul validation set
        if verbose:
            print(f"\n  Predizione sul validation set...")
            sys.stdout.flush()
        y_pred = grid.best_estimator_.predict(X_val)
        f1_val = f1_score(y_val, y_pred, average="micro")
        acc_val = accuracy_score(y_val, y_pred)

        # Parametri migliori (senza prefisso pipeline)
        best_params_clean = {
            k.replace("clf__", "").replace("scaler__", ""): v
            for k, v in grid.best_params_.items()
        }

        if verbose:
            print(f"\n{'─' * 80}")
            print(f"  ✅ RISULTATI FINALI")
            print(f"{'─' * 80}")
            print(f"  Tempo totale: {tempo:.1f}s")
            print(f"  Miglior score CV ({SCORING}): {grid.best_score_:.4f}")
            print(f"  Score Validation (F1-micro): {f1_val:.4f}")
            print(f"  Accuracy Validation: {acc_val:.4f}")
            print(f"\n  Parametri migliori:")
            for param, value in best_params_clean.items():
                print(f"    • {param}: {value}")

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
