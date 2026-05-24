"""
Modello SVM per il dataset Terremoto Nepal.

La pipeline usa:
1. imputazione mediana di sicurezza;
2. standardizzazione;
3. classificatore SVM.

Per la classificazione multiclasse, il default usa LinearSVC: in scikit-learn
questo corrisponde a una decomposizione One-vs-Rest / One-vs-All.
L'opzione SVC con kernel RBF resta disponibile come confronto e usa One-vs-One.

Le geo feature aggregate sono attive di default: gli ID geografici grezzi
vengono prima trasformati in statistiche supervisionate anti-leakage, poi
rimossi dalle feature finali per evitare che la SVM li interpreti come numeri
ordinati.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.geo_features import GeoFeatureEngineer


# ---------------------------------------------------------------------------
# Costanti di progetto
# ---------------------------------------------------------------------------

TARGET_COL = "damage_grade"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "Data" / "preprocessed" / "train_features_labels_preprocessed.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modelli" / "svm"
DEFAULT_EXCLUDE_COLUMNS = ["building_id", TARGET_COL]
GEO_ID_COLUMNS = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    """Legge i parametri da riga di comando per rendere lo script riusabile."""
    parser = argparse.ArgumentParser(description="Addestra e valuta una SVM multiclasse.")
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
        "--estimator",
        choices=["linear_svc", "svc_rbf"],
        default="linear_svc",
        help="Tipo di SVM. linear_svc e' piu' veloce, svc_rbf piu' costoso.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Parametro C della SVM. Default: 1.0.",
    )
    parser.add_argument(
        "--use-raw-geo-ids",
        action="store_true",
        help="Mantiene anche gli ID geografici grezzi dopo la generazione delle geo aggregate.",
    )
    parser.add_argument(
        "--no-geo-aggregate",
        action="store_true",
        help="Disattiva le geo feature aggregate e usa solo le feature preprocessate originali.",
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
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="Peso classi per gestire dataset sbilanciato. Default: balanced.",
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
    """Controlla che tutte le feature passate alla SVM siano numeriche."""
    feature_columns = X.columns.tolist()
    if not feature_columns:
        raise ValueError("Nessuna feature disponibile per addestrare la SVM.")

    non_numeric = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    if non_numeric:
        raise ValueError(f"La SVM richiede feature numeriche. Colonne non numeriche: {non_numeric}")

    return feature_columns


def build_svm_pipeline(
    estimator: str = "linear_svc",
    c: float = 1.0,
    class_weight: str | None = "balanced",
) -> Pipeline:
    """Costruisce la pipeline sklearn per SVM.

    La standardizzazione e' obbligatoria per SVM, perche' distanze e margini
    dipendono dalla scala delle feature. Non viene applicata PCA: qui la
    "decomposizione" rilevante e' quella multiclasse della SVM.
    """
    steps = [
        # Protezione extra: se nel dataset resta qualche NaN, la SVM non va in errore.
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    # balanced pesa di piu' le classi minoritarie e aiuta la F1-macro.
    class_weight_param = None if class_weight == "none" else class_weight

    if estimator == "linear_svc":
        # LinearSVC usa One-vs-Rest / One-vs-All per il problema multiclasse.
        svm_model = LinearSVC(
            C=c,
            class_weight=class_weight_param,
            max_iter=5000,
            random_state=RANDOM_STATE,
        )
    else:
        # SVC usa One-vs-One per il multiclasse; RBF e' piu' costoso ma non lineare.
        svm_model = SVC(
            C=c,
            class_weight=class_weight_param,
            kernel="rbf",
            gamma="scale",
            random_state=RANDOM_STATE,
        )

    steps.append(("svm", svm_model))
    return Pipeline(steps)


def get_multiclass_strategy(estimator: str) -> str:
    """Restituisce la decomposizione multiclasse usata dal classificatore SVM."""
    if estimator == "linear_svc":
        return "one_vs_rest"
    return "one_vs_one"


def evaluate_model(model: Pipeline, X_train, X_val, y_train, y_val) -> dict:
    """Addestra il modello e calcola le metriche principali."""
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    y_pred = model.predict(X_val)
    return {
        # Nel progetto F1-micro coincide con accuracy in classificazione single-label multiclass.
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "f1_micro": float(f1_score(y_val, y_pred, average="micro")),
        # F1-macro e' importante per capire se le classi minoritarie vengono trascurate.
        "f1_macro": float(f1_score(y_val, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, y_pred)),
        "fit_seconds": round(float(fit_seconds), 2),
    }


def save_outputs(output_dir: Path, metrics: dict, feature_columns: list[str]) -> None:
    """Salva metriche e lista feature per rendere il run tracciabile."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "svm_metrics.json").write_text(json.dumps(metrics, indent=4), encoding="utf-8")
    pd.DataFrame({"feature": feature_columns}).to_csv(output_dir / "svm_features.csv", index=False)


def main() -> dict:
    """Esegue il workflow completo: caricamento dati, split, training e report."""
    args = parse_args()

    print("\n" + "=" * 80)
    print("MODELLO SVM")
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

    model = build_svm_pipeline(
        estimator=args.estimator,
        c=args.c,
        class_weight=args.class_weight,
    )

    # Holdout stratificato: valida il modello senza alterare la distribuzione delle classi.
    scores = evaluate_model(model, X_train, X_val, y_train, y_val)
    metrics = {
        "dataset": str(args.dataset),
        "rows_used": int(len(df)),
        "n_features_input": int(len(feature_columns)),
        "test_size": float(args.test_size),
        "estimator": args.estimator,
        "class_weight": args.class_weight,
        "multiclass_strategy": get_multiclass_strategy(args.estimator),
        "geo_aggregate_enabled": not args.no_geo_aggregate,
        "raw_geo_ids_kept": bool(args.use_raw_geo_ids),
        "geo_smoothing": float(args.geo_smoothing),
        "geo_rare_threshold": int(args.geo_rare_threshold),
        "geo_n_splits": int(args.geo_n_splits),
        "scores": scores,
    }

    save_outputs(args.output_dir, metrics, feature_columns)

    print(f"Dataset: {args.dataset}")
    print(f"Righe usate: {len(df)}")
    print(f"Feature usate: {len(feature_columns)}")
    print(f"Estimator: {args.estimator}")
    print(f"Class weight: {args.class_weight}")
    print(f"Strategia multiclasse: {metrics['multiclass_strategy']}")
    print(f"Geo aggregate: {not args.no_geo_aggregate}")
    print(f"ID geo grezzi mantenuti: {args.use_raw_geo_ids}")
    print("\nMetriche validation:")
    print(pd.DataFrame([scores]).to_string(index=False))
    print(f"\nOutput salvato in: {args.output_dir}")

    return metrics


if __name__ == "__main__":
    main()
