"""
Valutazione standalone delle feature geografiche aggregate.

Lo script confronta:
1. baseline con le feature preprocessate originali;
2. baseline + feature generate da GeoFeatureEngineer.

La trasformazione sul train usa fit_transform_oof(...) per evitare leakage:
ogni fold riceve statistiche calcolate solo sugli altri fold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.data_selection import get_balanced_sample, get_stratified_sample
from src.preprocessing.geo_features import GeoFeatureEngineer


TARGET_COL = "damage_grade"
ID_COL = "building_id"
GEO_COLS = ("geo_level_1_id", "geo_level_2_id", "geo_level_3_id")
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confronto baseline vs baseline + geo aggregate features."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "Data" / "preprocessed" / "train_features_labels_preprocessed.csv",
        help="CSV con feature preprocessate e target.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["full", "balanced", "stratified"],
        default="stratified",
        help="Modalita di campionamento per una run piu veloce.",
    )
    parser.add_argument("--max-per-class", type=int, default=10000)
    parser.add_argument("--n-samples", type=int, default=30000)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--smoothing", type=float, default=20.0)
    parser.add_argument("--rare-threshold", type=int, default=10)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--estimator",
        choices=["hist_gradient_boosting", "random_forest", "logreg"],
        default="hist_gradient_boosting",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "geo_features")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset non trovato: {path}")

    df = pd.read_csv(path)
    required = [TARGET_COL, *GEO_COLS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti nel dataset: {missing}")
    return df


def sample_dataset(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.sample_mode == "balanced":
        return get_balanced_sample(df, TARGET_COL, args.max_per_class, args.random_state)
    if args.sample_mode == "stratified":
        return get_stratified_sample(df, TARGET_COL, args.n_samples, args.random_state)
    return df.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)


def build_estimator(name: str, random_state: int):
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=random_state),
        )
    return HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.08,
        random_state=random_state,
    )


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COL])
    if ID_COL in X.columns:
        X = X.drop(columns=[ID_COL])
    y = df[TARGET_COL]
    return X, y


def evaluate_model(name: str, estimator, X_train, y_train, X_val, y_val) -> dict:
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_val)

    return {
        "setup": name,
        "n_features": int(X_train.shape[1]),
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "f1_micro": float(f1_score(y_val, y_pred, average="micro")),
        "f1_macro": float(f1_score(y_val, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, y_pred)),
        "classification_report": classification_report(y_val, y_pred, output_dict=True),
    }


def main() -> None:
    args = parse_args()

    df = load_dataset(args.input)
    df = sample_dataset(df, args)

    train_df, val_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df[TARGET_COL],
    )

    X_train_base, y_train = prepare_xy(train_df)
    X_val_base, y_val = prepare_xy(val_df)

    geo_engineer = GeoFeatureEngineer(
        geo_columns=GEO_COLS,
        target_col=TARGET_COL,
        smoothing=args.smoothing,
        rare_threshold=args.rare_threshold,
        n_splits=args.n_splits,
        random_state=args.random_state,
        append_original=True,
    )

    X_train_geo = geo_engineer.fit_transform_oof(X_train_base, y_train)
    X_val_geo = geo_engineer.transform(X_val_base)

    if X_train_geo.isnull().any().any() or X_val_geo.isnull().any().any():
        raise ValueError("Le geo feature generate contengono NaN.")

    baseline_result = evaluate_model(
        "baseline_raw_geo",
        build_estimator(args.estimator, args.random_state),
        X_train_base,
        y_train,
        X_val_base,
        y_val,
    )
    geo_result = evaluate_model(
        "baseline_plus_geo_aggregate",
        build_estimator(args.estimator, args.random_state),
        X_train_geo,
        y_train,
        X_val_geo,
        y_val,
    )

    results = [baseline_result, geo_result]
    summary = {
        "input": str(args.input),
        "sample_mode": args.sample_mode,
        "n_rows_used": int(len(df)),
        "estimator": args.estimator,
        "smoothing": float(args.smoothing),
        "rare_threshold": int(args.rare_threshold),
        "n_splits_oof": int(args.n_splits),
        "baseline_f1_micro": baseline_result["f1_micro"],
        "geo_f1_micro": geo_result["f1_micro"],
        "delta_f1_micro": geo_result["f1_micro"] - baseline_result["f1_micro"],
        "baseline_accuracy": baseline_result["accuracy"],
        "geo_accuracy": geo_result["accuracy"],
        "delta_accuracy": geo_result["accuracy"] - baseline_result["accuracy"],
        "n_features_baseline": baseline_result["n_features"],
        "n_features_geo": geo_result["n_features"],
        "geo_feature_columns_added": int(X_train_geo.shape[1] - X_train_base.shape[1]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {key: value for key, value in result.items() if key != "classification_report"}
            for result in results
        ]
    ).to_csv(args.output_dir / "geo_features_results.csv", index=False)
    X_train_geo.head(50).to_csv(args.output_dir / "geo_features_preview.csv", index=False)

    with open(args.output_dir / "geo_features_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(args.output_dir / "geo_features_classification_reports.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                result["setup"]: result["classification_report"]
                for result in results
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("VALUTAZIONE GEO FEATURES COMPLETATA")
    print("=" * 80)
    print(f"Dataset: {args.input}")
    print(f"Righe usate: {len(df)}")
    print(f"Estimator: {args.estimator}")
    print(f"Feature baseline: {summary['n_features_baseline']}")
    print(f"Feature con geo aggregate: {summary['n_features_geo']}")
    print(f"Geo feature aggiunte: {summary['geo_feature_columns_added']}")
    print("\nRisultati:")
    print(pd.DataFrame([{k: v for k, v in r.items() if k != 'classification_report'} for r in results]).to_string(index=False))
    print("\nDelta:")
    print(f"F1-micro: {summary['delta_f1_micro']:+.6f}")
    print(f"Accuracy: {summary['delta_accuracy']:+.6f}")
    print(f"\nOutput salvato in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
