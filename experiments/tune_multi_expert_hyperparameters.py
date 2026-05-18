"""
Hyperparameter tuning mirato agli esperti piu' utili del Multi-Expert System.

Output:
    experiments/multi_expert_tuning_results.csv
    experiments/multi_expert_best_params.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.evaluate_multi_expert import load_feature_list, prepare_data
from src.preprocessing.data_selection import get_balanced_sample, get_stratified_sample


TARGET_COL = "damage_grade"
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Tuning mirato degli esperti del MES.")
    parser.add_argument("--sample-mode", choices=["balanced", "stratified"], default="balanced")
    parser.add_argument("--max-per-class", type=int, default=3000)
    parser.add_argument("--n-samples", type=int, default=30000)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def load_dataset():
    data_path = PROJECT_ROOT / "DataPreprocessed" / "processed" / "train_features_labels_preprocessed.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset preprocessato non trovato: {data_path}")
    return pd.read_csv(data_path)


def build_feature_sets(X, top_k):
    lasso_features = load_feature_list(
        PROJECT_ROOT / "Feature Selection" / "Embedded" / "outputs" / "lasso_selected_features.csv",
        column="feature",
        limit=top_k,
    )
    lasso_features = [feature for feature in lasso_features if feature in X.columns]
    if not lasso_features:
        lasso_features = X.columns.tolist()[:top_k]

    return {
        "full": X.columns.tolist(),
        "lasso_top25": lasso_features,
    }


def build_tuning_configs(feature_sets):
    return [
        {
            "name": "logistic_lasso_top25",
            "features": feature_sets["lasso_top25"],
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1500, random_state=RANDOM_STATE)),
                ]
            ),
            "param_grid": {
                "clf__C": [0.1, 0.3, 1.0, 3.0, 10.0],
                "clf__class_weight": [None, "balanced"],
            },
        },
        {
            "name": "random_forest_full",
            "features": feature_sets["full"],
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            "param_grid": {
                "n_estimators": [200, 400],
                "max_depth": [None, 16, 24],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            },
        },
        {
            "name": "hist_gradient_boosting_full",
            "features": feature_sets["full"],
            "estimator": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "learning_rate": [0.04, 0.06, 0.08],
                "max_iter": [160, 220],
                "max_leaf_nodes": [15, 31],
                "l2_regularization": [0.0, 0.01, 0.1],
            },
        },
        {
            "name": "hist_gradient_boosting_lasso_top25",
            "features": feature_sets["lasso_top25"],
            "estimator": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "learning_rate": [0.04, 0.06, 0.08],
                "max_iter": [160, 220],
                "max_leaf_nodes": [15, 31],
                "l2_regularization": [0.0, 0.01, 0.1],
            },
        },
        {
            "name": "adaboost_lasso_top25",
            "features": feature_sets["lasso_top25"],
            "estimator": AdaBoostClassifier(
                estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "n_estimators": [80, 120, 180],
                "learning_rate": [0.3, 0.5, 0.8],
                "estimator__max_depth": [1, 2, 3],
                "estimator__min_samples_leaf": [4, 8, 16],
            },
        },
    ]


def tune_config(config, X_train, y_train, X_val, y_val, cv_folds, n_jobs):
    name = config["name"]
    features = config["features"]
    param_grid = config["param_grid"]

    combinations = 1
    for values in param_grid.values():
        combinations *= len(values)

    print("\n" + "=" * 80)
    print(f"Tuning {name}")
    print(f"Feature: {len(features)} | combinazioni: {combinations} x {cv_folds} fold")
    print("=" * 80)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=config["estimator"],
        param_grid=param_grid,
        scoring="f1_micro",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=1,
    )

    start = time.time()
    grid.fit(X_train[features], y_train)
    elapsed = time.time() - start

    y_pred = grid.best_estimator_.predict(X_val[features])
    f1_micro = f1_score(y_val, y_pred, average="micro")
    f1_macro = f1_score(y_val, y_pred, average="macro")
    accuracy = accuracy_score(y_val, y_pred)

    print(f"Miglior CV f1_micro: {grid.best_score_:.4f}")
    print(f"Validation f1_micro: {f1_micro:.4f}")
    print(f"Best params: {grid.best_params_}")

    return {
        "model": name,
        "n_features": len(features),
        "cv_best_f1_micro": grid.best_score_,
        "val_f1_micro": f1_micro,
        "val_f1_macro": f1_macro,
        "val_accuracy": accuracy,
        "time_s": elapsed,
        "best_params": grid.best_params_,
        "classification_report": classification_report(y_val, y_pred, digits=4, output_dict=True),
    }


def main():
    args = parse_args()
    output_dir = PROJECT_ROOT / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TUNING ESPERTI MULTI-EXPERT SYSTEM")
    print("=" * 80)

    df = load_dataset()
    if args.sample_mode == "balanced":
        df_eval = get_balanced_sample(df, TARGET_COL, max_per_class=args.max_per_class)
    else:
        df_eval = get_stratified_sample(df, TARGET_COL, n_samples=args.n_samples)

    print(f"Dataset tuning: {df_eval.shape[0]} righe")
    X, y = prepare_data(df_eval)
    feature_sets = build_feature_sets(X, args.top_k)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results = []
    best_params = {}
    for config in build_tuning_configs(feature_sets):
        result = tune_config(
            config,
            X_train,
            y_train,
            X_val,
            y_val,
            cv_folds=args.cv_folds,
            n_jobs=args.n_jobs,
        )
        best_params[result["model"]] = result["best_params"]
        results.append(result)

    results_df = pd.DataFrame(
        [
            {key: value for key, value in result.items() if key not in {"best_params", "classification_report"}}
            | {"best_params": str(result["best_params"])}
            for result in results
        ]
    ).sort_values(["val_f1_micro", "val_f1_macro"], ascending=False)

    results_path = output_dir / "multi_expert_tuning_results.csv"
    params_path = output_dir / "multi_expert_best_params.json"
    results_df.to_csv(results_path, index=False)
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    print("\n" + "=" * 80)
    print("RISULTATI TUNING")
    print("=" * 80)
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nRisultati salvati in: {results_path}")
    print(f"Best params salvati in: {params_path}")

    return results_df


if __name__ == "__main__":
    main()
