"""
Benchmark di un sistema multi-esperto per Richter's Predictor.

Lo script confronta:
    - singoli esperti eterogenei;
    - aggregazioni del decision profile: mean, weighted_mean, median, product;
    - majority voting pesato.

Output:
    experiments/multi_expert_results.csv
    experiments/multi_expert_diversity.csv
"""

import argparse
import heapq
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ensemble import MultiExpertSystem
from src.preprocessing.data_selection import get_balanced_sample, get_stratified_sample


TARGET_COL = "damage_grade"
EXCLUDE_COLS = ["building_id", "geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Valuta un Multi-Expert System sul dataset terremoto.")
    parser.add_argument(
        "--sample-mode",
        choices=["balanced", "stratified"],
        default="balanced",
        help="Tipo di campionamento per il benchmark.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=5000,
        help="Numero massimo di esempi per classe con sample-mode=balanced.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20000,
        help="Numero totale di esempi con sample-mode=stratified.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Quota di validation set stratificato.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Numero di feature da usare per i ranking Lasso, Relief e Information Gain.",
    )
    parser.add_argument(
        "--relief-iterations",
        type=int,
        default=2000,
        help="Numero di iterazioni per calcolare il ranking Relief sul train split.",
    )
    parser.add_argument(
        "--top-experts",
        type=int,
        default=4,
        help="Numero di migliori esperti singoli da usare nella variante MES top-k.",
    )
    parser.add_argument(
        "--subset-max-rows",
        type=int,
        default=1200,
        help="Righe massime usate internamente da Best First e SBS.",
    )
    parser.add_argument(
        "--best-first-patience",
        type=int,
        default=3,
        help="Numero di espansioni senza miglioramento prima dello stop Best First.",
    )
    parser.add_argument(
        "--sbs-min-features",
        type=int,
        default=20,
        help="Numero minimo di feature a cui SBS puo' arrivare.",
    )
    parser.add_argument(
        "--sbs-max-steps",
        type=int,
        default=12,
        help="Numero massimo di rimozioni SBS per contenere il costo computazionale.",
    )
    parser.add_argument(
        "--diversity-weight",
        type=float,
        default=0.12,
        help="Peso del bonus di diversita' nella selezione greedy degli esperti.",
    )
    parser.add_argument(
        "--double-fault-weight",
        type=float,
        default=0.08,
        help="Peso della penalita' double fault nella selezione greedy degli esperti.",
    )
    parser.add_argument(
        "--tuned-params",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "multi_expert_best_params.json",
        help="JSON opzionale con iperparametri ottimizzati per gli esperti.",
    )
    return parser.parse_args()


def load_dataset():
    data_path = PROJECT_ROOT / "Data" / "preprocessed" / "train_features_labels_preprocessed.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset preprocessato non trovato: {data_path}. Esegui prima main.py."
        )
    return pd.read_csv(data_path)


def load_feature_list(path, column, limit=None):
    path = Path(path)
    if not path.exists():
        return []

    df = pd.read_csv(path)
    if column not in df.columns:
        return []

    values = df[column].dropna().astype(str).tolist()
    return values[:limit] if limit else values


def prepare_data(df):
    exclude = [col for col in EXCLUDE_COLS if col in df.columns]
    X = df.drop(columns=[TARGET_COL] + exclude)

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=float)

    y = df[TARGET_COL].astype(int)
    return X, y


def minmax_scale(df):
    min_values = df.min(axis=0)
    ranges = (df.max(axis=0) - min_values).replace(0, 1.0)
    return (df - min_values) / ranges


def compute_relief_ranking(X_train, y_train, n_iterations=2000, n_neighbors_search=50):
    X_scaled = minmax_scale(X_train).to_numpy(dtype=float)
    y_array = y_train.to_numpy()
    n_samples, n_features = X_scaled.shape

    k = max(3, min(n_neighbors_search, n_samples))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    nn.fit(X_scaled)
    neighbor_indices = nn.kneighbors(X_scaled, return_distance=False)

    rng = np.random.default_rng(RANDOM_STATE)
    sampled_indices = rng.integers(0, n_samples, size=n_iterations)
    weights = np.zeros(n_features, dtype=float)
    valid_updates = 0

    for sample_idx in sampled_indices:
        current_label = y_array[sample_idx]
        near_hit_idx = -1
        near_miss_idx = -1

        for neighbor_idx in neighbor_indices[sample_idx]:
            if neighbor_idx == sample_idx:
                continue
            if y_array[neighbor_idx] == current_label and near_hit_idx == -1:
                near_hit_idx = int(neighbor_idx)
            if y_array[neighbor_idx] != current_label and near_miss_idx == -1:
                near_miss_idx = int(neighbor_idx)
            if near_hit_idx != -1 and near_miss_idx != -1:
                break

        if near_hit_idx == -1 or near_miss_idx == -1:
            continue

        diff_hit = X_scaled[sample_idx] - X_scaled[near_hit_idx]
        diff_miss = X_scaled[sample_idx] - X_scaled[near_miss_idx]
        weights += -(diff_hit * diff_hit) + (diff_miss * diff_miss)
        valid_updates += 1

    if valid_updates == 0:
        raise ValueError("Relief non ha prodotto aggiornamenti validi.")

    return (
        pd.DataFrame(
            {
                "feature": X_train.columns,
                "relief_weight": weights / float(valid_updates),
            }
        )
        .sort_values("relief_weight", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def entropy(series):
    probabilities = series.value_counts(normalize=True)
    return float(-(probabilities * probabilities.apply(lambda value: math.log(value, 2))).sum())


def compute_information_gain_ranking(X_train, y_train):
    rows = []
    target_entropy = entropy(y_train)

    for feature in X_train.columns:
        series = X_train[feature]
        is_discrete = (
            pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_integer_dtype(series)
            or series.nunique(dropna=True) <= 50
        )
        if not is_discrete:
            continue

        pair = pd.DataFrame({"target": y_train, "feature": series}).dropna()
        conditional_entropy = 0.0
        for _, group in pair.groupby("feature", dropna=False):
            conditional_entropy += (len(group) / len(pair)) * entropy(group["target"])

        rows.append(
            {
                "feature": feature,
                "information_gain": max(0.0, target_entropy - conditional_entropy),
                "n_unique_values": int(series.nunique(dropna=True)),
            }
        )

    if not rows:
        raise ValueError("Nessuna feature discreta disponibile per Information Gain.")

    return (
        pd.DataFrame(rows)
        .sort_values("information_gain", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def sample_for_subset_selection(X_train, y_train, max_rows):
    if max_rows is None or len(X_train) <= max_rows:
        return X_train.copy(), y_train.copy()

    X_sub, _, y_sub, _ = train_test_split(
        X_train,
        y_train,
        train_size=max_rows,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    return X_sub.reset_index(drop=True), y_sub.reset_index(drop=True)


def compute_best_first_subset(X_train, y_train, max_rows=1200, patience=3):
    X_work, y_work = sample_for_subset_selection(X_train, y_train, max_rows)
    feature_names = X_work.columns.to_numpy()
    x_array = X_work.to_numpy(dtype=float)
    y_array = y_work.to_numpy()
    total_features = x_array.shape[1]

    def evaluate(feature_idx):
        if not feature_idx:
            return 0.0
        estimator = DecisionTreeClassifier(random_state=RANDOM_STATE)
        scores = cross_val_score(
            estimator,
            x_array[:, list(feature_idx)],
            y_array,
            cv=3,
            scoring="f1_micro",
            n_jobs=-1,
        )
        return float(np.mean(scores))

    open_list = []
    closed_set = set()
    best_score = -np.inf
    best_subset = tuple()

    for idx in range(total_features):
        subset = (idx,)
        score = evaluate(subset)
        heapq.heappush(open_list, (-score, subset))
        closed_set.add(subset)
        if score > best_score:
            best_score = score
            best_subset = subset

    expansions_without_improvement = 0
    evaluated_models = total_features

    while open_list and expansions_without_improvement < patience:
        _, current_subset = heapq.heappop(open_list)
        improved = False

        for idx in range(total_features):
            if idx in current_subset:
                continue

            new_subset = tuple(sorted(current_subset + (idx,)))
            if new_subset in closed_set:
                continue

            closed_set.add(new_subset)
            child_score = evaluate(new_subset)
            evaluated_models += 1
            heapq.heappush(open_list, (-child_score, new_subset))

            if child_score > best_score:
                best_score = child_score
                best_subset = new_subset
                improved = True

        if improved:
            expansions_without_improvement = 0
        else:
            expansions_without_improvement += 1

    return [str(feature_names[idx]) for idx in best_subset], {
        "score": best_score,
        "evaluated_models": evaluated_models,
        "n_rows_used": len(X_work),
        "patience": patience,
    }


def compute_sbs_subset(X_train, y_train, min_features=20, max_rows=1200, max_steps=12):
    X_work, y_work = sample_for_subset_selection(X_train, y_train, max_rows)
    X_subtrain, X_subval, y_subtrain, y_subval = train_test_split(
        X_work,
        y_work,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_work,
    )

    feature_names = X_work.columns.to_numpy()
    x_train_array = X_subtrain.to_numpy(dtype=float)
    x_val_array = X_subval.to_numpy(dtype=float)
    y_train_array = y_subtrain.to_numpy()
    y_val_array = y_subval.to_numpy()
    current_idx = np.arange(len(feature_names))

    def evaluate(feature_idx):
        estimator = KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            algorithm="brute",
            n_jobs=-1,
        )
        estimator.fit(x_train_array[:, feature_idx], y_train_array)
        predictions = estimator.predict(x_val_array[:, feature_idx])
        return float(f1_score(y_val_array, predictions, average="micro"))

    current_score = evaluate(current_idx)
    evaluated_models = 1
    steps = 0

    while len(current_idx) > min_features and steps < max_steps:
        best_score = -np.inf
        best_idx = None

        for feature_pos in range(len(current_idx)):
            candidate_idx = np.delete(current_idx, feature_pos)
            candidate_score = evaluate(candidate_idx)
            evaluated_models += 1

            if candidate_score > best_score:
                best_score = candidate_score
                best_idx = candidate_idx

        if best_idx is None or best_score < current_score:
            break

        current_idx = best_idx
        current_score = best_score
        steps += 1

    return [str(feature_names[idx]) for idx in current_idx], {
        "score": current_score,
        "evaluated_models": evaluated_models,
        "n_rows_used": len(X_work),
        "steps": steps,
    }


def build_feature_sets(
    X_train,
    y_train,
    top_k=25,
    relief_iterations=2000,
    subset_max_rows=1200,
    best_first_patience=3,
    sbs_min_features=20,
    sbs_max_steps=12,
):
    lasso_features = load_feature_list(
        PROJECT_ROOT / "Feature Selection" / "Embedded" / "outputs" / "lasso_selected_features.csv",
        column="feature",
        limit=top_k,
    )
    max_min_features = load_feature_list(
        PROJECT_ROOT / "Feature Selection" / "subset selection" / "outputs" / "max_min_selected_features.csv",
        column="selected_feature",
    )

    lasso_features = [feature for feature in lasso_features if feature in X_train.columns]
    max_min_features = [feature for feature in max_min_features if feature in X_train.columns]

    if not lasso_features:
        lasso_features = X_train.columns.tolist()[:top_k]
    if not max_min_features:
        max_min_features = lasso_features[: min(12, len(lasso_features))]

    relief_ranking = compute_relief_ranking(
        X_train,
        y_train,
        n_iterations=relief_iterations,
        n_neighbors_search=50,
    )
    relief_features = relief_ranking["feature"].head(top_k).tolist()
    relief_features = [feature for feature in relief_features if feature in X_train.columns]

    information_gain_ranking = compute_information_gain_ranking(X_train, y_train)
    information_gain_features = information_gain_ranking["feature"].head(top_k).tolist()
    information_gain_features = [
        feature for feature in information_gain_features if feature in X_train.columns
    ]

    best_first_features, best_first_summary = compute_best_first_subset(
        X_train,
        y_train,
        max_rows=subset_max_rows,
        patience=best_first_patience,
    )
    sbs_features, sbs_summary = compute_sbs_subset(
        X_train,
        y_train,
        min_features=sbs_min_features,
        max_rows=subset_max_rows,
        max_steps=sbs_max_steps,
    )

    return {
        "full": X_train.columns.tolist(),
        "lasso_top25": lasso_features,
        "max_min": max_min_features,
        "relief_top25": relief_features,
        "information_gain_top25": information_gain_features,
        "best_first": best_first_features,
        "sbs": sbs_features,
        "_summaries": {
            "best_first": best_first_summary,
            "sbs": sbs_summary,
        },
    }


def build_experts(feature_sets):
    return [
        {
            "name": "knn_lasso_top25",
            "features": feature_sets["lasso_top25"],
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)),
                ]
            ),
        },
        {
            "name": "knn_relief_top25",
            "features": feature_sets["relief_top25"],
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)),
                ]
            ),
        },
        {
            "name": "decision_tree_max_min",
            "features": feature_sets["max_min"],
            "estimator": DecisionTreeClassifier(
                criterion="entropy",
                max_depth=15,
                min_samples_leaf=8,
                random_state=RANDOM_STATE,
            ),
        },
        {
            "name": "decision_tree_best_first",
            "features": feature_sets["best_first"],
            "estimator": DecisionTreeClassifier(
                criterion="entropy",
                max_depth=15,
                min_samples_leaf=8,
                random_state=RANDOM_STATE,
            ),
        },
        {
            "name": "adaboost_lasso_top25",
            "features": feature_sets["lasso_top25"],
            "estimator": AdaBoostClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=2,
                    min_samples_leaf=8,
                    random_state=RANDOM_STATE,
                ),
                n_estimators=120,
                learning_rate=0.5,
                random_state=RANDOM_STATE,
            ),
        },
        {
            "name": "adaboost_best_first",
            "features": feature_sets["best_first"],
            "estimator": AdaBoostClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=2,
                    min_samples_leaf=8,
                    random_state=RANDOM_STATE,
                ),
                n_estimators=120,
                learning_rate=0.5,
                random_state=RANDOM_STATE,
            ),
        },
        {
            "name": "knn_sbs",
            "features": feature_sets["sbs"],
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)),
                ]
            ),
        },
        {
            "name": "random_forest_full",
            "features": feature_sets["full"],
            "estimator": RandomForestClassifier(
                n_estimators=250,
                max_depth=None,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        {
            "name": "random_forest_information_gain_top25",
            "features": feature_sets["information_gain_top25"],
            "estimator": RandomForestClassifier(
                n_estimators=250,
                max_depth=None,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        {
            "name": "hist_gradient_boosting_lasso_top25",
            "features": feature_sets["lasso_top25"],
            "estimator": HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=180,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=RANDOM_STATE,
            ),
        },
        {
            "name": "hist_gradient_boosting_full",
            "features": feature_sets["full"],
            "estimator": HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_iter=180,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=RANDOM_STATE,
            ),
        },
        {
            "name": "logistic_lasso_top25",
            "features": feature_sets["lasso_top25"],
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=1.0,
                            max_iter=1000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        },
    ]


def apply_tuned_params(experts, params_path):
    params_path = Path(params_path)
    if not params_path.exists():
        print(f"Parametri tuned non trovati: {params_path}. Uso configurazione default.")
        return experts

    with open(params_path, "r", encoding="utf-8") as f:
        tuned_params = json.load(f)

    updated_experts = []
    applied = []
    for expert in experts:
        expert_out = dict(expert)
        model_name = expert_out["name"]
        if model_name in tuned_params:
            estimator = clone(expert_out["estimator"])
            estimator.set_params(**tuned_params[model_name])
            expert_out["estimator"] = estimator
            applied.append(model_name)
        updated_experts.append(expert_out)

    if applied:
        print(f"Parametri tuned applicati a: {', '.join(applied)}")
    else:
        print(f"Nessun parametro tuned applicabile trovato in: {params_path}")

    return updated_experts


def save_feature_sets(feature_sets, output_dir):
    rows = []
    for set_name, features in feature_sets.items():
        if set_name.startswith("_"):
            continue
        for rank, feature in enumerate(features, start=1):
            rows.append({"feature_set": set_name, "rank": rank, "feature": feature})

    output_path = output_dir / "multi_expert_feature_sets.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)

    summaries = feature_sets.get("_summaries", {})
    if summaries:
        summary_rows = []
        for method, values in summaries.items():
            row = {"method": method}
            row.update(values)
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(output_dir / "multi_expert_subset_summaries.csv", index=False)

    return output_path


def evaluate_predictions(name, y_true, y_pred, elapsed_s=0.0):
    return {
        "model": name,
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "time_s": elapsed_s,
    }


def fit_and_score_experts(experts, X_train, y_train, X_val, y_val):
    rows = []
    weighted_experts = []
    validation_predictions = {}

    for expert in experts:
        start = time.time()
        estimator = expert["estimator"]
        features = expert["features"]
        estimator.fit(X_train[features], y_train)
        y_pred = estimator.predict(X_val[features])
        elapsed = time.time() - start
        validation_predictions[expert["name"]] = np.asarray(y_pred)

        row = evaluate_predictions(expert["name"], y_val, y_pred, elapsed)
        rows.append(row)

        weighted_expert = dict(expert)
        weighted_expert["weight"] = max(row["f1_micro"], 1e-6)
        weighted_experts.append(weighted_expert)

    return rows, weighted_experts, validation_predictions


def evaluate_aggregations(experts, X_train, y_train, X_val, y_val, label_prefix="mes"):
    rows = []
    systems = {}

    for aggregation in ["mean", "weighted_mean", "median", "product", "majority_vote"]:
        start = time.time()
        mes = MultiExpertSystem(experts=experts, aggregation=aggregation)
        mes.fit(X_train, y_train)
        y_pred = mes.predict(X_val)
        elapsed = time.time() - start

        rows.append(evaluate_predictions(f"{label_prefix}_{aggregation}", y_val, y_pred, elapsed))
        systems[f"{label_prefix}_{aggregation}"] = (mes, y_pred)

    return rows, systems


def fit_experts_for_stacking(experts, X_train, y_train):
    fitted = []
    for expert in experts:
        estimator = clone(expert["estimator"])
        features = expert["features"]
        estimator.fit(X_train[features], y_train)
        fitted.append(
            {
                "name": expert["name"],
                "estimator": estimator,
                "features": features,
            }
        )
    return fitted


def build_decision_profile_matrix(fitted_experts, X, classes):
    profile_blocks = []
    for expert in fitted_experts:
        estimator = expert["estimator"]
        features = expert["features"]

        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(X[features])
            estimator_classes = estimator.classes_
        else:
            predictions = estimator.predict(X[features])
            estimator_classes = classes
            proba = np.zeros((len(predictions), len(estimator_classes)))
            for idx, class_label in enumerate(estimator_classes):
                proba[:, idx] = predictions == class_label

        aligned = np.zeros((X.shape[0], len(classes)))
        for source_idx, class_label in enumerate(estimator_classes):
            target_idx = int(np.where(classes == class_label)[0][0])
            aligned[:, target_idx] = proba[:, source_idx]
        profile_blocks.append(aligned)

    return np.hstack(profile_blocks)


def evaluate_stacking(experts, X_train, y_train, X_val, y_val):
    rows = []
    systems = {}

    X_base, X_meta, y_base, y_meta = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    classes = np.array(sorted(y_train.unique()))

    start = time.time()
    fitted_experts = fit_experts_for_stacking(experts, X_base, y_base)
    X_meta_profile = build_decision_profile_matrix(fitted_experts, X_meta, classes)
    X_val_profile = build_decision_profile_matrix(fitted_experts, X_val, classes)

    meta_clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    meta_clf.fit(X_meta_profile, y_meta)
    y_pred = meta_clf.predict(X_val_profile)
    elapsed = time.time() - start

    rows.append(evaluate_predictions("mes_stacking_profile", y_val, y_pred, elapsed))
    systems["mes_stacking_profile"] = (None, y_pred)

    start = time.time()
    X_meta_plus = np.hstack([X_meta_profile, X_meta.to_numpy(dtype=float)])
    X_val_plus = np.hstack([X_val_profile, X_val.to_numpy(dtype=float)])
    meta_plus = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.5,
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    meta_plus.fit(X_meta_plus, y_meta)
    y_pred_plus = meta_plus.predict(X_val_plus)
    elapsed_plus = time.time() - start

    rows.append(evaluate_predictions("mes_stacking_profile_plus_features", y_val, y_pred_plus, elapsed_plus))
    systems["mes_stacking_profile_plus_features"] = (None, y_pred_plus)

    return rows, systems


def select_top_experts(weighted_experts, single_rows, top_k):
    if top_k <= 0 or top_k >= len(weighted_experts):
        return weighted_experts

    ranking = (
        pd.DataFrame(single_rows)
        .sort_values(["f1_micro", "f1_macro"], ascending=False)
        .head(top_k)
    )
    selected_names = set(ranking["model"])
    return [expert for expert in weighted_experts if expert["name"] in selected_names]


def pairwise_disagreement_and_double_fault(pred_a, pred_b, y_true):
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    disagreement = np.mean(correct_a != correct_b)
    double_fault = np.mean(~correct_a & ~correct_b)
    return float(disagreement), float(double_fault)


def select_diverse_experts(
    weighted_experts,
    single_rows,
    validation_predictions,
    y_val,
    top_k,
    diversity_weight=0.12,
    double_fault_weight=0.08,
):
    if top_k <= 0 or top_k >= len(weighted_experts):
        return weighted_experts, pd.DataFrame()

    metrics = {
        row["model"]: {
            "f1_micro": row["f1_micro"],
            "f1_macro": row["f1_macro"],
        }
        for row in single_rows
    }
    experts_by_name = {expert["name"]: expert for expert in weighted_experts}
    y_true = y_val.to_numpy()

    first_name = max(metrics, key=lambda name: (metrics[name]["f1_micro"], metrics[name]["f1_macro"]))
    selected_names = [first_name]
    selection_rows = [
        {
            "step": 1,
            "expert": first_name,
            "f1_micro": metrics[first_name]["f1_micro"],
            "avg_disagreement": 0.0,
            "avg_double_fault": 0.0,
            "selection_score": metrics[first_name]["f1_micro"],
        }
    ]

    while len(selected_names) < top_k:
        best_candidate = None
        best_row = None

        for candidate_name in metrics:
            if candidate_name in selected_names:
                continue

            disagreements = []
            double_faults = []
            candidate_pred = validation_predictions[candidate_name]

            for selected_name in selected_names:
                disagreement, double_fault = pairwise_disagreement_and_double_fault(
                    candidate_pred,
                    validation_predictions[selected_name],
                    y_true,
                )
                disagreements.append(disagreement)
                double_faults.append(double_fault)

            avg_disagreement = float(np.mean(disagreements))
            avg_double_fault = float(np.mean(double_faults))
            selection_score = (
                metrics[candidate_name]["f1_micro"]
                + diversity_weight * avg_disagreement
                - double_fault_weight * avg_double_fault
            )

            row = {
                "step": len(selected_names) + 1,
                "expert": candidate_name,
                "f1_micro": metrics[candidate_name]["f1_micro"],
                "avg_disagreement": avg_disagreement,
                "avg_double_fault": avg_double_fault,
                "selection_score": selection_score,
            }

            if best_row is None or (
                row["selection_score"],
                row["f1_micro"],
            ) > (
                best_row["selection_score"],
                best_row["f1_micro"],
            ):
                best_candidate = candidate_name
                best_row = row

        if best_candidate is None:
            break

        selected_names.append(best_candidate)
        selection_rows.append(best_row)

    selected = [experts_by_name[name] for name in selected_names]
    return selected, pd.DataFrame(selection_rows)


def main():
    args = parse_args()
    output_dir = PROJECT_ROOT / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MULTI-EXPERT SYSTEM - BENCHMARK")
    print("=" * 80)

    df = load_dataset()
    print(f"Dataset caricato: {df.shape[0]} righe x {df.shape[1]} colonne")

    if args.sample_mode == "balanced":
        df_eval = get_balanced_sample(df, TARGET_COL, max_per_class=args.max_per_class)
        print(f"Campione bilanciato: {df_eval.shape[0]} righe")
    else:
        df_eval = get_stratified_sample(df, TARGET_COL, n_samples=args.n_samples)
        print(f"Campione stratificato: {df_eval.shape[0]} righe")

    X, y = prepare_data(df_eval)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nCalcolo subset di feature sul train split...")
    feature_sets = build_feature_sets(
        X_train,
        y_train,
        top_k=args.top_k,
        relief_iterations=args.relief_iterations,
        subset_max_rows=args.subset_max_rows,
        best_first_patience=args.best_first_patience,
        sbs_min_features=args.sbs_min_features,
        sbs_max_steps=args.sbs_max_steps,
    )
    feature_sets_path = save_feature_sets(feature_sets, output_dir)

    print("Feature set disponibili:")
    for name, features in feature_sets.items():
        if name.startswith("_"):
            continue
        print(f"- {name}: {len(features)} feature")

    experts = apply_tuned_params(build_experts(feature_sets), args.tuned_params)

    print("\nValutazione esperti singoli...")
    single_rows, weighted_experts, validation_predictions = fit_and_score_experts(
        experts,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    print("\nValutazione aggregazioni del decision profile...")
    ensemble_rows, systems = evaluate_aggregations(weighted_experts, X_train, y_train, X_val, y_val)

    print("\nValutazione stacking sul decision profile...")
    stacking_rows, stacking_systems = evaluate_stacking(weighted_experts, X_train, y_train, X_val, y_val)
    ensemble_rows.extend(stacking_rows)
    systems.update(stacking_systems)

    top_experts = select_top_experts(weighted_experts, single_rows, args.top_experts)
    if len(top_experts) < len(weighted_experts):
        top_names = ", ".join(expert["name"] for expert in top_experts)
        print(f"\nValutazione MES top-{len(top_experts)} esperti: {top_names}")
        top_rows, top_systems = evaluate_aggregations(
            top_experts,
            X_train,
            y_train,
            X_val,
            y_val,
            label_prefix=f"mes_top{len(top_experts)}",
        )
        ensemble_rows.extend(top_rows)
        systems.update(top_systems)

    diverse_experts, diverse_selection_df = select_diverse_experts(
        weighted_experts,
        single_rows,
        validation_predictions,
        y_val,
        args.top_experts,
        diversity_weight=args.diversity_weight,
        double_fault_weight=args.double_fault_weight,
    )
    if len(diverse_experts) < len(weighted_experts):
        diverse_names = ", ".join(expert["name"] for expert in diverse_experts)
        diverse_path = output_dir / "multi_expert_diverse_selection.csv"
        diverse_selection_df.to_csv(diverse_path, index=False)
        print(f"\nValutazione MES diverse-{len(diverse_experts)} esperti: {diverse_names}")
        diverse_rows, diverse_systems = evaluate_aggregations(
            diverse_experts,
            X_train,
            y_train,
            X_val,
            y_val,
            label_prefix=f"mes_diverse{len(diverse_experts)}",
        )
        ensemble_rows.extend(diverse_rows)
        systems.update(diverse_systems)
    else:
        diverse_path = None

    results_df = pd.DataFrame(single_rows + ensemble_rows).sort_values(
        ["f1_micro", "f1_macro"], ascending=False
    )
    results_path = output_dir / "multi_expert_results.csv"
    results_df.to_csv(results_path, index=False)

    best_row = results_df.iloc[0]
    best_name = best_row["model"]
    print("\n" + "=" * 80)
    print("RISULTATI")
    print("=" * 80)
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print(f"\nMiglior modello: {best_name}")
    diversity_mes, _ = systems["mes_weighted_mean"]
    diversity_df = diversity_mes.diversity_report(X_val, y_val)
    diversity_path = output_dir / "multi_expert_diversity.csv"
    diversity_df.to_csv(diversity_path, index=False)

    if best_name.startswith("mes_"):
        _, best_pred = systems[best_name]

        print("\nClassification report miglior MES:")
        print(classification_report(y_val, best_pred, digits=4))
        print("Confusion matrix:")
        print(confusion_matrix(y_val, best_pred))
    else:
        print("Il migliore e' un esperto singolo: l'ensemble va raffinato prima della versione finale.")

    print(f"Diversita' esperti salvata in: {diversity_path}")
    if diverse_path is not None:
        print(f"Selezione esperti diversity-aware salvata in: {diverse_path}")
    print(f"Subset di feature salvati in: {feature_sets_path}")
    print(f"Risultati salvati in: {results_path}")
    return results_df


if __name__ == "__main__":
    main()
