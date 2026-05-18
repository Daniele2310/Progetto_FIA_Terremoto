from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


@dataclass
class SBSStep:
    step: int
    n_features_before: int
    n_features_after: int
    removed_feature: str
    score_before: float
    score_after: float
    delta_score: float


class SequentialBackwardSelector:
    """
    Sequential Backward Selection (SBS).

    Procedura top-down:
    - parte dal set completo delle feature;
    - a ogni iterazione rimuove la feature che massimizza la metrica;
    - termina quando rimuovere una feature peggiora la metrica oltre la tolleranza.

    Nota: effetto di nesting.
    Una feature rimossa non puo essere reinserita in seguito.
    """

    def __init__(
        self,
        estimator_name: str = "logreg",
        scoring: str = "accuracy",
        random_state: int = 42,
    ):
        estimator_name = estimator_name.lower().strip()
        if estimator_name not in {"logreg", "knn"}:
            raise ValueError("estimator_name deve essere 'logreg' oppure 'knn'.")

        scoring = scoring.lower().strip()
        if scoring not in {"accuracy", "f1_micro"}:
            raise ValueError("scoring deve essere 'accuracy' oppure 'f1_micro'.")

        self.estimator_name = estimator_name
        self.scoring = scoring
        self.random_state = random_state

    @staticmethod
    def _normalize_columns(columns: Optional[Iterable[str]]) -> list[str]:
        if columns is None:
            return []
        return list(columns)

    @staticmethod
    def _to_numeric_label(label: pd.Series) -> np.ndarray:
        if pd.api.types.is_numeric_dtype(label):
            return label.astype(int).to_numpy()
        codes, _ = pd.factorize(label, sort=True)
        return codes.astype(int)

    @staticmethod
    def _load_default_dataframe(project_root: Path) -> tuple[pd.DataFrame, str]:
        train_values_path = project_root / "Data" / "train_values.csv"
        train_labels_path = project_root / "Data" / "train_labels.csv"

        if train_values_path.exists() and train_labels_path.exists():
            train_values = pd.read_csv(train_values_path)
            train_labels = pd.read_csv(train_labels_path)
            merged = train_values.merge(train_labels, on="building_id", how="inner")
            return merged, f"{train_values_path} + {train_labels_path}"

        preprocessed_with_labels = project_root / "DataPreprocessed" / "processed" / "train_features_labels_preprocessed.csv"
        if preprocessed_with_labels.exists():
            return pd.read_csv(preprocessed_with_labels), str(preprocessed_with_labels)

        raise FileNotFoundError("Nessun dataset di default trovato. Usa --input per specificare un CSV.")

    @staticmethod
    def _prepare_features(
        df: pd.DataFrame,
        label_column: str,
        exclude_columns: list[str],
    ) -> tuple[pd.DataFrame, np.ndarray]:
        if label_column not in df.columns:
            raise ValueError(f"Colonna target non trovata: {label_column}")

        excluded = set(exclude_columns)
        excluded.add(label_column)

        feature_candidates = [col for col in df.columns if col not in excluded]
        if not feature_candidates:
            raise ValueError("Nessuna feature candidata disponibile.")

        x_raw = df[feature_candidates].copy()
        y = SequentialBackwardSelector._to_numeric_label(df[label_column])

        categorical_cols = [
            col
            for col in x_raw.columns
            if (
                pd.api.types.is_object_dtype(x_raw[col])
                or pd.api.types.is_string_dtype(x_raw[col])
                or isinstance(x_raw[col].dtype, pd.CategoricalDtype)
            )
        ]

        if categorical_cols:
            x_encoded = pd.get_dummies(x_raw, columns=categorical_cols, drop_first=False, dtype=float)
        else:
            x_encoded = x_raw.astype(float)

        if x_encoded.isnull().any().any():
            raise ValueError("SBS richiede input senza NaN: completa imputazione/pulizia prima dell'uso.")

        return x_encoded, y

    def _build_estimator(self):
        if self.estimator_name == "knn":
            return KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                algorithm="brute",
                n_jobs=-1,
            )

        return LogisticRegression(
            max_iter=1200,
            random_state=self.random_state,
        )

    def _score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.scoring == "f1_micro":
            return float(f1_score(y_true, y_pred, average="micro"))
        return float(accuracy_score(y_true, y_pred))

    def _evaluate_subset(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        feature_idx: np.ndarray,
    ) -> float:
        estimator = self._build_estimator()
        estimator.fit(x_train[:, feature_idx], y_train)
        y_pred = estimator.predict(x_val[:, feature_idx])
        return self._score(y_val, y_pred)

    @staticmethod
    def _theoretical_evaluations(initial_features: int, min_features: int) -> int:
        # 1 valutazione baseline + somma dei candidati testati a ogni step.
        if min_features >= initial_features:
            return 1
        upper = initial_features
        lower = min_features + 1
        candidates = (upper + lower) * (upper - lower + 1) // 2
        return int(1 + candidates)

    @staticmethod
    def _validate_history(history: list[SBSStep]) -> None:
        if not history:
            return

        removed_set: set[str] = set()
        prev_after = history[0].n_features_before
        for step in history:
            if step.removed_feature in removed_set:
                raise ValueError("History SBS non valida: feature rimossa piu volte.")
            removed_set.add(step.removed_feature)

            if step.n_features_before != prev_after:
                raise ValueError("History SBS non valida: cardinalita incoerente tra step.")
            if step.n_features_after != step.n_features_before - 1:
                raise ValueError("History SBS non valida: ogni step deve rimuovere una sola feature.")

            prev_after = step.n_features_after

    def select(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
        min_features: int = 10,
        test_size: float = 0.2,
        max_rows: Optional[int] = 30000,
        min_improvement: float = 0.0,
        max_steps: Optional[int] = None,
    ) -> dict[str, object]:
        if min_features <= 0:
            raise ValueError("min_features deve essere > 0")
        if not (0 < test_size < 1):
            raise ValueError("test_size deve stare in (0,1)")
        if max_rows is not None and max_rows <= 200:
            raise ValueError("max_rows deve essere > 200 oppure None")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps deve essere > 0 oppure None")

        x_work = x.copy()
        y_work = y.copy()

        sampling_applied = False
        sampled_rows = len(x_work)
        if max_rows is not None and len(x_work) > max_rows:
            x_work, _, y_work, _ = train_test_split(
                x_work,
                y_work,
                train_size=max_rows,
                random_state=self.random_state,
                stratify=y_work,
            )
            sampling_applied = True
            sampled_rows = len(x_work)

        x_train_df, x_val_df, y_train, y_val = train_test_split(
            x_work,
            y_work,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y_work,
        )

        feature_names = x_work.columns.to_numpy()
        x_train = x_train_df.to_numpy(dtype=float)
        x_val = x_val_df.to_numpy(dtype=float)

        total_features = x_train.shape[1]
        if min_features >= total_features:
            raise ValueError("min_features deve essere minore del numero di feature disponibili.")

        current_idx = np.arange(total_features)
        start = perf_counter()
        baseline_score = self._evaluate_subset(x_train, y_train, x_val, y_val, current_idx)
        evaluated_models = 1

        history: list[SBSStep] = []
        current_score = baseline_score
        stop_reason = "min_features_raggiunto"

        while len(current_idx) > min_features:
            if max_steps is not None and len(history) >= max_steps:
                stop_reason = "max_steps_raggiunto"
                break

            best_score = -np.inf
            best_idx: Optional[np.ndarray] = None
            best_removed_name = ""

            for feature_pos in range(len(current_idx)):
                candidate_idx = np.delete(current_idx, feature_pos)
                candidate_score = self._evaluate_subset(
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    candidate_idx,
                )
                evaluated_models += 1

                if candidate_score > best_score:
                    best_score = candidate_score
                    best_idx = candidate_idx
                    best_removed_name = str(feature_names[current_idx[feature_pos]])

            assert best_idx is not None

            # SBS standard: stop se la migliore rimozione peggiora la metrica.
            if best_score < current_score:
                stop_reason = "peggioramento_prestazioni"
                break

            # Soglia opzionale: richiede miglioramento minimo > 0.
            if best_score < (current_score + min_improvement):
                stop_reason = "miglioramento_insufficiente"
                break

            step = SBSStep(
                step=len(history) + 1,
                n_features_before=int(len(current_idx)),
                n_features_after=int(len(best_idx)),
                removed_feature=best_removed_name,
                score_before=float(current_score),
                score_after=float(best_score),
                delta_score=float(best_score - current_score),
            )
            history.append(step)
            current_idx = best_idx
            current_score = best_score

        elapsed_sec = perf_counter() - start
        self._validate_history(history)

        selected_features = feature_names[current_idx].tolist()
        removed_features = [step.removed_feature for step in history]

        theoretical_evals = self._theoretical_evaluations(
            initial_features=total_features,
            min_features=min_features,
        )

        summary = {
            "estimator": self.estimator_name,
            "scoring": self.scoring,
            "n_rows_original": int(len(x)),
            "n_rows_used": int(sampled_rows),
            "sampling_applied": bool(sampling_applied),
            "n_features_initial": int(total_features),
            "n_features_final": int(len(selected_features)),
            "min_features_target": int(min_features),
            "test_size": float(test_size),
            "baseline_score": float(baseline_score),
            "best_score_final": float(current_score),
            "delta_vs_baseline": float(current_score - baseline_score),
            "n_steps_executed": int(len(history)),
            "evaluated_models": int(evaluated_models),
            "theoretical_models_no_early_stop": int(theoretical_evals),
            "elapsed_seconds": float(elapsed_sec),
            "avg_seconds_per_model": float(elapsed_sec / evaluated_models),
            "stop_reason": stop_reason,
            "min_improvement": float(min_improvement),
            "max_steps": int(max_steps) if max_steps is not None else None,
            "random_state": int(self.random_state),
            "complexity_note": (
                "Costo elevato: per p feature, una run completa valuta circa O(p^2) modelli "
                "(oltre al costo del fit di ogni modello)."
            ),
        }

        history_df = pd.DataFrame([step.__dict__ for step in history])
        selected_df = pd.DataFrame(
            {
                "selected_feature": selected_features,
            }
        )
        removed_df = pd.DataFrame(
            {
                "removed_feature_order": np.arange(1, len(removed_features) + 1),
                "removed_feature": removed_features,
            }
        )

        return {
            "summary": summary,
            "history": history_df,
            "selected_features": selected_df,
            "removed_features": removed_df,
        }

    @staticmethod
    def plot_history(
        history_df: pd.DataFrame,
        baseline_score: float,
        output_path: Optional[Path] = None,
        show_plot: bool = False,
    ) -> None:
        if history_df.empty:
            return

        x_steps = [0] + history_df["step"].astype(int).tolist()
        y_scores = [baseline_score] + history_df["score_after"].astype(float).tolist()
        feature_counts = [
            int(history_df.iloc[0]["n_features_before"]),
            *history_df["n_features_after"].astype(int).tolist(),
        ]

        plt.figure(figsize=(10, 6))
        plt.plot(x_steps, y_scores, marker="o")
        plt.xlabel("Iterazione SBS")
        plt.ylabel("Score")
        plt.title("Andamento score durante SBS")
        plt.grid(alpha=0.2)

        for x_pos, y_pos, n_feat in zip(x_steps, y_scores, feature_counts):
            plt.annotate(f"{n_feat}f", (x_pos, y_pos), textcoords="offset points", xytext=(4, 4))

        plt.tight_layout()

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)

        if show_plot:
            plt.show()
        else:
            plt.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Subset selection supervisionata con Sequential Backward Selection (SBS). "
            "Partendo da tutte le feature, rimuove iterativamente quella meno utile."
        )
    )
    parser.add_argument("--input", type=Path, default=None, help="CSV input con feature + target.")
    parser.add_argument("--label-column", type=str, default="damage_grade", help="Nome colonna target.")
    parser.add_argument(
        "--exclude-columns",
        nargs="*",
        default=["building_id"],
        help="Colonne da escludere dalla selezione.",
    )
    parser.add_argument(
        "--estimator",
        type=str,
        default="logreg",
        choices=["logreg", "knn"],
        help="Modello usato per valutare i subset.",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="accuracy",
        choices=["accuracy", "f1_micro"],
        help="Metrica usata per scegliere la feature da rimuovere.",
    )
    parser.add_argument("--min-features", type=int, default=15, help="Numero minimo di feature da mantenere.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Quota del validation holdout.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=30000,
        help="Numero massimo di righe usate per contenere il costo computazionale.",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.0,
        help="Miglioramento minimo richiesto per accettare una rimozione.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Numero massimo di iterazioni SBS (utile per run esplorative).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Cartella di output per CSV/JSON/plot.",
    )
    parser.add_argument("--show-plots", action="store_true", help="Mostra plot oltre a salvarli.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    if args.input is not None:
        if not args.input.exists():
            raise FileNotFoundError(f"File non trovato: {args.input}")
        df = pd.read_csv(args.input)
        source_text = str(args.input)
    else:
        df, source_text = SequentialBackwardSelector._load_default_dataframe(project_root)

    selector = SequentialBackwardSelector(
        estimator_name=args.estimator,
        scoring=args.scoring,
        random_state=42,
    )

    x_encoded, y = selector._prepare_features(
        df=df,
        label_column=args.label_column,
        exclude_columns=selector._normalize_columns(args.exclude_columns),
    )

    results = selector.select(
        x=x_encoded,
        y=y,
        min_features=args.min_features,
        test_size=args.test_size,
        max_rows=args.max_rows,
        min_improvement=args.min_improvement,
        max_steps=args.max_steps,
    )

    summary = results["summary"]
    history = results["history"]
    selected = results["selected_features"]
    removed = results["removed_features"]

    assert isinstance(summary, dict)
    assert isinstance(history, pd.DataFrame)
    assert isinstance(selected, pd.DataFrame)
    assert isinstance(removed, pd.DataFrame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.output_dir / "sbs_history.csv", index=False)
    selected.to_csv(args.output_dir / "sbs_selected_features.csv", index=False)
    removed.to_csv(args.output_dir / "sbs_removed_features.csv", index=False)

    with open(args.output_dir / "sbs_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    selector.plot_history(
        history_df=history,
        baseline_score=float(summary["baseline_score"]),
        output_path=args.output_dir / "sbs_score_history.png",
        show_plot=args.show_plots,
    )

    print("\n" + "=" * 80)
    print("SUBSET SELECTION - SBS COMPLETATA")
    print("=" * 80)
    print(f"Sorgente dati: {source_text}")
    print(f"Estimator: {summary['estimator']}")
    print(f"Scoring: {summary['scoring']}")
    print(f"Feature iniziali/finali: {summary['n_features_initial']} -> {summary['n_features_final']}")
    print(f"Score baseline: {summary['baseline_score']:.6f}")
    print(f"Score finale: {summary['best_score_final']:.6f}")
    print(f"Delta: {summary['delta_vs_baseline']:.6f}")
    print(f"Modelli valutati: {summary['evaluated_models']}")
    print(f"Modelli teorici senza early-stop: {summary['theoretical_models_no_early_stop']}")
    print(f"Tempo totale (s): {summary['elapsed_seconds']:.2f}")
    print(f"Stop reason: {summary['stop_reason']}")
    print(f"Output salvato in: {args.output_dir.resolve()}")

    print("\nTop 15 feature selezionate:")
    print(selected.head(15).to_string(index=False))

    if not history.empty:
        print("\nUltimi 10 step SBS:")
        print(history.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
