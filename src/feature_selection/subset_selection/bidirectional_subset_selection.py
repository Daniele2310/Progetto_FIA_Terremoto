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
class BidirectionalStep:
    cycle: int
    phase: str
    n_features_before: int
    n_features_after: int
    feature_changed: str
    score_before: float
    score_after: float
    delta_score: float


class StepwiseBidirectionalSelector:
    """
    Stepwise Bidirectional Selection.

    Alterna due fasi per ogni ciclo:
    - Forward step: aggiunge la feature che massimizza lo score.
    - Backward step: valuta la rimozione di feature gia selezionate.

    Per allineamento alla logica mostrata a lezione:
    - aggiunta/rimozione viene accettata solo se migliora lo score
      oltre la soglia min_improvement;
    - se un intero ciclo non produce miglioramenti, l'algoritmo termina.
    """

    def __init__(
        self,
        estimator_name: str = "knn",
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

        preprocessed_with_labels = project_root / "Data" / "preprocessed" / "train_features_labels_preprocessed.csv"
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
        y = StepwiseBidirectionalSelector._to_numeric_label(df[label_column])

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
            raise ValueError("Bidirectional selection richiede input senza NaN.")

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

    def select(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
        min_features: int = 1,
        max_features: Optional[int] = None,
        test_size: float = 0.2,
        max_rows: Optional[int] = 30000,
        min_improvement: float = 0.0,
        max_cycles: Optional[int] = None,
    ) -> dict[str, object]:
        if min_features <= 0:
            raise ValueError("min_features deve essere > 0")
        if not (0 < test_size < 1):
            raise ValueError("test_size deve stare in (0,1)")
        if max_rows is not None and max_rows <= 200:
            raise ValueError("max_rows deve essere > 200 oppure None")
        if max_cycles is not None and max_cycles <= 0:
            raise ValueError("max_cycles deve essere > 0 oppure None")

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

        if max_features is None:
            max_features = total_features

        if not (min_features <= max_features <= total_features):
            raise ValueError("Richiesto min_features <= max_features <= n_features_totali.")

        all_idx = np.arange(total_features, dtype=int)
        current_idx = np.array([], dtype=int)

        start = perf_counter()
        evaluated_models = 0
        history: list[BidirectionalStep] = []
        cycle = 0
        current_score = float("-inf")
        stop_reason = "max_features_raggiunto"

        while True:
            cycle += 1
            if max_cycles is not None and cycle > max_cycles:
                stop_reason = "max_cycles_raggiunto"
                break

            improved_in_cycle = False

            # FORWARD STEP
            if len(current_idx) < max_features:
                forward_best_score = float("-inf")
                forward_best_idx: Optional[np.ndarray] = None
                forward_best_feature = ""

                remaining = np.setdiff1d(all_idx, current_idx)
                for idx_add in remaining:
                    candidate_idx = np.sort(np.append(current_idx, idx_add))
                    score = self._evaluate_subset(x_train, y_train, x_val, y_val, candidate_idx)
                    evaluated_models += 1
                    if score > forward_best_score:
                        forward_best_score = score
                        forward_best_idx = candidate_idx
                        forward_best_feature = str(feature_names[idx_add])

                assert forward_best_idx is not None

                accept_forward = False
                if len(current_idx) == 0:
                    accept_forward = True
                elif forward_best_score > current_score + min_improvement:
                    accept_forward = True

                if accept_forward:
                    before = current_score if np.isfinite(current_score) else forward_best_score
                    delta = forward_best_score - before if np.isfinite(current_score) else 0.0
                    history.append(
                        BidirectionalStep(
                            cycle=cycle,
                            phase="forward",
                            n_features_before=int(len(current_idx)),
                            n_features_after=int(len(forward_best_idx)),
                            feature_changed=forward_best_feature,
                            score_before=float(before),
                            score_after=float(forward_best_score),
                            delta_score=float(delta),
                        )
                    )
                    current_idx = forward_best_idx
                    current_score = float(forward_best_score)
                    improved_in_cycle = True

            # BACKWARD STEP
            if len(current_idx) > min_features:
                backward_best_score = float("-inf")
                backward_best_idx: Optional[np.ndarray] = None
                backward_best_feature = ""

                for pos in range(len(current_idx)):
                    idx_remove = current_idx[pos]
                    candidate_idx = np.delete(current_idx, pos)
                    score = self._evaluate_subset(x_train, y_train, x_val, y_val, candidate_idx)
                    evaluated_models += 1
                    if score > backward_best_score:
                        backward_best_score = score
                        backward_best_idx = candidate_idx
                        backward_best_feature = str(feature_names[idx_remove])

                assert backward_best_idx is not None

                # Logica richiesta a lezione: rimuovi solo se migliora.
                if backward_best_score > current_score + min_improvement:
                    history.append(
                        BidirectionalStep(
                            cycle=cycle,
                            phase="backward",
                            n_features_before=int(len(current_idx)),
                            n_features_after=int(len(backward_best_idx)),
                            feature_changed=backward_best_feature,
                            score_before=float(current_score),
                            score_after=float(backward_best_score),
                            delta_score=float(backward_best_score - current_score),
                        )
                    )
                    current_idx = backward_best_idx
                    current_score = float(backward_best_score)
                    improved_in_cycle = True

            if not improved_in_cycle:
                stop_reason = "nessun_miglioramento"
                break

            if len(current_idx) >= max_features:
                stop_reason = "max_features_raggiunto"
                break

        elapsed_sec = perf_counter() - start

        selected_features = feature_names[current_idx].tolist()
        history_df = pd.DataFrame([step.__dict__ for step in history])
        selected_df = pd.DataFrame({"selected_feature": selected_features})

        summary = {
            "estimator": self.estimator_name,
            "scoring": self.scoring,
            "n_rows_original": int(len(x)),
            "n_rows_used": int(sampled_rows),
            "sampling_applied": bool(sampling_applied),
            "n_features_initial": int(total_features),
            "n_features_final": int(len(selected_features)),
            "min_features": int(min_features),
            "max_features": int(max_features),
            "test_size": float(test_size),
            "best_score_final": float(current_score),
            "n_cycles_executed": int(cycle),
            "n_steps_accepted": int(len(history_df)),
            "evaluated_models": int(evaluated_models),
            "elapsed_seconds": float(elapsed_sec),
            "avg_seconds_per_model": float(elapsed_sec / max(1, evaluated_models)),
            "stop_reason": stop_reason,
            "min_improvement": float(min_improvement),
            "max_cycles": int(max_cycles) if max_cycles is not None else None,
            "random_state": int(self.random_state),
            "complexity_note": (
                "Il metodo bidirezionale e piu costoso di SFS/SBS: valuta numerosi candidati "
                "forward e backward a ogni ciclo."
            ),
        }

        return {
            "summary": summary,
            "history": history_df,
            "selected_features": selected_df,
        }

    @staticmethod
    def plot_history(
        history_df: pd.DataFrame,
        output_path: Optional[Path] = None,
        show_plot: bool = False,
    ) -> None:
        if history_df.empty:
            return

        plt.figure(figsize=(10, 6))
        x = np.arange(1, len(history_df) + 1)
        y = history_df["score_after"].to_numpy(dtype=float)
        colors = ["#1f77b4" if ph == "forward" else "#d62728" for ph in history_df["phase"]]

        plt.plot(x, y, color="#666666", linewidth=1)
        plt.scatter(x, y, c=colors)
        plt.xlabel("Step accettato")
        plt.ylabel("Score")
        plt.title("Andamento score - Stepwise Bidirectional")
        plt.grid(alpha=0.2)

        for i, (xx, yy, ph) in enumerate(zip(x, y, history_df["phase"])):
            label = "F" if ph == "forward" else "B"
            plt.annotate(label, (xx, yy), textcoords="offset points", xytext=(4, 4))

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
            "Subset selection con Stepwise Bidirectional Selection (forward + backward alternati)."
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
        default="knn",
        choices=["logreg", "knn"],
        help="Modello usato per valutare i subset.",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="accuracy",
        choices=["accuracy", "f1_micro"],
        help="Metrica usata per la selezione.",
    )
    parser.add_argument("--min-features", type=int, default=1, help="Numero minimo di feature da mantenere.")
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Numero massimo di feature consentite nel subset.",
    )
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
        help="Miglioramento minimo richiesto per accettare uno step.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Numero massimo di cicli bidirezionali.",
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
        df, source_text = StepwiseBidirectionalSelector._load_default_dataframe(project_root)

    selector = StepwiseBidirectionalSelector(
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
        max_features=args.max_features,
        test_size=args.test_size,
        max_rows=args.max_rows,
        min_improvement=args.min_improvement,
        max_cycles=args.max_cycles,
    )

    summary = results["summary"]
    history = results["history"]
    selected = results["selected_features"]

    assert isinstance(summary, dict)
    assert isinstance(history, pd.DataFrame)
    assert isinstance(selected, pd.DataFrame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.output_dir / "bidirectional_history.csv", index=False)
    selected.to_csv(args.output_dir / "bidirectional_selected_features.csv", index=False)
    with open(args.output_dir / "bidirectional_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    selector.plot_history(
        history_df=history,
        output_path=args.output_dir / "bidirectional_score_history.png",
        show_plot=args.show_plots,
    )

    print("\n" + "=" * 80)
    print("SUBSET SELECTION - STEPWISE BIDIRECTIONAL COMPLETATA")
    print("=" * 80)
    print(f"Sorgente dati: {source_text}")
    print(f"Estimator: {summary['estimator']}")
    print(f"Scoring: {summary['scoring']}")
    print(f"Feature iniziali/finali: {summary['n_features_initial']} -> {summary['n_features_final']}")
    print(f"Score finale: {summary['best_score_final']:.6f}")
    print(f"Modelli valutati: {summary['evaluated_models']}")
    print(f"Tempo totale (s): {summary['elapsed_seconds']:.2f}")
    print(f"Stop reason: {summary['stop_reason']}")
    print(f"Output salvato in: {args.output_dir.resolve()}")

    print("\nTop 15 feature selezionate:")
    print(selected.head(15).to_string(index=False))

    if not history.empty:
        print("\nUltimi 10 step accettati:")
        print(history.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
