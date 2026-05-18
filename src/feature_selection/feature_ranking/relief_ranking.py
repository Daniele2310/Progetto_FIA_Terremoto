from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class ReliefRanker:
    """
    Feature ranking supervisionato basato su Relief.

    Idea: per ogni campione x si cerca:
    - near hit: vicino piu vicino della stessa classe
    - near miss: vicino piu vicino di classe diversa

    Aggiornamento (media su m iterazioni):
    W_i = W_i - (x_i - near_hit_i)^2 + (x_i - near_miss_i)^2
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    @staticmethod
    def _normalize_columns(columns: Optional[Iterable[str]]) -> list[str]:
        if columns is None:
            return []
        return list(columns)

    @staticmethod
    def _to_numeric_label(label: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(label):
            return label.astype(int)
        codes, _ = pd.factorize(label, sort=True)
        return pd.Series(codes, index=label.index, dtype=int)

    @staticmethod
    def _minmax_scale(df: pd.DataFrame) -> pd.DataFrame:
        min_vals = df.min(axis=0)
        max_vals = df.max(axis=0)
        ranges = (max_vals - min_vals).replace(0, 1.0)
        return (df - min_vals) / ranges

    @staticmethod
    def _find_near_hit_index(
        neighbors_idx: np.ndarray,
        sample_index: int,
        y: np.ndarray,
    ) -> int:
        current_label = y[sample_index]
        for idx in neighbors_idx:
            if idx == sample_index:
                continue
            if y[idx] == current_label:
                return int(idx)
        return -1

    @staticmethod
    def _find_near_miss_index(
        neighbors_idx: np.ndarray,
        sample_index: int,
        y: np.ndarray,
    ) -> int:
        current_label = y[sample_index]
        for idx in neighbors_idx:
            if idx == sample_index:
                continue
            if y[idx] != current_label:
                return int(idx)
        return -1

    def rank(
        self,
        df: pd.DataFrame,
        label_column: str,
        exclude_columns: Optional[Iterable[str]] = None,
        n_iterations: int = 5000,
        n_neighbors_search: int = 50,
    ) -> dict[str, pd.DataFrame | dict[str, int | float]]:
        if label_column not in df.columns:
            raise ValueError(f"Colonna target non trovata: {label_column}")

        if n_iterations <= 0:
            raise ValueError("n_iterations deve essere > 0")

        exclude_columns = self._normalize_columns(exclude_columns)
        excluded = set(exclude_columns)
        excluded.add(label_column)

        feature_candidates = [col for col in df.columns if col not in excluded]
        if not feature_candidates:
            raise ValueError("Nessuna feature candidata disponibile.")

        X_raw = df[feature_candidates].copy()
        y = self._to_numeric_label(df[label_column]).to_numpy()

        categorical_cols = [
            col
            for col in X_raw.columns
            if (
                pd.api.types.is_object_dtype(X_raw[col])
                or pd.api.types.is_string_dtype(X_raw[col])
                or isinstance(X_raw[col].dtype, pd.CategoricalDtype)
            )
        ]

        if categorical_cols:
            X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=False, dtype=float)
        else:
            X_encoded = X_raw.astype(float)

        if X_encoded.isnull().any().any():
            raise ValueError("Il Relief richiede input senza NaN: completa prima imputazione/pulizia.")

        class_counts = pd.Series(y).value_counts()
        if (class_counts < 2).any():
            raise ValueError("Ogni classe deve avere almeno 2 campioni per trovare il near hit.")

        X = self._minmax_scale(X_encoded).to_numpy(dtype=float)
        n_samples, n_features = X.shape

        # Limite pratico: almeno un near hit e un near miss tra i vicini esplorati.
        k = max(3, min(n_neighbors_search, n_samples))
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
        nn.fit(X)
        neighbor_indices = nn.kneighbors(X, return_distance=False)

        rng = np.random.default_rng(self.random_state)
        sampled_indices = rng.integers(0, n_samples, size=n_iterations)
        weights = np.zeros(n_features, dtype=float)

        valid_updates = 0
        for i in sampled_indices:
            near_hit_idx = self._find_near_hit_index(neighbor_indices[i], i, y)
            near_miss_idx = self._find_near_miss_index(neighbor_indices[i], i, y)

            if near_hit_idx == -1 or near_miss_idx == -1:
                continue

            diff_hit = X[i] - X[near_hit_idx]
            diff_miss = X[i] - X[near_miss_idx]
            weights += -(diff_hit * diff_hit) + (diff_miss * diff_miss)
            valid_updates += 1

        if valid_updates == 0:
            raise ValueError(
                "Nessun aggiornamento Relief valido: aumenta n_neighbors_search o controlla il dataset."
            )

        weights = weights / float(valid_updates)
        ranking = pd.DataFrame(
            {
                "feature": X_encoded.columns,
                "relief_weight": weights,
                "abs_relief_weight": np.abs(weights),
            }
        ).sort_values(by="relief_weight", ascending=False, kind="stable").reset_index(drop=True)

        summary = {
            "label_column": label_column,
            "n_samples": int(n_samples),
            "n_features_after_encoding": int(n_features),
            "n_iterations_requested": int(n_iterations),
            "n_valid_updates": int(valid_updates),
            "n_neighbors_search": int(k),
            "random_state": int(self.random_state),
        }

        return {
            "relief_ranking": ranking,
            "summary": pd.DataFrame([summary]),
        }

    @staticmethod
    def plot_top_relief(
        ranking: pd.DataFrame,
        top_n: int = 20,
        output_path: Optional[Path] = None,
        show_plot: bool = False,
    ) -> pd.DataFrame:
        if top_n <= 0:
            raise ValueError("top_n deve essere > 0")

        top_df = ranking.head(top_n).copy()
        if top_df.empty:
            return top_df

        plt.figure(figsize=(12, max(6, int(len(top_df) * 0.35))))
        plt.barh(top_df["feature"][::-1], top_df["relief_weight"].values[::-1], color="#1f77b4")
        plt.xlabel("Relief weight")
        plt.ylabel("Feature")
        plt.title(f"Top {len(top_df)} feature per Relief")
        plt.grid(axis="x", alpha=0.2)
        plt.tight_layout()

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)

        if show_plot:
            plt.show()
        else:
            plt.close()

        return top_df


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Feature ranking supervisionato con Relief su feature originali "
            "(con one-hot automatico sulle categoriche)."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="CSV di input con feature ed etichetta.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="damage_grade",
        help="Nome della colonna etichetta (target).",
    )
    parser.add_argument(
        "--exclude-columns",
        nargs="*",
        default=["building_id"],
        help="Colonne da escludere dal ranking.",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=5000,
        help="Numero di campioni estratti casualmente per aggiornare i pesi.",
    )
    parser.add_argument(
        "--n-neighbors-search",
        type=int,
        default=50,
        help="Numero di vicini tra cui cercare near hit e near miss.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Numero di feature da mostrare in top output e plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Cartella di output per file CSV e grafico.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Mostra il plot oltre a salvarlo su file.",
    )
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
        df, source_text = _load_default_dataframe(project_root)

    ranker = ReliefRanker(random_state=42)
    results = ranker.rank(
        df=df,
        label_column=args.label_column,
        exclude_columns=args.exclude_columns,
        n_iterations=args.n_iterations,
        n_neighbors_search=args.n_neighbors_search,
    )

    ranking = results["relief_ranking"]
    summary = results["summary"]
    assert isinstance(ranking, pd.DataFrame)
    assert isinstance(summary, pd.DataFrame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.output_dir / "relief_ranking.csv", index=False)
    summary.to_csv(args.output_dir / "relief_summary.csv", index=False)

    top_df = ranker.plot_top_relief(
        ranking=ranking,
        top_n=args.top_n,
        output_path=args.output_dir / "relief_top_features.png",
        show_plot=args.show_plots,
    )
    top_df.to_csv(args.output_dir / "relief_top_features.csv", index=False)

    print("\n" + "=" * 80)
    print("FEATURE RANKING - RELIEF")
    print("=" * 80)
    print(f"Sorgente dati: {source_text}")
    print(f"Target: {args.label_column}")
    print(f"Output salvato in: {args.output_dir.resolve()}")
    print("\nRiepilogo run:")
    print(summary.to_string(index=False))
    print("\nTop 10 feature per Relief:")
    print(ranking.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
