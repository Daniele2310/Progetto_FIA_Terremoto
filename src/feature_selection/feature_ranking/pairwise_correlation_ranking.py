from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PairwiseCorrelationRanker:
    """
    Ranker basato su correlazione pair-wise tra feature.

    - Analisi non supervisionata: dipendenza tra una feature e tutte le altre.
    - Analisi supervisionata (opzionale): correlazione tra feature e label.
    """

    def __init__(self, method: str = "pearson"):
        if method != "pearson":
            raise ValueError("Questo ranker supporta solo il metodo 'pearson'.")
        self.method = "pearson"

    @staticmethod
    def _normalize_columns(columns: Optional[Iterable[str]]) -> list[str]:
        if columns is None:
            return []
        return list(columns)

    @staticmethod
    def _numeric_label(label: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(label):
            return label.astype(float)

        codes, _ = pd.factorize(label, sort=True)
        return pd.Series(codes, index=label.index, dtype=float)

    @staticmethod
    def _validate_top_n(top_n: int) -> int:
        if top_n <= 0:
            raise ValueError("top_n deve essere > 0")
        return top_n

    @staticmethod
    def top_correlated_pairs(corr_matrix: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        top_n = PairwiseCorrelationRanker._validate_top_n(top_n)

        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        pairs = (
            corr_matrix.where(upper_triangle)
            .stack()
            .reset_index()
            .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "corr"})
        )

        if pairs.empty:
            return pd.DataFrame(columns=["feature_1", "feature_2", "corr", "abs_corr"])

        pairs["abs_corr"] = pairs["corr"].abs()
        return pairs.sort_values(by="abs_corr", ascending=False, kind="stable").head(top_n).reset_index(drop=True)

    @staticmethod
    def plot_top_pairwise_correlations(
        corr_matrix: pd.DataFrame,
        top_n: int = 20,
        output_path: Optional[Path] = None,
        show_plot: bool = False,
    ) -> pd.DataFrame:
        top_pairs = PairwiseCorrelationRanker.top_correlated_pairs(corr_matrix, top_n=top_n)
        if top_pairs.empty:
            return top_pairs

        labels = top_pairs.apply(lambda row: f"{row['feature_1']} vs {row['feature_2']}", axis=1)

        plt.figure(figsize=(12, max(6, int(len(top_pairs) * 0.35))))
        colors = ["#d73027" if val < 0 else "#1a9850" for val in top_pairs["corr"]]
        plt.barh(labels[::-1], top_pairs["corr"].values[::-1], color=colors[::-1])
        plt.xlabel("Correlazione")
        plt.ylabel("Coppie di feature")
        plt.title(f"Top {len(top_pairs)} coppie piu correlate")
        plt.grid(axis="x", alpha=0.2)
        plt.tight_layout()

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)

        if show_plot:
            plt.show()
        else:
            plt.close()

        return top_pairs

    @staticmethod
    def plot_top_target_correlations(
        supervised_ranking: pd.DataFrame,
        top_n: int = 20,
        output_path: Optional[Path] = None,
        show_plot: bool = False,
    ) -> pd.DataFrame:
        top_n = PairwiseCorrelationRanker._validate_top_n(top_n)

        if "corr_with_label" not in supervised_ranking.columns:
            raise ValueError("supervised_ranking deve contenere la colonna 'corr_with_label'.")

        top_target = supervised_ranking.head(top_n).copy()
        if top_target.empty:
            return top_target

        plt.figure(figsize=(12, max(6, int(len(top_target) * 0.35))))
        colors = ["#d73027" if val < 0 else "#1a9850" for val in top_target["corr_with_label"]]
        plt.barh(top_target["feature"][::-1], top_target["corr_with_label"].values[::-1], color=colors[::-1])
        plt.xlabel("Correlazione con il target")
        plt.ylabel("Feature")
        plt.title(f"Top {len(top_target)} feature correlate con il target")
        plt.grid(axis="x", alpha=0.2)
        plt.tight_layout()

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)

        if show_plot:
            plt.show()
        else:
            plt.close()

        return top_target

    def rank(
        self,
        df: pd.DataFrame,
        label_column: Optional[str] = None,
        exclude_columns: Optional[Iterable[str]] = None,
    ) -> dict[str, pd.DataFrame]:
        exclude_columns = self._normalize_columns(exclude_columns)
        excluded = set(exclude_columns)

        if label_column and label_column in df.columns:
            excluded.add(label_column)

        feature_candidates = [col for col in df.columns if col not in excluded]
        numeric_features = [col for col in feature_candidates if pd.api.types.is_numeric_dtype(df[col])]

        if len(numeric_features) < 2:
            raise ValueError(
                "Servono almeno 2 feature numeriche per la correlazione pair-wise."
            )

        X = df[numeric_features].copy()
        corr_matrix = X.corr(method=self.method)

        # Evita assegnazioni in-place su array potenzialmente read-only.
        abs_corr = corr_matrix.abs().mask(np.eye(len(corr_matrix), dtype=bool))

        mean_abs_corr = abs_corr.mean(skipna=True)
        max_abs_corr = abs_corr.max(skipna=True)

        pairwise_ranking = (
            pd.DataFrame(
                {
                    "feature": mean_abs_corr.index,
                    "mean_abs_corr_with_other_features": mean_abs_corr.values,
                    "max_abs_corr_with_other_features": max_abs_corr.values,
                }
            )
            .sort_values(
                by=["mean_abs_corr_with_other_features", "max_abs_corr_with_other_features"],
                ascending=False,
                kind="stable",
            )
            .reset_index(drop=True)
        )

        results: dict[str, pd.DataFrame] = {
            "correlation_matrix": corr_matrix,
            "pairwise_ranking": pairwise_ranking,
        }

        if label_column and label_column in df.columns:
            y = self._numeric_label(df[label_column])
            corr_with_label = X.apply(lambda col: col.corr(y, method=self.method))

            supervised_ranking = (
                pd.DataFrame(
                    {
                        "feature": corr_with_label.index,
                        "corr_with_label": corr_with_label.values,
                        "abs_corr_with_label": corr_with_label.abs().values,
                    }
                )
                .sort_values(by="abs_corr_with_label", ascending=False, kind="stable")
                .reset_index(drop=True)
            )

            combined_ranking = pairwise_ranking.merge(
                supervised_ranking,
                on="feature",
                how="left",
            )
            combined_ranking["balance_score"] = (
                combined_ranking["abs_corr_with_label"]
                - combined_ranking["mean_abs_corr_with_other_features"]
            )
            combined_ranking = combined_ranking.sort_values(
                by="balance_score",
                ascending=False,
                kind="stable",
            ).reset_index(drop=True)

            results["supervised_ranking"] = supervised_ranking
            results["combined_ranking"] = combined_ranking

        return results


def _load_default_dataframe(project_root: Path) -> tuple[pd.DataFrame, str]:
    train_values_path = project_root / "Data" / "train_values.csv"
    train_labels_path = project_root / "Data" / "train_labels.csv"

    if train_values_path.exists() and train_labels_path.exists():
        train_values = pd.read_csv(train_values_path)
        train_labels = pd.read_csv(train_labels_path)
        merged = train_values.merge(train_labels, on="building_id", how="inner")
        return merged, f"{train_values_path} + {train_labels_path}"

    preprocessed_with_labels = (
        project_root / "Data" / "preprocessed" / "train_features_labels_preprocessed.csv"
    )
    if preprocessed_with_labels.exists():
        return pd.read_csv(preprocessed_with_labels), str(preprocessed_with_labels)

    raise FileNotFoundError(
        "Nessun dataset di default trovato. Usa --input per specificare un CSV."
    )


def _prepare_dataframe_for_correlation(
    df: pd.DataFrame,
    label_column: Optional[str],
    exclude_columns: list[str],
    skip_one_hot: bool,
) -> tuple[pd.DataFrame, list[str]]:
    if skip_one_hot:
        return df.copy(), []

    excluded = set(exclude_columns)
    if label_column:
        excluded.add(label_column)

    categorical_cols = [
        col
        for col in df.columns
        if col not in excluded
        and (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
            or isinstance(df[col].dtype, pd.CategoricalDtype)
        )
    ]

    if not categorical_cols:
        return df.copy(), []

    encoded_df = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=float)
    return encoded_df, categorical_cols


def _save_results(results: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    name_map = {
        "correlation_matrix": "correlation_matrix.csv",
        "pairwise_ranking": "pairwise_ranking.csv",
        "supervised_ranking": "supervised_ranking.csv",
        "combined_ranking": "combined_ranking.csv",
    }

    for key, dataframe in results.items():
        if key in name_map:
            dataframe.to_csv(output_dir / name_map[key], index=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Feature ranking pair-wise con correlazione lineare su feature originali (no PCA). "
            "Nei casi supervisionati calcola anche la correlazione feature-label."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="CSV di input con feature (ed eventualmente label).",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="damage_grade",
        help="Nome della colonna etichetta per analisi supervisionata.",
    )
    parser.add_argument(
        "--exclude-columns",
        nargs="*",
        default=["building_id"],
        help="Colonne da escludere dal ranking delle feature.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Cartella di output per i CSV risultanti.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Numero di elementi da mostrare nei plot e nei top ranking.",
    )
    parser.add_argument(
        "--skip-one-hot",
        action="store_true",
        help="Non applicare one-hot encoding automatico alle colonne categoriche.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Mostra i plot oltre a salvarli su file.",
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

    ranker = PairwiseCorrelationRanker()

    label_column = args.label_column if args.label_column in df.columns else None
    if label_column is None:
        print(
            f"Label '{args.label_column}' non trovata: eseguo solo ranking non supervisionato."
        )

    prepared_df, encoded_cols = _prepare_dataframe_for_correlation(
        df=df,
        label_column=label_column,
        exclude_columns=args.exclude_columns,
        skip_one_hot=args.skip_one_hot,
    )

    if encoded_cols:
        print(
            f"Applicato one-hot encoding su {len(encoded_cols)} colonne categoriche: {', '.join(encoded_cols)}"
        )

    results = ranker.rank(
        df=prepared_df,
        label_column=label_column,
        exclude_columns=args.exclude_columns,
    )

    _save_results(results, args.output_dir)

    top_pairs = ranker.plot_top_pairwise_correlations(
        corr_matrix=results["correlation_matrix"],
        top_n=args.top_n,
        output_path=args.output_dir / "top_pairwise_correlations.png",
        show_plot=args.show_plots,
    )
    top_pairs.to_csv(args.output_dir / "top_pairwise_correlated_pairs.csv", index=False)

    top_target = None
    if "supervised_ranking" in results:
        top_target = ranker.plot_top_target_correlations(
            supervised_ranking=results["supervised_ranking"],
            top_n=args.top_n,
            output_path=args.output_dir / "top_target_correlations.png",
            show_plot=args.show_plots,
        )
        top_target.to_csv(args.output_dir / "top_target_correlations.csv", index=False)

    print("\n" + "=" * 80)
    print("FEATURE RANKING PAIR-WISE COMPLETATO")
    print("=" * 80)
    print(f"Sorgente dati: {source_text}")
    print("Metodo correlazione: pearson")
    print(f"Output salvato in: {args.output_dir.resolve()}")

    print("\nTop 10 pairwise_ranking:")
    print(results["pairwise_ranking"].head(10).to_string(index=False))

    print("\nTop coppie correlate:")
    print(top_pairs.head(10).to_string(index=False))

    if "supervised_ranking" in results:
        print("\nTop 10 supervised_ranking:")
        print(results["supervised_ranking"].head(10).to_string(index=False))
        if top_target is not None:
            print("\nTop feature correlate al target:")
            print(top_target.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
