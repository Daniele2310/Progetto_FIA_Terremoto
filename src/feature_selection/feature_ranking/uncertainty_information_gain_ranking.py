from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd


class InformationGainRanker:
    """
    Ranker supervisionato basato su incertezza (entropia) e Information Gain.

    H(Y) = - sum_k p(y_k) * log2(p(y_k))
    IG(Y, X) = H(Y) - H(Y|X)

    Il metodo e applicato solo a feature discrete.
    """

    def __init__(self, log_base: float = 2.0):
        if log_base <= 1:
            raise ValueError("log_base deve essere > 1")
        self.log_base = float(log_base)

    @staticmethod
    def _normalize_columns(columns: Optional[Iterable[str]]) -> list[str]:
        if columns is None:
            return []
        return list(columns)

    @staticmethod
    def _is_discrete_feature(series: pd.Series) -> bool:
        return (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_integer_dtype(series)
        )

    def _entropy(self, series: pd.Series) -> float:
        values = series.dropna()
        if values.empty:
            return 0.0

        probs = values.value_counts(normalize=True)
        log_probs = probs.apply(lambda p: math.log(p, self.log_base))
        entropy = float(-(probs * log_probs).sum())
        return entropy

    def _conditional_entropy(self, target: pd.Series, feature: pd.Series) -> float:
        pair = pd.DataFrame({"target": target, "feature": feature}).dropna()
        if pair.empty:
            return 0.0

        n_total = len(pair)
        entropy_sum = 0.0

        for _, group in pair.groupby("feature", dropna=False):
            p_x = len(group) / n_total
            h_y_given_x = self._entropy(group["target"])
            entropy_sum += p_x * h_y_given_x

        return float(entropy_sum)

    def information_gain(self, target: pd.Series, feature: pd.Series) -> dict[str, float | int]:
        pair = pd.DataFrame({"target": target, "feature": feature}).dropna()
        n_samples = int(len(pair))
        if n_samples == 0:
            return {
                "target_entropy": 0.0,
                "conditional_entropy": 0.0,
                "information_gain": 0.0,
                "n_samples": 0,
                "n_unique_values": 0,
            }

        y = pair["target"]
        x = pair["feature"]

        h_y = self._entropy(y)
        h_y_given_x = self._conditional_entropy(y, x)
        ig = h_y - h_y_given_x

        tolerance = 1e-10
        if ig < 0 and ig > -tolerance:
            ig = 0.0

        if ig < -tolerance:
            raise ValueError("Information Gain negativo oltre la tolleranza numerica.")
        if ig - h_y > tolerance:
            raise ValueError("Information Gain maggiore di H(Y), risultato non valido.")

        return {
            "target_entropy": float(h_y),
            "conditional_entropy": float(h_y_given_x),
            "information_gain": float(ig),
            "n_samples": n_samples,
            "n_unique_values": int(x.nunique(dropna=True)),
        }

    def rank(
        self,
        df: pd.DataFrame,
        label_column: str,
        exclude_columns: Optional[Iterable[str]] = None,
    ) -> dict[str, pd.DataFrame | dict]:
        if label_column not in df.columns:
            raise ValueError(f"Colonna target non trovata: {label_column}")

        exclude_columns = self._normalize_columns(exclude_columns)
        excluded = set(exclude_columns)
        excluded.add(label_column)

        feature_candidates = [col for col in df.columns if col not in excluded]
        discrete_features = [col for col in feature_candidates if self._is_discrete_feature(df[col])]
        skipped_features = [col for col in feature_candidates if col not in discrete_features]

        if not discrete_features:
            raise ValueError("Nessuna feature discreta disponibile per Information Gain.")

        rows = []
        target = df[label_column]

        for feature_name in discrete_features:
            stats = self.information_gain(target=target, feature=df[feature_name])
            rows.append(
                {
                    "feature": feature_name,
                    "target_entropy": stats["target_entropy"],
                    "conditional_entropy": stats["conditional_entropy"],
                    "information_gain": stats["information_gain"],
                    "n_samples": stats["n_samples"],
                    "n_unique_values": stats["n_unique_values"],
                }
            )

        ranking = (
            pd.DataFrame(rows)
            .sort_values(by="information_gain", ascending=False, kind="stable")
            .reset_index(drop=True)
        )

        summary = {
            "label_column": label_column,
            "n_feature_candidates": len(feature_candidates),
            "n_discrete_features": len(discrete_features),
            "n_skipped_non_discrete": len(skipped_features),
            "target_entropy_global": float(self._entropy(target.dropna())),
            "log_base": self.log_base,
        }

        skipped_df = pd.DataFrame({"skipped_feature": skipped_features})

        return {
            "information_gain_ranking": ranking,
            "skipped_features": skipped_df,
            "summary": summary,
        }

    @staticmethod
    def plot_top_information_gain(
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
        plt.barh(top_df["feature"][::-1], top_df["information_gain"].values[::-1], color="#1f77b4")
        plt.xlabel("Information Gain")
        plt.ylabel("Feature")
        plt.title(f"Top {len(top_df)} feature per Information Gain")
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


def _save_results(results: dict[str, pd.DataFrame | dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    ranking = results["information_gain_ranking"]
    skipped = results["skipped_features"]
    summary = results["summary"]

    assert isinstance(ranking, pd.DataFrame)
    assert isinstance(skipped, pd.DataFrame)
    assert isinstance(summary, dict)

    ranking.to_csv(output_dir / "uncertainty_information_gain_ranking.csv", index=False)
    skipped.to_csv(output_dir / "uncertainty_skipped_non_discrete_features.csv", index=False)

    with open(output_dir / "uncertainty_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Feature ranking supervisionato basato su incertezza: "
            "entropia H(Y) e Information Gain IG(Y,X)."
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
        help="Colonne da escludere dall'analisi.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Numero di feature da mostrare nel plot e nella tabella top.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Cartella di output per i file risultanti.",
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

    ranker = InformationGainRanker(log_base=2)
    results = ranker.rank(
        df=df,
        label_column=args.label_column,
        exclude_columns=args.exclude_columns,
    )

    _save_results(results, args.output_dir)

    ranking = results["information_gain_ranking"]
    assert isinstance(ranking, pd.DataFrame)

    top_df = ranker.plot_top_information_gain(
        ranking=ranking,
        top_n=args.top_n,
        output_path=args.output_dir / "uncertainty_top_information_gain.png",
        show_plot=args.show_plots,
    )
    top_df.to_csv(args.output_dir / "uncertainty_top_information_gain.csv", index=False)

    summary = results["summary"]
    assert isinstance(summary, dict)

    print("\n" + "=" * 80)
    print("FEATURE RANKING - INCERTEZZA (ENTROPIA / INFORMATION GAIN)")
    print("=" * 80)
    print(f"Sorgente dati: {source_text}")
    print(f"Target: {summary['label_column']}")
    print(f"Feature candidate: {summary['n_feature_candidates']}")
    print(f"Feature discrete usate: {summary['n_discrete_features']}")
    print(f"Feature non discrete scartate: {summary['n_skipped_non_discrete']}")
    print(f"Entropia globale target H(Y): {summary['target_entropy_global']:.6f}")
    print(f"Output salvato in: {args.output_dir.resolve()}")

    print("\nTop 10 feature per Information Gain:")
    print(ranking.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
