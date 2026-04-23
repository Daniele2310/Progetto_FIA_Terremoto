from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class MaxMinStep:
    """Rappresenta una singola iterazione di selezione."""

    step: int
    selected_feature: str
    relevance: float
    redundancy: float
    score: float


class MaxMinSubsetSelector:
    """
    Selezione subset supervisionata con criterio Max-Min.

    Idea:
    - massimizzare la rilevanza rispetto al target;
    - minimizzare la ridondanza rispetto alle feature gia scelte.

    Score greedy usato in questo script:
        score(feature) = relevance(feature, target) - redundancy(feature, selected_set)

    Dove:
    - relevance = |corr(feature, target)|
    - redundancy = max |corr(feature, feature_gia_selezionata)|

    Nota:
    questo e un criterio "configuration based" semplice e leggibile,
    coerente con gli appunti: sceglie feature utili al target evitando
    informazione ridondante nel subset.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    @staticmethod
    def _normalize_columns(columns: Optional[Iterable[str]]) -> list[str]:
        """Normalizza l'input delle colonne escluse in una lista."""
        if columns is None:
            return []
        return list(columns)

    @staticmethod
    def _to_numeric_label(label: pd.Series) -> pd.Series:
        """Converte il target in formato numerico."""
        if pd.api.types.is_numeric_dtype(label):
            return label.astype(float)

        codes, _ = pd.factorize(label, sort=True)
        return pd.Series(codes, index=label.index, dtype=float)

    @staticmethod
    def _load_default_dataframe(project_root: Path) -> tuple[pd.DataFrame, str]:
        """Carica il dataset di default dal progetto."""
        train_values_path = project_root / "Data" / "train_values.csv"
        train_labels_path = project_root / "Data" / "train_labels.csv"

        if train_values_path.exists() and train_labels_path.exists():
            train_values = pd.read_csv(train_values_path)
            train_labels = pd.read_csv(train_labels_path)
            merged = train_values.merge(train_labels, on="building_id", how="inner")
            return merged, f"{train_values_path} + {train_labels_path}"

        preprocessed_with_labels = (
            project_root / "DataPreprocessed" / "train_features_labels_preprocessed.csv"
        )
        if preprocessed_with_labels.exists():
            return pd.read_csv(preprocessed_with_labels), str(preprocessed_with_labels)

        raise FileNotFoundError(
            "Nessun dataset di default trovato. Usa --input per specificare un CSV."
        )

    @staticmethod
    def _prepare_features(
        df: pd.DataFrame,
        label_column: str,
        exclude_columns: list[str],
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepara le feature per il calcolo del criterio Max-Min."""
        if label_column not in df.columns:
            raise ValueError(f"Colonna target non trovata: {label_column}")

        excluded = set(exclude_columns)
        excluded.add(label_column)

        feature_candidates = [col for col in df.columns if col not in excluded]
        if not feature_candidates:
            raise ValueError("Nessuna feature candidata disponibile.")

        x_raw = df[feature_candidates].copy()
        y = MaxMinSubsetSelector._to_numeric_label(df[label_column])

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
            x_encoded = pd.get_dummies(
                x_raw,
                columns=categorical_cols,
                drop_first=False,
                dtype=float,
            )
        else:
            x_encoded = x_raw.astype(float)

        if x_encoded.isnull().any().any():
            raise ValueError(
                "Max-Min richiede input senza NaN: completa imputazione/pulizia prima dell'uso."
            )

        return x_encoded, y

    @staticmethod
    def _safe_abs_corr(series_a: pd.Series, series_b: pd.Series) -> float:
        """Restituisce la correlazione assoluta gestendo eventuali NaN."""
        corr = series_a.corr(series_b)
        if pd.isna(corr):
            return 0.0
        return float(abs(corr))

    def select(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        max_features: Optional[int] = None,
        stop_at_negative_score: bool = True,
    ) -> dict[str, pd.DataFrame | dict[str, object]]:
        """Esegue la selezione greedy del subset con criterio Max-Min."""
        if x.empty:
            raise ValueError("Il dataframe delle feature e vuoto.")
        if len(x) != len(y):
            raise ValueError("Feature e target devono avere lo stesso numero di righe.")
        if max_features is not None:
            if max_features <= 0:
                raise ValueError("max_features deve essere > 0 oppure None.")
            if max_features > x.shape[1]:
                raise ValueError("max_features non puo superare il numero di feature disponibili.")

        # Calcolo della rilevanza supervisionata di ogni feature verso il target.
        relevance = pd.Series(
            {feature: self._safe_abs_corr(x[feature], y) for feature in x.columns},
            dtype=float,
        )

        # Matrice di correlazione assoluta tra feature per il calcolo della ridondanza.
        feature_corr = x.corr(method="pearson").abs().fillna(0.0)

        selected_features: list[str] = []
        available_features = list(x.columns)
        history: list[MaxMinStep] = []
        stop_reason = "tutte_le_feature_selezionate"
        score_threshold = 0.0 if stop_at_negative_score else float("-inf")

        step_idx = 1
        while available_features:
            # Applica un eventuale limite superiore sul numero di feature selezionate.
            if max_features is not None and len(selected_features) >= max_features:
                stop_reason = "raggiunto_numero_massimo_feature"
                break

            candidate_rows: list[dict[str, float | str]] = []

            for feature in available_features:
                if not selected_features:
                    # Primo step: nessuna ridondanza da penalizzare.
                    redundancy = 0.0
                else:
                    # Penalizzo il legame piu forte con il subset gia scelto.
                    redundancy = float(feature_corr.loc[feature, selected_features].max())

                score = float(relevance.loc[feature] - redundancy)
                candidate_rows.append(
                    {
                        "feature": feature,
                        "relevance": float(relevance.loc[feature]),
                        "redundancy": redundancy,
                        "score": score,
                    }
                )

            candidates_df = pd.DataFrame(candidate_rows).sort_values(
                by=["score", "relevance", "redundancy", "feature"],
                ascending=[False, False, True, True],
                kind="stable",
            ).reset_index(drop=True)

            best = candidates_df.iloc[0]
            # Interrompe la selezione se il miglior compromesso disponibile e negativo.
            if float(best["score"]) < score_threshold:
                stop_reason = "score_negativo"
                break

            chosen_feature = str(best["feature"])
            selected_features.append(chosen_feature)
            available_features.remove(chosen_feature)

            history.append(
                MaxMinStep(
                    step=step_idx,
                    selected_feature=chosen_feature,
                    relevance=float(best["relevance"]),
                    redundancy=float(best["redundancy"]),
                    score=float(best["score"]),
                )
            )
            step_idx += 1

        history_df = pd.DataFrame([step.__dict__ for step in history])
        selected_df = history_df[["step", "selected_feature", "relevance", "redundancy", "score"]].copy()

        summary = {
            "metodo": "max_min_subset_selection",
            "formula_score": "score = rilevanza(feature, target) - ridondanza(feature, selected_set)",
            "definizione_rilevanza": "correlazione assoluta di Pearson tra feature e target",
            "definizione_ridondanza": "massima correlazione assoluta con le feature gia selezionate",
            "numero_righe": int(len(x)),
            "numero_feature_iniziali": int(x.shape[1]),
            "numero_feature_selezionate": int(len(selected_features)),
            "numero_massimo_feature": int(max_features) if max_features is not None else None,
            "stop_su_score_negativo": bool(stop_at_negative_score),
            "motivo_stop": stop_reason,
            "random_state": int(self.random_state),
        }

        return {
            "summary": summary,
            "selected_features": selected_df,
            "correlation_matrix": feature_corr,
        }


def _build_parser() -> argparse.ArgumentParser:
    """Costruisce il parser della riga di comando."""
    parser = argparse.ArgumentParser(
        description=(
            "Subset selection supervisionata con criterio Max-Min: "
            "massimizza la rilevanza verso il target e minimizza la ridondanza tra feature."
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
        "--max-features",
        type=int,
        default=None,
        help="Numero massimo di feature selezionabili. Default: nessun limite, stop automatico su score negativo.",
    )
    parser.add_argument(
        "--disable-negative-stop",
        action="store_true",
        help="Disattiva lo stop automatico quando il miglior candidato ha score negativo.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Cartella di output per CSV/JSON/plot.",
    )
    return parser


def main() -> None:
    """Esegue il flusso completo da riga di comando."""
    parser = _build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    if args.input is not None:
        if not args.input.exists():
            raise FileNotFoundError(f"File non trovato: {args.input}")
        df = pd.read_csv(args.input)
        source_text = str(args.input)
    else:
        df, source_text = MaxMinSubsetSelector._load_default_dataframe(project_root)

    selector = MaxMinSubsetSelector(random_state=42)
    x_encoded, y = selector._prepare_features(
        df=df,
        label_column=args.label_column,
        exclude_columns=selector._normalize_columns(args.exclude_columns),
    )

    results = selector.select(
        x=x_encoded,
        y=y,
        max_features=args.max_features,
        stop_at_negative_score=not args.disable_negative_stop,
    )

    summary = results["summary"]
    selected = results["selected_features"]
    corr_matrix = results["correlation_matrix"]

    assert isinstance(summary, dict)
    assert isinstance(selected, pd.DataFrame)
    assert isinstance(corr_matrix, pd.DataFrame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_dir / "max_min_subset.csv", index=False)
    corr_matrix.to_csv(args.output_dir / "max_min_feature_correlation_matrix.csv", index=True)

    with open(args.output_dir / "max_min_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("SUBSET SELECTION - MAX-MIN COMPLETATA")
    print("=" * 80)
    print(f"Sorgente dati: {source_text}")
    print(f"Feature iniziali: {summary['numero_feature_iniziali']}")
    print(f"Feature selezionate: {summary['numero_feature_selezionate']}")
    print(f"Motivo stop: {summary['motivo_stop']}")
    print(f"Output salvato in: {args.output_dir.resolve()}")

    print("\nFeature selezionate dal criterio Max-Min:")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
