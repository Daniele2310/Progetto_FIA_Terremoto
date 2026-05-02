from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RECOMMENDED_ALPHA = 0.002
RECOMMENDED_ALPHA_METRICS = {
    "accuracy": 0.591815,
    "f1_macro": 0.462779,
    "balanced_accuracy": 0.454610,
    "n_features": 49,
}


class LassoFeatureSelector:
    """
    Feature selection embedded tramite regressione Lasso.

    Idea operativa:
    - codifico eventuali feature categoriche con one-hot encoding;
    - standardizzo le feature, perche la penalizzazione L1 e sensibile alla scala;
    - addestro un modello Lasso;
    - considero selezionate le feature con coefficiente diverso da zero.

    Nota:
    il target damage_grade e trattato come variabile numerica ordinata (1, 2, 3),
    cosi la regressione Lasso puo essere usata come strumento embedded di selezione.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    @staticmethod
    def _normalize_columns(columns: Optional[Iterable[str]]) -> list[str]:
        """Normalizza una lista opzionale di colonne."""
        if columns is None:
            return []
        return list(columns)

    @staticmethod
    def _load_default_dataframe(project_root: Path) -> tuple[pd.DataFrame, str]:
        """Carica il dataset di default del progetto."""
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
        """Prepara le feature candidate e il target numerico."""
        if label_column not in df.columns:
            raise ValueError(f"Colonna target non trovata: {label_column}")

        excluded = set(exclude_columns)
        excluded.add(label_column)

        feature_candidates = [col for col in df.columns if col not in excluded]
        if not feature_candidates:
            raise ValueError("Nessuna feature candidata disponibile.")

        x_raw = df[feature_candidates].copy()
        y = pd.to_numeric(df[label_column], errors="raise").astype(float)

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
                "Lasso richiede input senza NaN: completa imputazione/pulizia prima dell'uso."
            )

        return x_encoded, y

    def select(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        alpha: Optional[float] = None,
        cv_folds: int = 5,
        max_iter: int = 10000,
        selection_tolerance: float = 1e-8,
    ) -> dict[str, object]:
        """Addestra il Lasso e restituisce feature selezionate e summary."""
        if x.empty:
            raise ValueError("Il dataframe delle feature e vuoto.")
        if len(x) != len(y):
            raise ValueError("Feature e target devono avere lo stesso numero di righe.")
        if alpha is not None and alpha <= 0:
            raise ValueError("alpha deve essere > 0.")
        if cv_folds < 2:
            raise ValueError("cv_folds deve essere almeno 2.")

        # StandardScaler porta le feature su scala comparabile prima della penalizzazione L1.
        if alpha is None:
            model = LassoCV(
                cv=cv_folds,
                random_state=self.random_state,
                max_iter=max_iter,
                n_jobs=None,
            )
            training_mode = "lasso_cv"
        else:
            model = Lasso(
                alpha=alpha,
                random_state=self.random_state,
                max_iter=max_iter,
            )
            training_mode = "lasso_fixed_alpha"

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("lasso", model),
            ]
        )
        pipeline.fit(x, y)

        trained_model = pipeline.named_steps["lasso"]
        coefficients = pd.DataFrame(
            {
                "feature": x.columns,
                "coefficient": trained_model.coef_.astype(float),
                "abs_coefficient": np.abs(trained_model.coef_.astype(float)),
            }
        ).sort_values(
            by=["abs_coefficient", "feature"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)

        selected = coefficients.loc[
            coefficients["abs_coefficient"] > selection_tolerance
        ].copy()
        selected.insert(0, "rank", range(1, len(selected) + 1))

        summary = {
            "metodo": "lasso_feature_selection",
            "tipo_metodo": "embedded",
            "modello": "LassoCV" if alpha is None else "Lasso",
            "training_mode": training_mode,
            "target_trattato_come": "variabile numerica ordinata",
            "penalizzazione": "L1",
            "numero_righe": int(len(x)),
            "numero_feature_iniziali": int(x.shape[1]),
            "numero_feature_selezionate": int(len(selected)),
            "selection_tolerance": float(selection_tolerance),
            "cv_folds": int(cv_folds) if alpha is None else None,
            "alpha_usato": float(trained_model.alpha_ if alpha is None else trained_model.alpha),
            "intercetta": float(trained_model.intercept_),
            "massimo_coefficiente_assoluto": float(coefficients["abs_coefficient"].max()),
            "random_state": int(self.random_state),
        }

        return {
            "summary": summary,
            "selected_features": selected,
            "all_coefficients": coefficients,
        }


def _ask_alpha_choice(default_alpha: Optional[float]) -> Optional[float]:
    """
    Chiede all'utente come determinare alpha.

    Possibili scelte:
    - LassoCV: alpha scelto automaticamente via cross-validation;
    - alpha consigliato 0.002: compromesso osservato tra selezione e prestazioni;
    - alpha personalizzato inserito manualmente.

    Se l'input non e disponibile, mantiene il default ricevuto.
    """
    if default_alpha is not None:
        return default_alpha

    print("\n" + "=" * 80)
    print("SCELTA DEL PARAMETRO ALPHA PER IL LASSO")
    print("=" * 80)
    print(
        "Alpha controlla la forza della penalizzazione L1: "
        "piu alpha cresce, piu il Lasso tende ad azzerare coefficienti e quindi a selezionare meno feature."
    )
    print("\nOpzioni disponibili:")
    print("1) Default automatico con LassoCV")
    print("   Il modello sceglie alpha tramite cross-validation.")
    print(f"2) Valore consigliato: alpha = {RECOMMENDED_ALPHA}")
    print(
        "   Compromesso osservato nelle prove interne: "
        f"accuracy = {RECOMMENDED_ALPHA_METRICS['accuracy']:.6f}, "
        f"f1_macro = {RECOMMENDED_ALPHA_METRICS['f1_macro']:.6f}, "
        f"balanced_accuracy = {RECOMMENDED_ALPHA_METRICS['balanced_accuracy']:.6f}, "
        f"feature selezionate = {RECOMMENDED_ALPHA_METRICS['n_features']}"
    )
    print("3) Inserisci manualmente un valore di alpha")

    try:
        choice = input("Seleziona opzione [1-3] (default=1): ").strip()
    except EOFError:
        choice = ""

    if choice not in {"1", "2", "3"}:
        choice = "1"

    if choice == "1":
        return None
    if choice == "2":
        return RECOMMENDED_ALPHA

    while True:
        try:
            raw_value = input("Inserisci un valore numerico positivo per alpha: ").strip()
        except EOFError:
            return None

        try:
            alpha_value = float(raw_value)
        except ValueError:
            print("Valore non valido. Inserisci un numero, ad esempio 0.002")
            continue

        if alpha_value <= 0:
            print("Alpha deve essere maggiore di zero.")
            continue

        return alpha_value


def _build_parser() -> argparse.ArgumentParser:
    """Costruisce il parser della riga di comando."""
    parser = argparse.ArgumentParser(
        description=(
            "Feature selection embedded con regressione Lasso: "
            "le feature con coefficiente finale diverso da zero vengono selezionate."
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
        "--alpha",
        type=float,
        default=None,
        help="Valore fisso di alpha. Se omesso, viene stimato automaticamente con LassoCV.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Numero di fold per LassoCV quando alpha non e specificato.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10000,
        help="Numero massimo di iterazioni del solver.",
    )
    parser.add_argument(
        "--selection-tolerance",
        type=float,
        default=1e-8,
        help="Soglia minima su |coefficiente| per considerare una feature selezionata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Cartella di output per CSV/JSON.",
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
        df, source_text = LassoFeatureSelector._load_default_dataframe(project_root)

    chosen_alpha = _ask_alpha_choice(args.alpha)

    selector = LassoFeatureSelector(random_state=42)
    x_encoded, y = selector._prepare_features(
        df=df,
        label_column=args.label_column,
        exclude_columns=selector._normalize_columns(args.exclude_columns),
    )

    results = selector.select(
        x=x_encoded,
        y=y,
        alpha=chosen_alpha,
        cv_folds=args.cv_folds,
        max_iter=args.max_iter,
        selection_tolerance=args.selection_tolerance,
    )

    summary = results["summary"]
    selected = results["selected_features"]
    coefficients = results["all_coefficients"]

    assert isinstance(summary, dict)
    assert isinstance(selected, pd.DataFrame)
    assert isinstance(coefficients, pd.DataFrame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_dir / "lasso_selected_features.csv", index=False)
    coefficients.to_csv(args.output_dir / "lasso_all_coefficients.csv", index=False)

    with open(args.output_dir / "lasso_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("FEATURE SELECTION EMBEDDED - LASSO COMPLETATA")
    print("=" * 80)
    print(f"Sorgente dati: {source_text}")
    print(f"Feature iniziali: {summary['numero_feature_iniziali']}")
    print(f"Feature selezionate: {summary['numero_feature_selezionate']}")
    print(
        "Modalita alpha: "
        f"{'LassoCV (automatico)' if chosen_alpha is None else f'valore fisso = {chosen_alpha}'}"
    )
    print(f"Alpha usato: {summary['alpha_usato']:.10f}")
    print(f"Output salvato in: {args.output_dir.resolve()}")

    if selected.empty:
        print("\nNessuna feature ha superato la soglia di selezione.")
    else:
        print("\nFeature selezionate dal Lasso:")
        print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
