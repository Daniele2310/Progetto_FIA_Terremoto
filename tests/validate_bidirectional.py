from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.feature_selection.subset_selection.bidirectional_subset_selection import StepwiseBidirectionalSelector


def _build_estimator(name: str, random_state: int):
    selector = StepwiseBidirectionalSelector(estimator_name=name, scoring="accuracy", random_state=random_state)
    return selector._build_estimator()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Valida il subset prodotto da Stepwise Bidirectional Selection e calcola "
            "accuracy/f1_micro su holdout."
        )
    )
    parser.add_argument("--input", type=Path, default=None, help="CSV input con feature + target.")
    parser.add_argument("--label-column", type=str, default="damage_grade", help="Nome colonna target.")
    parser.add_argument(
        "--exclude-columns",
        nargs="*",
        default=["building_id"],
        help="Colonne da escludere dalla preparazione feature.",
    )
    parser.add_argument(
        "--selected-features-path",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "bidirectional_selected_features.csv",
        help="CSV con colonna selected_feature (usato solo con protocol=same-split).",
    )
    parser.add_argument(
        "--estimator",
        type=str,
        default="knn",
        choices=["logreg", "knn"],
        help="Modello usato per la validazione.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Quota holdout.")
    parser.add_argument(
        "--protocol",
        type=str,
        default="outer-holdout",
        choices=["outer-holdout", "same-split"],
        help=(
            "outer-holdout: selezione su train interno e valutazione su test esterno (raccomandato). "
            "same-split: usa feature gia selezionate e valuta sullo stesso split campionato."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=3000,
        help="Numero massimo di righe usate in validazione (None non supportato via CLI).",
    )
    parser.add_argument(
        "--selection-test-size",
        type=float,
        default=0.2,
        help="Quota holdout interna usata dalla selection (solo protocol=outer-holdout).",
    )
    parser.add_argument(
        "--selection-min-features",
        type=int,
        default=1,
        help="Parametro min_features per la selection interna (solo protocol=outer-holdout).",
    )
    parser.add_argument(
        "--selection-max-features",
        type=int,
        default=None,
        help="Parametro max_features per la selection interna (solo protocol=outer-holdout).",
    )
    parser.add_argument(
        "--selection-min-improvement",
        type=float,
        default=0.0,
        help="Parametro min_improvement per la selection interna (solo protocol=outer-holdout).",
    )
    parser.add_argument(
        "--selection-max-cycles",
        type=int,
        default=None,
        help="Parametro max_cycles per la selection interna (solo protocol=outer-holdout).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Cartella output metriche.",
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
        df, source_text = StepwiseBidirectionalSelector._load_default_dataframe(project_root)

    selector = StepwiseBidirectionalSelector(estimator_name=args.estimator, scoring="accuracy", random_state=42)

    x_encoded, y = selector._prepare_features(
        df=df,
        label_column=args.label_column,
        exclude_columns=selector._normalize_columns(args.exclude_columns),
    )

    if len(x_encoded) > args.max_rows:
        x_encoded, _, y, _ = train_test_split(
            x_encoded,
            y,
            train_size=args.max_rows,
            random_state=42,
            stratify=y,
        )

    if args.protocol == "same-split":
        if not args.selected_features_path.exists():
            raise FileNotFoundError(
                "File selected features non trovato. Esegui prima bidirectional_subset_selection.py"
            )

        selected_df = pd.read_csv(args.selected_features_path)
        if "selected_feature" not in selected_df.columns:
            raise ValueError("Il file selected features deve contenere la colonna 'selected_feature'.")

        selected_features = [f for f in selected_df["selected_feature"].tolist() if f in x_encoded.columns]
        if not selected_features:
            raise ValueError("Nessuna feature selezionata valida trovata nel dataset corrente.")

        x_train, x_test, y_train, y_test = train_test_split(
            x_encoded,
            y,
            test_size=args.test_size,
            random_state=42,
            stratify=y,
        )

        selection_summary = {
            "mode": "precomputed_selected_features",
            "n_features_selected": int(len(selected_features)),
        }

    else:
        x_selection, x_test, y_selection, y_test = train_test_split(
            x_encoded,
            y,
            test_size=args.test_size,
            random_state=42,
            stratify=y,
        )

        selection_results = selector.select(
            x=x_selection,
            y=y_selection,
            min_features=args.selection_min_features,
            max_features=args.selection_max_features,
            test_size=args.selection_test_size,
            max_rows=None,
            min_improvement=args.selection_min_improvement,
            max_cycles=args.selection_max_cycles,
        )
        selected_df = selection_results["selected_features"]
        selection_summary = selection_results["summary"]

        assert isinstance(selected_df, pd.DataFrame)
        assert isinstance(selection_summary, dict)

        selected_features = [f for f in selected_df["selected_feature"].tolist() if f in x_selection.columns]
        if not selected_features:
            raise ValueError("Selection interna non ha prodotto feature valide.")

        x_train, y_train = x_selection, y_selection

    estimator_subset = _build_estimator(args.estimator, random_state=42)
    estimator_subset.fit(x_train[selected_features], y_train)
    y_pred_subset = estimator_subset.predict(x_test[selected_features])

    subset_accuracy = float(accuracy_score(y_test, y_pred_subset))
    subset_f1_micro = float(f1_score(y_test, y_pred_subset, average="micro"))

    estimator_full = _build_estimator(args.estimator, random_state=42)
    estimator_full.fit(x_train, y_train)
    y_pred_full = estimator_full.predict(x_test)

    full_accuracy = float(accuracy_score(y_test, y_pred_full))
    full_f1_micro = float(f1_score(y_test, y_pred_full, average="micro"))

    metrics = {
        "source": source_text,
        "estimator": args.estimator,
        "protocol": args.protocol,
        "n_rows_used": int(len(x_encoded)),
        "n_features_full": int(x_encoded.shape[1]),
        "n_features_selected": int(len(selected_features)),
        "subset_accuracy": subset_accuracy,
        "subset_f1_micro": subset_f1_micro,
        "full_accuracy": full_accuracy,
        "full_f1_micro": full_f1_micro,
        "delta_accuracy_subset_minus_full": subset_accuracy - full_accuracy,
        "delta_f1_subset_minus_full": subset_f1_micro - full_f1_micro,
        "selection_summary": selection_summary,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "bidirectional_validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame([metrics]).to_csv(args.output_dir / "bidirectional_validation_metrics.csv", index=False)

    print("\n" + "=" * 80)
    print("VALIDAZIONE BIDIRECTIONAL COMPLETATA")
    print("=" * 80)
    print(f"Protocollo: {metrics['protocol']}")
    print(f"Righe usate: {metrics['n_rows_used']}")
    print(f"Feature full / selected: {metrics['n_features_full']} / {metrics['n_features_selected']}")
    print(f"Subset accuracy: {metrics['subset_accuracy']:.6f}")
    print(f"Subset f1_micro: {metrics['subset_f1_micro']:.6f}")
    print(f"Full accuracy: {metrics['full_accuracy']:.6f}")
    print(f"Full f1_micro: {metrics['full_f1_micro']:.6f}")
    print(f"Delta accuracy (subset-full): {metrics['delta_accuracy_subset_minus_full']:.6f}")
    print(f"Delta f1 (subset-full): {metrics['delta_f1_subset_minus_full']:.6f}")
    print(f"Output salvato in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
