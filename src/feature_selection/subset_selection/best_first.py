from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split


@dataclass
class BestFirstStep:
    step: int
    expanded_subset: str
    best_child_generated: str
    score_of_expanded: float
    score_of_best_child: float
    global_best_score: float
    stagnation_count: int


class BestFirstSelector:
    """
    Best First Search (Forward Selection).

    Procedura:
    1. Inizializzazione: Parte dal subset vuoto.
    2. Passo 1: Valuta tutte le singole feature e le inserisce in una coda di priorità.
    3. Espansione: Estrae iterativamente il subset migliore dalla coda e lo espande
       aggiungendo una nuova feature alla volta.
    4. Stop: L'algoritmo si ferma se dopo 'k' espansioni di nodi non si riesce a
       migliorare lo score globale (patience).

    Valutazione: Decision Tree Classifier con 5-fold cross-validation (Accuracy).
    """

    def __init__(
            self,
            patience: int = 5, # numero max. di espansioni senza miglioramento prima di fermarsi
            random_state: int = 42,
    ):
        if patience <= 0:
            raise ValueError("patience deve essere > 0.")
        self.patience = patience  # Corrisponde al "k" delle tue istruzioni
        self.random_state = random_state

    # ---------------------------------------------------------
    # Metodi helper
    # ---------------------------------------------------------
    @staticmethod
    def _normalize_columns(columns: Optional[Iterable[str]]) -> list[str]:
        if columns is None: return []
        return list(columns)

    @staticmethod
    def _to_numeric_label(label: pd.Series) -> np.ndarray:
        if pd.api.types.is_numeric_dtype(label):
            return label.astype(int).to_numpy()
        codes, _ = pd.factorize(label, sort=True)
        return codes.astype(int)

    @staticmethod
    def _load_default_dataframe(project_root: Path) -> tuple[pd.DataFrame, str]:
        train_values_path = project_root / "Data" / "raw" / "train_values.csv"
        train_labels_path = project_root / "Data" / "raw" / "train_labels.csv"

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
        y = BestFirstSelector._to_numeric_label(df[label_column])

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
            raise ValueError("Best First richiede input senza NaN: completa imputazione/pulizia prima dell'uso.")

        return x_encoded, y

    def _build_estimator(self): # serve per istanziare un Decision Tree
        # Utilizzo esclusivo del Decision Tree Classifier
        return DecisionTreeClassifier(random_state=self.random_state)

    def _evaluate_subset(
            self, x: np.ndarray, y: np.ndarray, feature_idx: tuple
    ) -> float:
        """
        Valuta il subset usando 5-fold cross validation.
        Ritorna l'Accuracy media sui 5 fold.
        """
        if not feature_idx:
            return 0.0  # Subset vuoto

        idx_list = list(feature_idx)
        estimator = self._build_estimator()

        # 5-fold cross validation calcolando l'accuracy
        scores = cross_val_score(
            estimator,
            x[:, idx_list],
            y,
            cv=5,
            scoring='accuracy',
            n_jobs=-1  # Usa tutti i core disponibili
        )

        return float(np.mean(scores))

    # ---------------------------------------------------------
    # Core Logic del Best First
    # ---------------------------------------------------------
    def select(
            self,
            x: pd.DataFrame,
            y: np.ndarray,
            max_rows: Optional[int] = 20000,
    ) -> dict[str, object]:
        if x.empty:
            raise ValueError("Il dataframe delle feature e vuoto.")
        if len(x) != len(y):
            raise ValueError("Feature e target devono avere lo stesso numero di righe.")
        if max_rows is not None and max_rows <= 200:
            raise ValueError("max_rows deve essere > 200 oppure None.")

        # Preparazione Dati
        x_work, y_work = x.copy(), y.copy()
        sampled_rows = len(x_work)

        # Sotto-campionamento opzionale per velocizzare il processo
        if max_rows is not None and len(x_work) > max_rows:
            x_work, _, y_work, _ = train_test_split(
                x_work, y_work, train_size=max_rows, random_state=self.random_state, stratify=y_work
            )
            sampled_rows = len(x_work)

        feature_names = x_work.columns.to_numpy()
        x_train = x_work.to_numpy(dtype=float)
        y_train = y_work
        total_features = x_train.shape[1]

        start = perf_counter()
        evaluated_models = 0

        # Strutture dati per Best First
        open_list = []  # Coda di priorità (Min-Heap con score negativi)
        closed_set = set()  # Set per evitare di valutare subset già esplorati
        # Un diario per tutte le combinazioni già testate (per evitare di calcolare 2 volte lo stesso modello)

        best_global_score = -np.inf
        best_global_subset = tuple()
        expansions_without_improvement = 0
        history: list[BestFirstStep] = []

        # ==========================================
        # 1. & 2. Inizializzazione e Passo 1 (Singole Feature)
        # ==========================================
        for i in range(total_features):
            subset = (i,)
            score = self._evaluate_subset(x_train, y_train, subset)
            evaluated_models += 1

            # heapq inserisce in ordine crescente. Usiamo -score affinché il punteggio più alto sia in cima
            heapq.heappush(open_list, (-score, subset))
            closed_set.add(subset)

            if score > best_global_score:
                best_global_score = score
                best_global_subset = subset

        # ==========================================
        # 3. & 4. & 5. Espansione, Progressione, Stop (k patience)
        # ==========================================
        step_count = 0
        stop_reason = "coda_esaurita"

        while open_list:
            if expansions_without_improvement >= self.patience:
                stop_reason = f"patience_{self.patience}_raggiunta"
                break

            # Estrazione del migliore (es. {X2})
            neg_score, current_subset = heapq.heappop(open_list)
            current_score = -neg_score
            step_count += 1

            improved_in_this_expansion = False
            best_child_this_step = None
            best_child_score = -np.inf

            # Espansione aggiungendo una feature alla volta
            for i in range(total_features):
                if i not in current_subset:
                    # Creazione del nuovo subset (es. {X2, X4}) e sorting per unicità nel set
                    new_subset = tuple(sorted(current_subset + (i,)))

                    if new_subset not in closed_set:
                        closed_set.add(new_subset)
                        child_score = self._evaluate_subset(x_train, y_train, new_subset)
                        evaluated_models += 1

                        # Inserimento nella coda di priorità (rimarrà in lista come potenziale ripartenza)
                        heapq.heappush(open_list, (-child_score, new_subset))

                        if child_score > best_child_score:
                            best_child_score = child_score
                            best_child_this_step = new_subset

                        # Se un figlio batte il massimo globale
                        if child_score > best_global_score:
                            best_global_score = child_score
                            best_global_subset = new_subset
                            improved_in_this_expansion = True

            # Gestione Patience
            if improved_in_this_expansion:
                expansions_without_improvement = 0
            else:
                expansions_without_improvement += 1

            # Log dello step
            step_obj = BestFirstStep(
                step=step_count,
                expanded_subset=", ".join([feature_names[idx] for idx in current_subset]),
                best_child_generated=", ".join(
                    [feature_names[idx] for idx in best_child_this_step]) if best_child_this_step else "",
                score_of_expanded=float(current_score),
                score_of_best_child=float(best_child_score),
                global_best_score=float(best_global_score),
                stagnation_count=expansions_without_improvement
            )
            history.append(step_obj)

        elapsed_sec = perf_counter() - start
        selected_features = [feature_names[idx] for idx in best_global_subset]

        summary = {
            "estimator": "DecisionTree (5-fold CV)",
            "scoring": "accuracy",
            "n_rows_used": int(sampled_rows),
            "n_features_initial": int(total_features),
            "n_features_final": len(selected_features),
            "best_score_final": float(best_global_score),
            "evaluated_models": int(evaluated_models),
            "elapsed_seconds": float(elapsed_sec),
            "stop_reason": stop_reason,
            "patience": self.patience
        }

        history_df = pd.DataFrame([step.__dict__ for step in history])
        selected_df = pd.DataFrame({"selected_feature": selected_features})

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

        x_steps = history_df["step"].astype(int).tolist()
        y_best = history_df["global_best_score"].astype(float).tolist()
        y_expanded = history_df["score_of_expanded"].astype(float).tolist()

        plt.figure(figsize=(10, 6))
        plt.plot(x_steps, y_best, marker="o", label="Best score globale")
        plt.plot(x_steps, y_expanded, marker="x", linestyle="--", label="Score subset espanso")
        plt.xlabel("Espansione")
        plt.ylabel("Accuracy CV")
        plt.title("Andamento Best First Search")
        plt.grid(alpha=0.2)
        plt.legend()
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
            "Subset selection con Best First Search: espande i subset piu promettenti "
            "e si ferma dopo k espansioni senza miglioramento globale."
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
        "--patience",
        type=int,
        default=5,
        help="Numero massimo di espansioni senza miglioramento globale prima dello stop.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=20000,
        help="Numero massimo di righe usate per contenere il costo computazionale.",
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

    project_root = Path(__file__).resolve().parents[3]

    if args.input is not None:
        if not args.input.exists():
            raise FileNotFoundError(f"File non trovato: {args.input}")
        df = pd.read_csv(args.input)
        source_text = str(args.input)
    else:
        df, source_text = BestFirstSelector._load_default_dataframe(project_root)

    selector = BestFirstSelector(
        patience=args.patience,
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
        max_rows=args.max_rows,
    )

    summary = results["summary"]
    history = results["history"]
    selected = results["selected_features"]

    assert isinstance(summary, dict)
    assert isinstance(history, pd.DataFrame)
    assert isinstance(selected, pd.DataFrame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.output_dir / "best_first_history.csv", index=False)
    selected.to_csv(args.output_dir / "best_first_selected_features.csv", index=False)
    with open(args.output_dir / "best_first_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    selector.plot_history(
        history_df=history,
        output_path=args.output_dir / "best_first_score_history.png",
        show_plot=args.show_plots,
    )

    print("\n" + "=" * 80)
    print("SUBSET SELECTION - BEST FIRST COMPLETATA")
    print("=" * 80)
    print(f"Sorgente dati: {source_text}")
    print(f"Feature iniziali/finali: {summary['n_features_initial']} -> {summary['n_features_final']}")
    print(f"Score finale: {summary['best_score_final']:.6f}")
    print(f"Modelli valutati: {summary['evaluated_models']}")
    print(f"Tempo totale (s): {summary['elapsed_seconds']:.2f}")
    print(f"Stop reason: {summary['stop_reason']}")
    print(f"Patience: {summary['patience']}")
    print(f"Output salvato in: {args.output_dir.resolve()}")

    print("\nTop feature selezionate:")
    print(selected.to_string(index=False))

    if not history.empty:
        print("\nUltimi 10 step Best First:")
        print(history.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
