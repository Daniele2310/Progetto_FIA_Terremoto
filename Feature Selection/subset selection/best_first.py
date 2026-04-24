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