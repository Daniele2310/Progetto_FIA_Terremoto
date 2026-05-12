"""
Sistema multi-esperto tradizionale basato su decision profile.

Ogni esperto e' un classificatore sklearn addestrato su un sottoinsieme di
feature. In inferenza, le probabilita' prodotte dagli esperti compongono il
decision profile e vengono aggregate con regole semplici: mean, weighted_mean,
median, product oppure majority_vote.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone


@dataclass(frozen=True)
class ExpertSpec:
    """Configurazione di un esperto del sistema multi-esperto."""

    name: str
    estimator: object
    features: list[str] | None = None
    weight: float = 1.0


class MultiExpertSystem:
    """Multi-Expert System con aggregazione del decision profile."""

    def __init__(self, experts: Iterable[ExpertSpec | dict], aggregation: str = "weighted_mean"):
        self.experts = [self._coerce_expert(expert) for expert in experts]
        if not self.experts:
            raise ValueError("Il sistema multi-esperto richiede almeno un esperto.")

        allowed = {"mean", "weighted_mean", "median", "product", "majority_vote"}
        if aggregation not in allowed:
            raise ValueError(f"Aggregazione non supportata: {aggregation}. Valori ammessi: {sorted(allowed)}")

        self.aggregation = aggregation
        self.fitted_experts_: list[dict] = []
        self.classes_: np.ndarray | None = None

    @staticmethod
    def _coerce_expert(expert: ExpertSpec | dict) -> ExpertSpec:
        if isinstance(expert, ExpertSpec):
            return expert
        return ExpertSpec(
            name=expert["name"],
            estimator=expert["estimator"],
            features=expert.get("features"),
            weight=float(expert.get("weight", 1.0)),
        )

    def fit(self, X: pd.DataFrame, y) -> "MultiExpertSystem":
        """Addestra tutti gli esperti."""
        self.fitted_experts_ = []
        self.classes_ = np.array(sorted(pd.Series(y).unique()))

        for expert in self.experts:
            features = self._resolve_features(X, expert.features)
            estimator = clone(expert.estimator)
            estimator.fit(X[features], y)
            self.fitted_experts_.append(
                {
                    "name": expert.name,
                    "estimator": estimator,
                    "features": features,
                    "weight": expert.weight,
                }
            )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predice la classe finale aggregando il decision profile."""
        if self.aggregation == "majority_vote":
            return self._predict_majority_vote(X)

        scores = self.predict_profile(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_profile(self, X: pd.DataFrame) -> np.ndarray:
        """
        Restituisce il vettore aggregato a c elementi per ogni campione.

        Shape: (n_samples, n_classes).
        """
        profiles = self.decision_profile(X)

        if self.aggregation == "mean":
            return profiles.mean(axis=1)

        if self.aggregation == "weighted_mean":
            weights = self._normalized_weights()
            return np.average(profiles, axis=1, weights=weights)

        if self.aggregation == "median":
            return np.median(profiles, axis=1)

        if self.aggregation == "product":
            product = np.prod(np.clip(profiles, 1e-12, 1.0), axis=1)
            normalizer = product.sum(axis=1, keepdims=True)
            return product / np.where(normalizer == 0, 1.0, normalizer)

        raise ValueError(f"Aggregazione non supportata per decision profile: {self.aggregation}")

    def decision_profile(self, X: pd.DataFrame) -> np.ndarray:
        """
        Costruisce il decision profile.

        Shape: (n_samples, n_experts, n_classes).
        """
        self._check_fitted()
        return np.stack([self._expert_proba(expert, X) for expert in self.fitted_experts_], axis=1)

    def predict_experts(self, X: pd.DataFrame) -> pd.DataFrame:
        """Restituisce le predizioni individuali di ogni esperto."""
        self._check_fitted()
        data = {}
        for expert in self.fitted_experts_:
            data[expert["name"]] = expert["estimator"].predict(X[expert["features"]])
        return pd.DataFrame(data, index=X.index)

    def diversity_report(self, X: pd.DataFrame, y) -> pd.DataFrame:
        """
        Calcola misure pairwise di diversita' tra esperti.

        Include disagreement, double fault, Q-statistic e correlazione phi.
        """
        predictions = self.predict_experts(X)
        y_true = np.asarray(y)
        rows = []

        for name_a, name_b in combinations(predictions.columns, 2):
            correct_a = predictions[name_a].to_numpy() == y_true
            correct_b = predictions[name_b].to_numpy() == y_true

            n11 = int(np.sum(correct_a & correct_b))
            n00 = int(np.sum(~correct_a & ~correct_b))
            n10 = int(np.sum(correct_a & ~correct_b))
            n01 = int(np.sum(~correct_a & correct_b))
            total = n11 + n00 + n10 + n01

            q_den = (n11 * n00) + (n10 * n01)
            phi_den = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))

            rows.append(
                {
                    "expert_a": name_a,
                    "expert_b": name_b,
                    "n11_both_correct": n11,
                    "n00_both_wrong": n00,
                    "n10_a_correct_b_wrong": n10,
                    "n01_a_wrong_b_correct": n01,
                    "disagreement": (n10 + n01) / total if total else 0.0,
                    "double_fault": n00 / total if total else 0.0,
                    "q_statistic": ((n11 * n00) - (n10 * n01)) / q_den if q_den else 0.0,
                    "correlation_phi": ((n11 * n00) - (n10 * n01)) / phi_den if phi_den else 0.0,
                }
            )

        return pd.DataFrame(rows)

    def _predict_majority_vote(self, X: pd.DataFrame) -> np.ndarray:
        predictions = self.predict_experts(X)
        weights = {expert["name"]: expert["weight"] for expert in self.fitted_experts_}
        final_predictions = []

        for _, row in predictions.iterrows():
            class_scores = {label: 0.0 for label in self.classes_}
            for expert_name, predicted_class in row.items():
                class_scores[predicted_class] += weights[expert_name]
            final_predictions.append(max(class_scores, key=class_scores.get))

        return np.array(final_predictions)

    def _expert_proba(self, expert: dict, X: pd.DataFrame) -> np.ndarray:
        estimator = expert["estimator"]
        X_sub = X[expert["features"]]

        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(X_sub)
            estimator_classes = estimator.classes_
        else:
            predictions = estimator.predict(X_sub)
            estimator_classes = self.classes_
            proba = np.zeros((len(predictions), len(estimator_classes)))
            for idx, class_label in enumerate(estimator_classes):
                proba[:, idx] = predictions == class_label

        aligned = np.zeros((X.shape[0], len(self.classes_)))
        for source_idx, class_label in enumerate(estimator_classes):
            target_idx = int(np.where(self.classes_ == class_label)[0][0])
            aligned[:, target_idx] = proba[:, source_idx]

        return aligned

    def _normalized_weights(self) -> np.ndarray:
        weights = np.array([expert["weight"] for expert in self.fitted_experts_], dtype=float)
        if np.any(weights < 0):
            raise ValueError("I pesi degli esperti devono essere non negativi.")
        if weights.sum() == 0:
            return np.ones_like(weights) / len(weights)
        return weights / weights.sum()

    @staticmethod
    def _resolve_features(X: pd.DataFrame, features: list[str] | None) -> list[str]:
        if features is None:
            return X.columns.tolist()

        valid_features = [feature for feature in features if feature in X.columns]
        if not valid_features:
            raise ValueError("Nessuna feature valida trovata per un esperto.")
        return valid_features

    def _check_fitted(self) -> None:
        if not self.fitted_experts_ or self.classes_ is None:
            raise RuntimeError("Il sistema multi-esperto deve essere addestrato con fit().")
