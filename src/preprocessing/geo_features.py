from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


@dataclass
class _LevelStats:
    count_map: dict[Any, float]
    freq_map: dict[Any, float]
    mean_map: dict[Any, float]
    rare_values: set[Any]
    class_prob_maps: dict[Any, dict[Any, float]]


class GeoFeatureEngineer:
    """
    Costruisce feature geografiche aggregate a partire da geo_level_1/2/3.

    Feature principali generate:
    - count e frequency per ciascun livello geografico
    - flag di area rara
    - target mean smoothed per geo_level_1/2/3
    - probabilita smoothed per classe del target
    - feature gerarchiche finali con fallback 3 -> 2 -> 1 -> globale

    Per evitare leakage sul train, usare fit_transform_oof(...).
    """

    # =======================
    # INIZIALIZZAZIONE
    # =======================
    def __init__(
        self,
        geo_columns: tuple[str, str, str] = (
            "geo_level_1_id",
            "geo_level_2_id",
            "geo_level_3_id",
        ),
        target_col: str = "damage_grade",
        smoothing: float = 20.0,
        rare_threshold: int = 10,
        n_splits: int = 5,
        random_state: int = 42,
        append_original: bool = True,
    ) -> None:
        if len(geo_columns) != 3:
            raise ValueError("geo_columns deve contenere esattamente 3 colonne.")
        if smoothing < 0:
            raise ValueError("smoothing deve essere >= 0.")
        if rare_threshold < 1:
            raise ValueError("rare_threshold deve essere >= 1.")
        if n_splits < 2:
            raise ValueError("n_splits deve essere >= 2.")

        self.geo_columns = geo_columns
        self.target_col = target_col
        self.smoothing = float(smoothing)
        self.rare_threshold = int(rare_threshold)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)
        self.append_original = bool(append_original)

        self._fitted = False
        self.classes_: list[Any] = []
        self.global_mean_: float = np.nan
        self.global_class_probs_: dict[Any, float] = {}
        self.level_1_stats_: Optional[_LevelStats] = None
        self.level_2_stats_: Optional[_LevelStats] = None
        self.level_3_stats_: Optional[_LevelStats] = None

    # =======================
    # INTERFACCIA PUBBLICA
    # =======================
    # Metodi da usare dall'esterno:
    # - fit: apprende le statistiche geografiche dal train
    # - transform: applica le feature geo a un dataframe nuovo
    # - fit_transform: scorciatoia standard
    # - fit_transform_oof: versione anti-leakage per costruire le feature sul train
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series | pd.DataFrame] = None) -> "GeoFeatureEngineer":
        df = self._prepare_training_frame(X, y)
        self._fit_from_training_frame(df)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        self._validate_geo_columns(X)

        features = self._build_feature_frame(X)
        if self.append_original:
            return pd.concat([X.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
        return features.reset_index(drop=True)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series | pd.DataFrame] = None) -> pd.DataFrame:
        self.fit(X, y=y)
        return self.transform(X)

    def fit_transform_oof(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series | pd.DataFrame] = None,
    ) -> pd.DataFrame:
        df = self._prepare_training_frame(X, y)
        self._validate_target(df[self.target_col])

        splitter = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        oof_parts: list[pd.DataFrame] = []
        indices: list[pd.Index] = []

        for train_idx, valid_idx in splitter.split(df, df[self.target_col]):
            fold_train = df.iloc[train_idx].reset_index(drop=True)
            fold_valid = df.iloc[valid_idx].reset_index(drop=True)

            fold_engineer = GeoFeatureEngineer(
                geo_columns=self.geo_columns,
                target_col=self.target_col,
                smoothing=self.smoothing,
                rare_threshold=self.rare_threshold,
                n_splits=self.n_splits,
                random_state=self.random_state,
                append_original=False,
            )
            fold_engineer._fit_from_training_frame(fold_train)
            oof_parts.append(fold_engineer._build_feature_frame(fold_valid))
            indices.append(df.index[valid_idx])

        oof_features = pd.concat(oof_parts, axis=0)
        oof_features.index = np.concatenate([idx.to_numpy() for idx in indices])
        oof_features = oof_features.loc[df.index]

        self._fit_from_training_frame(df)

        if self.append_original:
            return pd.concat([X.reset_index(drop=True), oof_features.reset_index(drop=True)], axis=1)
        return oof_features.reset_index(drop=True)

    # =======================
    # FIT INTERNO DELLE STATISTICHE
    # =======================
    # Qui vengono calcolate, sul train:
    # - media globale del target
    # - probabilita globali delle classi
    # - statistiche aggregate per geo_level_1/2/3
    # usando smoothing gerarchico.
    def _fit_from_training_frame(self, df: pd.DataFrame) -> None:
        self._validate_geo_columns(df)
        self._validate_target(df[self.target_col])

        g1, g2, g3 = self.geo_columns
        total_rows = len(df)
        if total_rows == 0:
            raise ValueError("Il dataframe di training e vuoto.")

        self.classes_ = sorted(df[self.target_col].dropna().unique().tolist())
        self.global_mean_ = float(df[self.target_col].mean())
        self.global_class_probs_ = {
            cls: float((df[self.target_col] == cls).mean())
            for cls in self.classes_
        }

        level1_df = df[[g1, self.target_col]].copy()
        level1_grouped = level1_df.groupby(g1, dropna=False)
        self.level_1_stats_ = self._build_level_stats(
            grouped=level1_grouped,
            parent_mean_map={},
            parent_class_prob_maps={},
            parent_keys=[],
            global_mean=self.global_mean_,
            global_class_probs=self.global_class_probs_,
            total_rows=total_rows,
        )

        level2_df = df[[g1, g2, self.target_col]].copy()
        level2_grouped = level2_df.groupby([g1, g2], dropna=False)
        self.level_2_stats_ = self._build_level_stats(
            grouped=level2_grouped,
            parent_mean_map=self.level_1_stats_.mean_map,
            parent_class_prob_maps=self.level_1_stats_.class_prob_maps,
            parent_keys=[0],
            global_mean=self.global_mean_,
            global_class_probs=self.global_class_probs_,
            total_rows=total_rows,
        )

        level3_df = df[[g1, g2, g3, self.target_col]].copy()
        level3_grouped = level3_df.groupby([g1, g2, g3], dropna=False)
        self.level_3_stats_ = self._build_level_stats(
            grouped=level3_grouped,
            parent_mean_map=self.level_2_stats_.mean_map,
            parent_class_prob_maps=self.level_2_stats_.class_prob_maps,
            parent_keys=[0, 1],
            global_mean=self.global_mean_,
            global_class_probs=self.global_class_probs_,
            total_rows=total_rows,
        )

        self._fitted = True

    # =======================
    # COSTRUZIONE STATISTICHE PER LIVELLO
    # =======================
    # Questo blocco costruisce le statistiche di un singolo livello geografico:
    # - count
    # - frequency
    # - target mean smoothed
    # - probabilita di classe smoothed
    # - insieme delle aree rare
    def _build_level_stats(
        self,
        grouped: Any,
        parent_mean_map: dict[Any, float],
        parent_class_prob_maps: dict[Any, dict[Any, float]],
        parent_keys: list[int],
        global_mean: float,
        global_class_probs: dict[Any, float],
        total_rows: int,
    ) -> _LevelStats:
        count_map: dict[Any, float] = {}
        freq_map: dict[Any, float] = {}
        mean_map: dict[Any, float] = {}
        class_prob_maps: dict[Any, dict[Any, float]] = {}
        rare_values: set[Any] = set()

        for key, group in grouped:
            normalized_key = self._normalize_group_key(key)
            parent_key = self._get_parent_key(normalized_key, parent_keys)

            count = float(len(group))
            freq = float(count / total_rows)
            raw_mean = float(group[self.target_col].mean())
            parent_mean = parent_mean_map.get(parent_key, global_mean)
            smoothed_mean = self._smooth_value(raw_mean, parent_mean, count)

            count_map[normalized_key] = count
            freq_map[normalized_key] = freq
            mean_map[normalized_key] = smoothed_mean
            if count < self.rare_threshold:
                rare_values.add(normalized_key)

            class_prob_maps[normalized_key] = {}
            for cls in self.classes_:
                raw_prob = float((group[self.target_col] == cls).mean())
                parent_prob = parent_class_prob_maps.get(parent_key, {}).get(
                    cls,
                    global_class_probs[cls],
                )
                class_prob_maps[normalized_key][cls] = self._smooth_value(
                    raw_prob,
                    parent_prob,
                    count,
                )

        return _LevelStats(
            count_map=count_map,
            freq_map=freq_map,
            mean_map=mean_map,
            rare_values=rare_values,
            class_prob_maps=class_prob_maps,
        )

    # =======================
    # COSTRUZIONE DELLE FEATURE FINALI
    # =======================
    # Da ogni riga con geo_level_1/2/3 vengono ricavate le feature finali:
    # - statistiche raw per ciascun livello
    # - target mean smoothed per livello
    # - probabilita per classe per livello
    # - feature gerarchiche con fallback 3 -> 2 -> 1 -> globale
    def _build_feature_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.level_1_stats_ is not None
        assert self.level_2_stats_ is not None
        assert self.level_3_stats_ is not None

        g1, g2, g3 = self.geo_columns
        rows: list[dict[str, Any]] = []

        for row in X[[g1, g2, g3]].itertuples(index=False, name=None):
            key1 = row[0]
            key2 = (row[0], row[1])
            key3 = (row[0], row[1], row[2])

            row_features: dict[str, Any] = {
                f"{g1}_count": self._get_count(self.level_1_stats_, key1),
                f"{g1}_freq": self._get_freq(self.level_1_stats_, key1),
                f"{g1}_is_rare": self._is_rare(self.level_1_stats_, key1),
                f"{g2}_count": self._get_count(self.level_2_stats_, key2),
                f"{g2}_freq": self._get_freq(self.level_2_stats_, key2),
                f"{g2}_is_rare": self._is_rare(self.level_2_stats_, key2),
                f"{g3}_count": self._get_count(self.level_3_stats_, key3),
                f"{g3}_freq": self._get_freq(self.level_3_stats_, key3),
                f"{g3}_is_rare": self._is_rare(self.level_3_stats_, key3),
                f"{g1}_target_mean_smoothed": self._get_mean(self.level_1_stats_, key1, self.global_mean_),
                f"{g2}_target_mean_smoothed": self._get_mean(
                    self.level_2_stats_,
                    key2,
                    self._get_mean(self.level_1_stats_, key1, self.global_mean_),
                ),
                f"{g3}_target_mean_smoothed": self._get_mean(
                    self.level_3_stats_,
                    key3,
                    self._get_mean(
                        self.level_2_stats_,
                        key2,
                        self._get_mean(self.level_1_stats_, key1, self.global_mean_),
                    ),
                ),
                "geo_hierarchical_target_mean": self._hierarchical_mean(key1, key2, key3),
            }

            for cls in self.classes_:
                row_features[f"{g1}_class_{cls}_prob_smoothed"] = self._get_class_prob(
                    self.level_1_stats_,
                    key1,
                    cls,
                    self.global_class_probs_[cls],
                )
                row_features[f"{g2}_class_{cls}_prob_smoothed"] = self._get_class_prob(
                    self.level_2_stats_,
                    key2,
                    cls,
                    row_features[f"{g1}_class_{cls}_prob_smoothed"],
                )
                row_features[f"{g3}_class_{cls}_prob_smoothed"] = self._get_class_prob(
                    self.level_3_stats_,
                    key3,
                    cls,
                    row_features[f"{g2}_class_{cls}_prob_smoothed"],
                )
                row_features[f"geo_hierarchical_class_{cls}_prob"] = self._hierarchical_class_prob(
                    key1,
                    key2,
                    key3,
                    cls,
                )

            rows.append(row_features)

        return pd.DataFrame(rows, index=X.index)

    # =======================
    # FALLBACK GERARCHICO
    # =======================
    # Se una zona fine non e presente, si prova a risalire la gerarchia:
    # geo_level_3 -> geo_level_2 -> geo_level_1 -> globale.
    def _hierarchical_mean(self, key1: Any, key2: Any, key3: Any) -> float:
        if key3 in self.level_3_stats_.mean_map:
            return self.level_3_stats_.mean_map[key3]
        if key2 in self.level_2_stats_.mean_map:
            return self.level_2_stats_.mean_map[key2]
        if key1 in self.level_1_stats_.mean_map:
            return self.level_1_stats_.mean_map[key1]
        return self.global_mean_

    def _hierarchical_class_prob(self, key1: Any, key2: Any, key3: Any, cls: Any) -> float:
        if key3 in self.level_3_stats_.class_prob_maps:
            return self.level_3_stats_.class_prob_maps[key3][cls]
        if key2 in self.level_2_stats_.class_prob_maps:
            return self.level_2_stats_.class_prob_maps[key2][cls]
        if key1 in self.level_1_stats_.class_prob_maps:
            return self.level_1_stats_.class_prob_maps[key1][cls]
        return self.global_class_probs_[cls]

    # =======================
    # PREPARAZIONE INPUT TRAIN
    # =======================
    # Questi metodi trasformano X e y nel formato interno atteso:
    # - se y e separato, viene aggiunto al dataframe
    # - se il target e gia dentro X, viene riusato
    # - vengono controllate coerenza e cardinalita.
    def _prepare_training_frame(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series | pd.DataFrame],
    ) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X deve essere un pandas DataFrame.")

        if y is None:
            if self.target_col not in X.columns:
                raise ValueError(
                    f"Target non trovato. Passa y separatamente oppure includi '{self.target_col}' in X."
                )
            return X.copy()

        y_series = self._coerce_target(y)
        if len(y_series) != len(X):
            raise ValueError("X e y devono avere la stessa lunghezza.")

        df = X.copy()
        df[self.target_col] = y_series.to_numpy()
        return df

    def _coerce_target(self, y: pd.Series | pd.DataFrame) -> pd.Series:
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("y DataFrame deve contenere una sola colonna target.")
            return y.iloc[:, 0]
        if isinstance(y, pd.Series):
            return y
        raise TypeError("y deve essere un pandas Series o DataFrame.")

    # =======================
    # VALIDAZIONI
    # =======================
    # Controlli difensivi su:
    # - presenza delle colonne geografiche
    # - validita del target
    # - stato fitted dell'oggetto
    def _validate_geo_columns(self, df: pd.DataFrame) -> None:
        missing = [col for col in self.geo_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Colonne geografiche mancanti: {missing}")

    def _validate_target(self, y: pd.Series) -> None:
        if y.isnull().any():
            raise ValueError("Il target contiene NaN.")
        if y.nunique() < 2:
            raise ValueError("Il target deve avere almeno 2 classi.")

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("GeoFeatureEngineer non e ancora stato addestrato.")

    # =======================
    # UTILITY NUMERICHE E LOOKUP
    # =======================
    # Funzioni di supporto per:
    # - smoothing
    # - normalizzazione delle chiavi geografiche
    # - accesso sicuro alle mappe statistiche
    def _smooth_value(self, raw_value: float, parent_value: float, count: float) -> float:
        return float((count * raw_value + self.smoothing * parent_value) / (count + self.smoothing))

    def _normalize_group_key(self, key: Any) -> Any:
        if isinstance(key, tuple):
            return tuple(key)
        return key

    def _get_parent_key(self, key: Any, parent_keys: list[int]) -> Any:
        if not parent_keys:
            return None
        if not isinstance(key, tuple):
            return None
        if len(parent_keys) == 1:
            return key[parent_keys[0]]
        return tuple(key[idx] for idx in parent_keys)

    def _get_count(self, stats: _LevelStats, key: Any) -> float:
        return float(stats.count_map.get(key, 0.0))

    def _get_freq(self, stats: _LevelStats, key: Any) -> float:
        return float(stats.freq_map.get(key, 0.0))

    def _is_rare(self, stats: _LevelStats, key: Any) -> int:
        return int(key in stats.rare_values)

    def _get_mean(self, stats: _LevelStats, key: Any, fallback: float) -> float:
        return float(stats.mean_map.get(key, fallback))

    def _get_class_prob(self, stats: _LevelStats, key: Any, cls: Any, fallback: float) -> float:
        if key not in stats.class_prob_maps:
            return float(fallback)
        return float(stats.class_prob_maps[key].get(cls, fallback))
