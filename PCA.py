from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class PCAHandler:
    def __init__(self, n_components: Optional[int | float] = None):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.feature_columns_: list[str] = []
        self.fitted_: bool = False

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise ValueError("Il PCAHandler non e stato ancora addestrato.")

    @staticmethod
    def _normalize_columns_arg(columns: Optional[Iterable[str]]) -> list[str]:
        if columns is None:
            return []
        return list(columns)

    @staticmethod
    def _validate_threshold(threshold: float) -> float:
        if not (0 < threshold <= 1):
            raise ValueError("threshold deve stare in (0,1]")
        return threshold

    @staticmethod
    def _prepare_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        if not feature_columns:
            raise ValueError("Nessuna feature disponibile per il PCA.")

        X = df[feature_columns]

        if not all(np.issubdtype(dtype, np.number) for dtype in X.dtypes):
            raise ValueError("Il PCA richiede input numerico.")

        if X.isnull().any().any():
            raise ValueError("Il PCA richiede input senza valori NaN.")

        return X

    def fit(self, df: pd.DataFrame, exclude_columns: Optional[Iterable[str]] = None):
        exclude_columns = self._normalize_columns_arg(exclude_columns)
        self.feature_columns_ = [col for col in df.columns if col not in exclude_columns]

        X = self._prepare_matrix(df, self.feature_columns_)
        self.pca.fit(X)
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame, preserve_columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
        self._check_fitted()
        preserve_columns = self._normalize_columns_arg(preserve_columns)

        missing_cols = [col for col in self.feature_columns_ if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colonne mancanti: {missing_cols}")

        missing_preserve_cols = [col for col in preserve_columns if col not in df.columns]
        if missing_preserve_cols:
            raise ValueError(f"Colonne da preservare mancanti: {missing_preserve_cols}")

        X = self._prepare_matrix(df, self.feature_columns_)
        X_pca = self.pca.transform(X)

        pca_columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=df.index)

        if preserve_columns:
            return pd.concat([df[preserve_columns], pca_df], axis=1)

        return pca_df

    def fit_transform(
        self,
        df: pd.DataFrame,
        exclude_columns: Optional[Iterable[str]] = None,
        preserve_columns: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        self.fit(df=df, exclude_columns=exclude_columns)
        return self.transform(df=df, preserve_columns=preserve_columns)

    def explained_variance_ratio(self) -> pd.Series:
        self._check_fitted()
        values = self.pca.explained_variance_ratio_
        return pd.Series(values, index=[f"PC{i+1}" for i in range(len(values))])

    def cumulative_explained_variance(self) -> pd.Series:
        self._check_fitted()
        cumulative = np.cumsum(self.pca.explained_variance_ratio_)
        return pd.Series(cumulative, index=[f"PC{i+1}" for i in range(len(cumulative))])

    def choose_n_components(self, threshold: float = 0.95) -> int:
        self._check_fitted()
        threshold = self._validate_threshold(threshold)

        cumulative = np.cumsum(self.pca.explained_variance_ratio_)
        return int(np.argmax(cumulative >= threshold) + 1)

    def get_loadings(self) -> pd.DataFrame:
        self._check_fitted()

        return pd.DataFrame(
            self.pca.components_,
            index=[f"PC{i+1}" for i in range(self.pca.components_.shape[0])],
            columns=self.feature_columns_,
        )

    def build_report(self, threshold: Optional[float] = None) -> dict:
        self._check_fitted()

        explained = self.explained_variance_ratio()
        cumulative = self.cumulative_explained_variance()

        if threshold is not None:
            threshold = self._validate_threshold(threshold)
            suggested = self.choose_n_components(threshold=threshold)
            threshold_value = float(threshold)
            suggested_value = int(suggested)
        else:
            threshold_value = None
            suggested_value = None

        return {
            "n_components_input": self.n_components,
            "n_components_calcolate": int(len(explained)),
            "threshold": threshold_value,
            "n_components_consigliate": suggested_value,
            "explained_variance_ratio": explained.to_dict(),
            "cumulative_explained_variance": cumulative.to_dict(),
        }

    def plot_scree(
        self,
        output_path: Optional[str | Path] = None,
        threshold: Optional[float] = None,
        selected_n: Optional[int] = None,
        show_plot: bool = True,
    ) -> None:
        self._check_fitted()

        variance = self.pca.explained_variance_ratio_
        cumulative = np.cumsum(variance)
        x = np.arange(1, len(variance) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(x, variance, marker="o", label="Varianza singola")
        plt.plot(x, cumulative, marker="o", label="Varianza cumulativa")

        if threshold is not None:
            threshold = self._validate_threshold(threshold)
            plt.axhline(y=threshold, linestyle="--", label=f"Soglia {threshold}")

        if selected_n is not None:
            if not (1 <= selected_n <= len(x)):
                raise ValueError(f"selected_n deve stare tra 1 e {len(x)}.")
            plt.axvline(
                x=selected_n,
                linestyle="--",
                color="red",
                label=f"Componenti scelte: {selected_n}",
            )

        plt.xlabel("Componenti")
        plt.ylabel("Varianza")
        plt.title("Scree Plot")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)

        if show_plot:
            plt.show()
        else:
            plt.close()

    def refit_with_threshold(
        self,
        train_df: pd.DataFrame,
        threshold: float = 0.95,
        exclude_columns: Optional[Iterable[str]] = None,
    ) -> int:
        threshold = self._validate_threshold(threshold)
        exclude_columns = self._normalize_columns_arg(exclude_columns)

        full_pca = PCA()
        feature_columns = [col for col in train_df.columns if col not in exclude_columns]
        X_train = self._prepare_matrix(train_df, feature_columns)
        full_pca.fit(X_train)

        cumulative = np.cumsum(full_pca.explained_variance_ratio_)
        selected_n = int(np.argmax(cumulative >= threshold) + 1)

        self.n_components = selected_n
        self.pca = PCA(n_components=selected_n)
        self.fit(train_df, exclude_columns=exclude_columns)

        return selected_n

