from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class PCAHandler:
    DEFAULT_EXCLUDE_COLUMNS = (
        "building_id",
        "geo_level_1_id",
        "geo_level_2_id",
        "geo_level_3_id",
        "damage_grade",
    )

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

    @classmethod
    def _build_exclude_columns(cls, exclude_columns: Optional[Iterable[str]] = None) -> list[str]:
        columns = list(cls.DEFAULT_EXCLUDE_COLUMNS)
        columns.extend(cls._normalize_columns_arg(exclude_columns))
        return list(dict.fromkeys(columns))

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

    @staticmethod
    def _existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
        return [col for col in columns if col in df.columns]

    @staticmethod
    def _validate_n_components(n_components: int, max_components: int) -> int:
        if not (1 <= n_components <= max_components):
            raise ValueError(
                f"Il numero di componenti deve stare tra 1 e {max_components}."
            )
        return n_components

    def _prompt_n_components(self, max_components: int) -> int:
        print("\n" + "=" * 80)
        print("SCELTA COMPONENTI PCA")
        print("=" * 80)
        print("Osserva lo scree plot e scegli il numero di componenti")
        print("nel punto di gomito.")

        try:
            raw_value = input(
                f"Inserisci il numero di componenti da mantenere [1-{max_components}]: "
            ).strip()
        except EOFError:
            raw_value = ""

        if not raw_value:
            raise ValueError("Devi inserire il numero di componenti PCA da mantenere.")

        try:
            n_components = int(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Il numero di componenti PCA deve essere un intero tra 1 e {max_components}."
            ) from exc

        return self._validate_n_components(n_components, max_components=max_components)

    def fit(self, df: pd.DataFrame, exclude_columns: Optional[Iterable[str]] = None):
        exclude_columns = self._build_exclude_columns(exclude_columns)
        self.feature_columns_ = [col for col in df.columns if col not in exclude_columns]

        X = self._prepare_matrix(df, self.feature_columns_)
        self.pca.fit(X)
        self.fitted_ = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        preserve_columns: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
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

    def explained_variance(self) -> pd.Series:
        self._check_fitted()
        values = self.pca.explained_variance_
        return pd.Series(values, index=[f"PC{i+1}" for i in range(len(values))])

    def cumulative_explained_variance(self) -> pd.Series:
        self._check_fitted()
        cumulative = np.cumsum(self.pca.explained_variance_)
        return pd.Series(cumulative, index=[f"PC{i+1}" for i in range(len(cumulative))])

    def build_variance_table(self) -> pd.DataFrame:
        self._check_fitted()
        variance_table = pd.DataFrame(
            {
                "explained_variance": self.explained_variance().round(6),
                "cumulative_explained_variance": self.cumulative_explained_variance().round(6),
            }
        )
        variance_table["delta_explained_variance"] = (
            variance_table["explained_variance"].diff().round(6)
        )
        variance_table.loc[variance_table.index[0], "delta_explained_variance"] = np.nan
        return variance_table

    def get_loadings(self) -> pd.DataFrame:
        self._check_fitted()

        return pd.DataFrame(
            self.pca.components_,
            index=[f"PC{i+1}" for i in range(self.pca.components_.shape[0])],
            columns=self.feature_columns_,
        )

    def build_report(self, selected_n: Optional[int] = None) -> dict:
        self._check_fitted()

        variance_table = self.build_variance_table()

        return {
            "n_components_input": self.n_components,
            "n_components_calcolate": int(len(variance_table)),
            "n_components_selezionate": int(selected_n) if selected_n is not None else None,
            "feature_usate_per_pca": list(self.feature_columns_),
            "explained_variance": variance_table["explained_variance"].to_dict(),
            "cumulative_explained_variance": (
                variance_table["cumulative_explained_variance"].to_dict()
            ),
        }

    def plot_scree(
        self,
        output_path: Optional[str | Path] = None,
        selected_n: Optional[int] = None,
        show_plot: bool = True,
    ) -> None:
        self._check_fitted()

        explained_variance = self.explained_variance()
        component_numbers = np.arange(1, len(explained_variance) + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            component_numbers,
            explained_variance.values,
            marker="o",
            label="Varianza spiegata",
        )

        if selected_n is not None:
            self._validate_n_components(selected_n, max_components=len(component_numbers))
            ax.axvline(
                x=selected_n,
                linestyle="--",
                color="red",
                label=f"Componenti scelte: {selected_n}",
            )

        ax.set_xlabel("Numero di componenti principali")
        ax.set_ylabel("Varianza")
        ax.set_ylim(bottom=0)
        ax.set_title("Scree Plot PCA")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path)

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def run_interactive_pipeline(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        labels_df: Optional[pd.DataFrame] = None,
        output_dir: str | Path = "DataPreprocessed",
        id_columns: Optional[Iterable[str]] = None,
        target_column: str = "damage_grade",
        exclude_columns: Optional[Iterable[str]] = None,
    ) -> dict:
        if train_df.isnull().any().any() or test_df.isnull().any().any():
            raise ValueError(
                "Sono presenti valori NaN: completa imputazione/pulizia prima di applicare PCA."
            )

        id_columns = self._normalize_columns_arg(id_columns)
        exclude_columns = self._build_exclude_columns(exclude_columns)

        train_preserve_columns = self._existing_columns(train_df, id_columns)
        test_preserve_columns = self._existing_columns(test_df, id_columns)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        variance_output_path = output_dir / "pca_variance_summary.csv"
        loadings_output_path = output_dir / "pca_loadings.csv"
        scree_plot_output_path = output_dir / "scree_plot.png"

        self.n_components = None
        self.pca = PCA(n_components=None)
        self.fit(train_df, exclude_columns=exclude_columns)

        variance_table = self.build_variance_table()
        variance_table.to_csv(variance_output_path)

        print("\n" + "=" * 80)
        print("TABELLA VARIANZA SPIEGATA PCA")
        print("=" * 80)
        print(variance_table.to_string())

        print("\n" + "=" * 80)
        print("LOG SUPPORTO SCELTA GOMITO PCA")
        print("=" * 80)
        for component_name, row in variance_table.iterrows():
            delta_value = row["delta_explained_variance"]
            if pd.isna(delta_value):
                delta_text = "n/a"
            else:
                delta_text = f"{delta_value:.6f}"

            print(
                f"{component_name}: "
                f"varianza={row['explained_variance']:.6f}, "
                f"cumulata={row['cumulative_explained_variance']:.6f}, "
                f"delta_vs_precedente={delta_text}"
            )

        self.plot_scree(
            output_path=scree_plot_output_path,
            show_plot=True,
        )

        selected_n = self._prompt_n_components(max_components=len(variance_table))

        self.n_components = selected_n
        self.pca = PCA(n_components=selected_n)
        self.fit(train_df, exclude_columns=exclude_columns)

        train_transformed = self.transform(
            train_df,
            preserve_columns=train_preserve_columns,
        )
        test_transformed = self.transform(
            test_df,
            preserve_columns=test_preserve_columns,
        )

        loadings = self.get_loadings()
        loadings.to_csv(loadings_output_path)

        self.plot_scree(
            output_path=scree_plot_output_path,
            selected_n=selected_n,
            show_plot=False,
        )

        report = self.build_report(selected_n=selected_n)
        report["excluded_columns"] = exclude_columns
        report["id_columns_preserved_train"] = train_preserve_columns
        report["id_columns_preserved_test"] = test_preserve_columns
        report["variance_table_output_path"] = str(variance_output_path)
        report["loadings_output_path"] = str(loadings_output_path)
        report["scree_plot_output_path"] = str(scree_plot_output_path)

        train_with_target = None
        if labels_df is not None:
            merge_columns = self._existing_columns(labels_df, id_columns)
            if target_column not in labels_df.columns:
                raise ValueError(
                    f"La colonna target '{target_column}' non e presente nel dataframe delle label."
                )
            if not merge_columns:
                raise ValueError(
                    "Non e stata trovata alcuna colonna identificativa comune per riallegare le label."
                )

            label_frame = labels_df[merge_columns + [target_column]].copy()
            train_with_target = pd.merge(
                train_transformed,
                label_frame,
                on=merge_columns,
                how="inner",
            )
        elif target_column in train_df.columns:
            keep_columns = self._existing_columns(train_df, id_columns) + [target_column]
            train_with_target = pd.merge(
                train_transformed,
                train_df[keep_columns].copy(),
                on=self._existing_columns(train_df, id_columns),
                how="inner",
            )

        print("\n" + "=" * 80)
        print("PCA COMPLETATA")
        print("=" * 80)
        print("Colonne escluse dalla PCA e poi preservate/riallegate se presenti:")
        print(exclude_columns)
        print(f"Numero componenti risultante: {selected_n}")
        print(
            "Varianza cumulativa finale raggiunta: "
            f"{self.cumulative_explained_variance().iloc[-1]:.6f}"
        )
        print(f"Nuove dimensioni train: {train_transformed.shape}")
        print(f"Nuove dimensioni test: {test_transformed.shape}")
        print(f"Tabella varianza salvata in: {variance_output_path}")
        print(f"Loadings salvati in: {loadings_output_path}")
        print(f"Scree plot salvato in: {scree_plot_output_path}")

        return {
            "train_transformed": train_transformed,
            "test_transformed": test_transformed,
            "train_with_target": train_with_target,
            "report": report,
            "selected_n_components": selected_n,
        }
