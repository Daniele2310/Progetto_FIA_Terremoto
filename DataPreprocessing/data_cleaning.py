"""
Modulo DataQuality - Controllo duplicati, pulizia colonne e analisi outlier
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

COLONNE_CONTINUE = [
    "count_floors_pre_eq",
    "age",
    "area_percentage",
    "height_percentage",
    "count_families"
]

class DataQualityHandler:
    """Classe per il controllo qualità del dataset"""

    def __init__(self, percorso_file):
        self.percorso_file = Path(percorso_file)
        self.data = None
        self.report = {
            "shape": None,
            "duplicati_building_id": 0,
            "outliers": None
        }

    def carica_dati(self):
        """Carica il dataset dal percorso specificato."""
        self.data = pd.read_csv(self.percorso_file)

        print(f"Dataset caricato: {self.data.shape[0]} righe x {self.data.shape[1]} colonne")
        print("\nPrime righe del dataset:")
        print(self.data.head())

        self.report["shape"] = self.data.shape
        return self.data

    def pulisci_nomi_colonne(self):
        """Standardizza i nomi delle colonne in lowercase e rimuove spazi iniziali/finali"""
        if self.data is None:
            raise ValueError("I dati non sono stati ancora caricati.")

        self.data.columns = self.data.columns.str.lower().str.strip()

        print("\nNomi colonne standardizzati:")
        print(self.data.columns.tolist())

        return self.data

    def controlla_duplicati_building_id(self):
        """Controlla la presenza di duplicati nella colonna building_id"""
        if self.data is None:
            raise ValueError("I dati non sono stati ancora caricati.")

        num_duplicati = self.data["building_id"].duplicated().sum()
        self.report["duplicati_building_id"] = num_duplicati

        print(f"\nNumero di building_id duplicati: {num_duplicati}")
        return num_duplicati

    def analizza_outlier(self, colonne=COLONNE_CONTINUE):
        """Analizza gli outlier nelle colonne selezionate usando il metodo IQR"""
        if self.data is None:
            raise ValueError("I dati non sono stati ancora caricati.")

        outliers_summary = {}

        for col in colonne:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            mask_outliers = (self.data[col] < lower) | (self.data[col] > upper)
            n_outliers = mask_outliers.sum()
            perc_outliers = mask_outliers.mean() * 100

            outliers_summary[col] = {
                "Q1": Q1,
                "Q3": Q3,
                "IQR": IQR,
                "lower_bound": lower,
                "upper_bound": upper,
                "n_outliers": n_outliers,
                "perc_outliers": perc_outliers
            }

        outliers_df = pd.DataFrame(outliers_summary).T.round(2)
        self.report["outliers"] = outliers_df

        print("\nReport outlier:")
        print(outliers_df)

        return outliers_df

    def plot_boxplot(self, colonne=COLONNE_CONTINUE):
        """Mostra i boxplot delle colonne continue"""
        if self.data is None:
            raise ValueError("I dati non sono stati ancora caricati.")

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for i, feature in enumerate(colonne):
            axes[i].boxplot(self.data[feature].dropna(), vert=False)
            axes[i].set_title(feature)
            axes[i].set_xlabel("Value")
            axes[i].grid(axis="x", linestyle="--", alpha=0.5)

        for j in range(len(colonne), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


