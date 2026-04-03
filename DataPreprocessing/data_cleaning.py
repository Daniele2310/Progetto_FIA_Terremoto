import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

COLONNE_CONTINUE = [
    "count_floors_pre_eq",
    "age",
    "area_percentage",
    "height_percentage",
    "count_families"
]

COLONNE_DA_STANDARDIZZARE = [
    "count_floors_pre_eq",
    "age",
    "area_percentage",
    "height_percentage",
    "count_families"
]


class DataQualityHandler:
    """Classe per il controllo qualità del dataset."""

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.scaler = None
        self.report = {
            "shape": self.data.shape,
            "duplicati_building_id": 0,
            "outliers": None,
            "normalizzazione": None
        }

    def pulisci_nomi_colonne(self):
        """Standardizza i nomi delle colonne."""
        self.data.columns = self.data.columns.str.lower().str.strip()

        return self.data

    def controlla_duplicati_building_id(self):
        """Controlla la presenza di duplicati nella colonna building_id."""
        if "building_id" not in self.data.columns:
            raise ValueError("La colonna 'building_id' non è presente nel dataset.")

        num_duplicati = self.data["building_id"].duplicated().sum()
        self.report["duplicati_building_id"] = num_duplicati

        print(f"\nNumero di building_id duplicati: {num_duplicati}")
        return num_duplicati

    def analizza_outlier(self, colonne=COLONNE_CONTINUE):
        """Analizza gli outlier nelle colonne selezionate usando il metodo IQR."""
        outliers_summary = {}

        for col in colonne:
            if col not in self.data.columns:
                print(f"Colonna '{col}' non presente nel dataset, salto.")
                continue

            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1

            # GESTIONE IQR = 0 E CODE LUNGHE
            if IQR == 0:
                # 1. Calcoliamo la frequenza relativa di ogni valore presente nella colonna
                frequenze = self.data[col].value_counts(normalize=True)

                # 2. Definiamo una soglia di rarità rigorosa (es. meno dello 0.5% del dataset)
                soglia_rarita = 0.005

                # 3. Troviamo quali sono i valori "rari"
                valori_anomali = frequenze[frequenze < soglia_rarita].index

                # 4. Creiamo la maschera per gli outlier basata sulla rarità
                mask_outliers = self.data[col].isin(valori_anomali)

                # Nota: lower e upper qui perdono di significato continuo,
                # ma possiamo impostarli ai valori min/max non rari per compatibilità con il tuo dizionario
                valori_normali = frequenze[frequenze >= soglia_rarita].index
                lower = valori_normali.min()
                upper = valori_normali.max()
            else:
                # Usiamo 3.0 (Extreme IQR) invece di 1.5 per le distribuzioni asimmetriche fisiche
                # in alternativa, potresti mantenere 1.5 ma calcolarlo su np.log1p(valid_data)
                lower = Q1 - 3.0 * IQR
                upper = Q3 + 3.0 * IQR
                # lower = Q1 - 1.5 * IQR
                # upper = Q3 + 1.5 * IQR

                lower = max(0, lower)
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

        return outliers_df



    def plot_boxplot(self, colonne=COLONNE_CONTINUE):
        """Mostra i boxplot delle colonne continue."""
        colonne_presenti = [col for col in colonne if col in self.data.columns]

        if not colonne_presenti:
            print("Nessuna colonna continua disponibile per il boxplot.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for i, feature in enumerate(colonne_presenti):
            axes[i].boxplot(self.data[feature].dropna(), vert=False)
            axes[i].set_title(feature)
            axes[i].set_xlabel("Value")
            axes[i].grid(axis="x", linestyle="--", alpha=0.5)

        for j in range(len(colonne_presenti), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def aggiungi_feature_age_flag(self, upper_bound, max_age_considerata=250):
        """
        Aggiunge una feature booleana:
        - 1 se age > upper_bound e age <= max_age_considerata
        - 0 altrimenti

        Parametri
        ----------
        upper_bound : float
            Upper bound calcolato sull'età dal training set.
        max_age_considerata : int, default=250
            Valore massimo di age da considerare per assegnare il flag.
            Se age > 250, il flag resta 0.
        """
        if "age" not in self.data.columns:
            raise ValueError("La colonna 'age' non è presente nel dataset.")

        self.data["monum_flag"] = (
                (self.data["age"] > upper_bound) &
                (self.data["age"] <= max_age_considerata)
        ).astype(int)

        counts = self.data["monum_flag"].value_counts()

        n_1 = counts.get(1, 0)
        n_0 = counts.get(0, 0)

        print(
            f"Feature 'monum_flag' aggiunta usando upper_bound={upper_bound:.3f} "
            f"e max_age_considerata={max_age_considerata}\n"
            f"Valori 1: {n_1}\n"
            f"Valori 0: {n_0}"
        )

        return self.data


    def fit_standardizzazione(self, colonne=COLONNE_DA_STANDARDIZZARE):
        """
        Esegue il fit dello StandardScaler sul training set.
        """
        colonne_presenti = [col for col in colonne if col in self.data.columns]

        if not colonne_presenti:
            raise ValueError("Nessuna colonna da standardizzare presente nel dataset.")

        self.scaler = StandardScaler()
        self.scaler.fit(self.data[colonne_presenti])

        self.report["standardizzazione"] = {
            col: {
                "mean_train": float(self.scaler.mean_[i]),
                "std_train": float(self.scaler.scale_[i])
            }
            for i, col in enumerate(colonne_presenti)
        }

        for col, valori in self.report["standardizzazione"].items():
            print(
                f"Parametri salvati per '{col}': "
                f"mean={valori['mean_train']:.4f}, std={valori['std_train']:.4f}"
            )

        return self.scaler

    def applica_standardizzazione(self, scaler=None, colonne=COLONNE_DA_STANDARDIZZARE):
        """
        Applica la standardizzazione usando uno scaler già fittato.
        """
        if scaler is None:
            scaler = self.scaler

        if scaler is None:
            raise ValueError("Scaler non disponibile. Esegui prima fit_standardizzazione() sul train.")

        colonne_presenti = [col for col in colonne if col in self.data.columns]

        if not colonne_presenti:
            print("Nessuna colonna da standardizzare presente nel dataset.")
            return self.data

        self.data[colonne_presenti] = scaler.transform(self.data[colonne_presenti])

        print(f"Standardizzazione applicata alle colonne: {colonne_presenti}")
        return self.data

    def esegui_controlli(self, plot=False):
        """Esegue i controlli principali di data quality."""
        self.pulisci_nomi_colonne()
        self.controlla_duplicati_building_id()
        self.analizza_outlier()

        if plot:
            self.plot_boxplot()

        return self.report