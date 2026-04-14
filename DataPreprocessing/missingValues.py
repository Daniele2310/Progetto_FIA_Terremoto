"""
Modulo MissingValues - Controllo valori nulli e bilanciamento labels
"""

import numpy as np
import pandas as pd


class MissingValuesHandler:
    """Classe per il controllo e la gestione dei valori nulli nel dataset."""
    
    def __init__(self, null_threshold=70):
        self.null_threshold = null_threshold
        self.report = {}
    
    def analizza(self, df, target_col='damage_grade'):
        """Esegue l'analisi dei valori nulli e del bilanciamento."""
        self.report = {
            'total_nulls': 0,
            'sparse_cols': [],
            'sparse_rows': [],
            'label_distribution': None,
            'label_percent': None
        }
        
        total_nulls = df.isnull().sum().sum()
        self.report['total_nulls'] = total_nulls
        print(f"Valori nulli totali: {total_nulls}")
        
        null_percent_cols = (df.isnull().sum() / len(df)) * 100
        sparse_cols = null_percent_cols[null_percent_cols > self.null_threshold]
        
        if not sparse_cols.empty:
            print(f"Colonne con >{self.null_threshold}% nulli:")
            for col, percent in sparse_cols.items():
                print(f"  {col}: {percent:.1f}%")
            self.report['sparse_cols'] = sparse_cols.to_dict()
        
        null_percent_rows = (df.isnull().sum(axis=1) / df.shape[1]) * 100
        sparse_rows = null_percent_rows[null_percent_rows > self.null_threshold]
        
        if not sparse_rows.empty:
            print(f"Righe con >{self.null_threshold}% nulli: {len(sparse_rows)}")
            self.report['sparse_rows'] = sparse_rows.index.tolist()
        
        if target_col in df.columns:
            label_counts = df[target_col].value_counts().sort_index()
            label_percent = (df[target_col].value_counts(normalize=True) * 100).round(1).sort_index()
            
            print("\nDistribuzione label:")
            for label, count in label_counts.items():
                percent = label_percent[label]
                print(f"  Classe {label}: {count} ({percent}%)")
            
            self.report['label_distribution'] = label_counts.to_dict()
            self.report['label_percent'] = label_percent.to_dict()
        
        return self.report

    def sostituisci_range_con_nan(self, df, colonna='age', min_val=250, max_val=995):
        """
        Sostituisce con NaN i valori compresi nel range [min_val, max_val] su una colonna.
        Ritorna dataframe aggiornato, maschera valori sostituiti e numero sostituzioni.
        """
        if colonna not in df.columns:
            raise ValueError(f"La colonna '{colonna}' non è presente nel dataframe.")

        df_out = df.copy()
        mask_range = df_out[colonna].between(min_val, max_val, inclusive='both')
        n_sostituiti = int(mask_range.sum())

        df_out.loc[mask_range, colonna] = np.nan

        return df_out, mask_range, n_sostituiti

    def _fit_mediane_gerarchiche(self, train_df, colonna, gerarchia_gruppi):
        """Costruisce mappe di mediane per ogni livello di gerarchia."""
        mappe = []

        for cols in gerarchia_gruppi:
            if not all(c in train_df.columns for c in cols):
                continue

            mediane = train_df.groupby(cols)[colonna].median().dropna()
            mappa = {}

            for chiave, valore in mediane.items():
                if not isinstance(chiave, tuple):
                    chiave = (chiave,)
                mappa[chiave] = float(valore)

            mappe.append({
                'cols': cols,
                'mappa_mediane': mappa
            })

        return mappe

    def _applica_mediane_gerarchiche(self, df, colonna, mappe_gerarchiche, mediana_globale):
        """
        Applica l'imputazione gerarchica: prima mediane per gruppo,
        poi fallback alla mediana globale.
        """
        df_out = df.copy()
        riempiti_per_livello = {}

        missing_idx = df_out.index[df_out[colonna].isna()]

        for livello in mappe_gerarchiche:
            if len(missing_idx) == 0:
                break

            cols = livello['cols']
            mappa_mediane = livello['mappa_mediane']

            chiavi = [tuple(v) for v in df_out.loc[missing_idx, cols].to_numpy()]
            valori_imputati = pd.Series(
                [mappa_mediane.get(k, np.nan) for k in chiavi],
                index=missing_idx
            )

            idx_da_riempire = valori_imputati[valori_imputati.notna()].index
            if len(idx_da_riempire) > 0:
                df_out.loc[idx_da_riempire, colonna] = valori_imputati.loc[idx_da_riempire]

            riempiti_per_livello[' + '.join(cols)] = int(len(idx_da_riempire))
            missing_idx = df_out.index[df_out[colonna].isna()]

        n_fallback_globale = int(len(missing_idx))
        if n_fallback_globale > 0:
            df_out.loc[missing_idx, colonna] = mediana_globale

        return df_out, riempiti_per_livello, n_fallback_globale

    def imputa_mediana_multivariata(self, train_df, test_df, colonna='age', gerarchia_gruppi=None):
        """
        Imputazione con mediane per gruppi (gerarchia) calcolate sul train.
        Se un gruppo non esiste, usa fallback alla mediana globale del train.
        """
        if colonna not in train_df.columns or colonna not in test_df.columns:
            raise ValueError(f"La colonna '{colonna}' deve essere presente in train e test.")

        if gerarchia_gruppi is None:
            gerarchia_gruppi = [
                ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id'],
                ['geo_level_1_id', 'geo_level_2_id'],
                ['geo_level_1_id']
            ]

        mediana_globale = train_df[colonna].median()
        if pd.isna(mediana_globale):
            raise ValueError(f"Impossibile calcolare la mediana della colonna '{colonna}' sul train.")

        mediana_globale = float(mediana_globale)
        mappe_gerarchiche = self._fit_mediane_gerarchiche(train_df, colonna, gerarchia_gruppi)

        n_missing_train_prima = int(train_df[colonna].isna().sum())
        n_missing_test_prima = int(test_df[colonna].isna().sum())

        train_out, train_riempiti_livello, train_fallback = self._applica_mediane_gerarchiche(
            train_df,
            colonna,
            mappe_gerarchiche,
            mediana_globale
        )

        test_out, test_riempiti_livello, test_fallback = self._applica_mediane_gerarchiche(
            test_df,
            colonna,
            mappe_gerarchiche,
            mediana_globale
        )

        n_missing_train_dopo = int(train_out[colonna].isna().sum())
        n_missing_test_dopo = int(test_out[colonna].isna().sum())

        report = {
            'strategia': 'multivariata',
            'colonna': colonna,
            'mediana_globale_train': mediana_globale,
            'livelli_usati': [' + '.join(x['cols']) for x in mappe_gerarchiche],
            'train_riempiti_per_livello': train_riempiti_livello,
            'test_riempiti_per_livello': test_riempiti_livello,
            'train_fallback_globale': train_fallback,
            'test_fallback_globale': test_fallback,
            'n_missing_train_prima': n_missing_train_prima,
            'n_missing_train_dopo': n_missing_train_dopo,
            'n_missing_test_prima': n_missing_test_prima,
            'n_missing_test_dopo': n_missing_test_dopo
        }

        return train_out, test_out, report

    

