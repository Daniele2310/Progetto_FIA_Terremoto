"""
Modulo MissingValues - Controllo valori nulli e bilanciamento labels
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


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

    def imputa_univariata_media(self, train_df, test_df, colonna='age'):
        """
        Imputazione univariata con media calcolata sul train.
        """
        if colonna not in train_df.columns or colonna not in test_df.columns:
            raise ValueError(f"La colonna '{colonna}' deve essere presente in train e test.")

        media_train = train_df[colonna].mean()
        if pd.isna(media_train):
            raise ValueError(f"Impossibile calcolare la media della colonna '{colonna}' sul train.")

        media_train = float(media_train)

        train_out = train_df.copy()
        test_out = test_df.copy()

        n_missing_train_prima = int(train_out[colonna].isna().sum())
        n_missing_test_prima = int(test_out[colonna].isna().sum())

        train_out[colonna] = train_out[colonna].fillna(media_train)
        test_out[colonna] = test_out[colonna].fillna(media_train)

        n_missing_train_dopo = int(train_out[colonna].isna().sum())
        n_missing_test_dopo = int(test_out[colonna].isna().sum())

        report = {
            'strategia': 'univariata_media',
            'colonna': colonna,
            'valore_imputazione_train': media_train,
            'n_missing_train_prima': n_missing_train_prima,
            'n_missing_train_dopo': n_missing_train_dopo,
            'n_missing_test_prima': n_missing_test_prima,
            'n_missing_test_dopo': n_missing_test_dopo
        }

        return train_out, test_out, report

    def imputa_univariata_mediana(self, train_df, test_df, colonna='age'):
        """
        Imputazione univariata con mediana calcolata sul train.
        """
        if colonna not in train_df.columns or colonna not in test_df.columns:
            raise ValueError(f"La colonna '{colonna}' deve essere presente in train e test.")

        mediana_train = train_df[colonna].median()
        if pd.isna(mediana_train):
            raise ValueError(f"Impossibile calcolare la mediana della colonna '{colonna}' sul train.")

        mediana_train = float(mediana_train)

        train_out = train_df.copy()
        test_out = test_df.copy()

        n_missing_train_prima = int(train_out[colonna].isna().sum())
        n_missing_test_prima = int(test_out[colonna].isna().sum())

        train_out[colonna] = train_out[colonna].fillna(mediana_train)
        test_out[colonna] = test_out[colonna].fillna(mediana_train)

        n_missing_train_dopo = int(train_out[colonna].isna().sum())
        n_missing_test_dopo = int(test_out[colonna].isna().sum())

        report = {
            'strategia': 'univariata_mediana',
            'colonna': colonna,
            'valore_imputazione_train': mediana_train,
            'n_missing_train_prima': n_missing_train_prima,
            'n_missing_train_dopo': n_missing_train_dopo,
            'n_missing_test_prima': n_missing_test_prima,
            'n_missing_test_dopo': n_missing_test_dopo
        }

        return train_out, test_out, report

    def imputa_univariata_media_mediana(self, train_df, test_df, colonna='age'):
        """
        Richiama entrambe le imputazioni univariate: media e mediana.
        """
        train_media, test_media, report_media = self.imputa_univariata_media(
            train_df=train_df,
            test_df=test_df,
            colonna=colonna
        )

        train_mediana, test_mediana, report_mediana = self.imputa_univariata_mediana(
            train_df=train_df,
            test_df=test_df,
            colonna=colonna
        )

        return {
            'media': {
                'train': train_media,
                'test': test_media,
                'report': report_media
            },
            'mediana': {
                'train': train_mediana,
                'test': test_mediana,
                'report': report_mediana
            }
        }

    def _seleziona_feature_numeriche(self, df, colonna_target, feature_cols=None):
        """
        Seleziona feature numeriche da usare come predittori.
        """
        if feature_cols is None:
            feature_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in {colonna_target, 'building_id'}
            ]

        feature_cols = [c for c in feature_cols if c in df.columns and c != colonna_target]

        if not feature_cols:
            raise ValueError(
                f"Nessuna feature numerica disponibile per predire '{colonna_target}'."
            )

        return feature_cols

    def _prepara_predictor_numerici(self, train_df, test_df, feature_cols):
        """
        Prepara i predittori numerici riempiendo i missing con mediana del train.
        """
        X_train = train_df[feature_cols].copy()
        X_test = test_df[feature_cols].copy()

        mediane = X_train.median(numeric_only=True)
        X_train = X_train.fillna(mediane)
        X_test = X_test.fillna(mediane)

        return X_train, X_test

    def imputa_multivariata_regressione_lineare(self, train_df, test_df, colonna='age', feature_cols=None):
        """
        Imputazione multivariata con regressione lineare sui predittori numerici.
        """
        if colonna not in train_df.columns or colonna not in test_df.columns:
            raise ValueError(f"La colonna '{colonna}' deve essere presente in train e test.")

        feature_cols = self._seleziona_feature_numeriche(train_df, colonna, feature_cols)
        X_train, X_test = self._prepara_predictor_numerici(train_df, test_df, feature_cols)

        train_out = train_df.copy()
        test_out = test_df.copy()

        mask_target_nota = train_out[colonna].notna()
        n_noti = int(mask_target_nota.sum())

        if n_noti < 2:
            raise ValueError(
                f"Valori noti insufficienti in '{colonna}' per la regressione lineare (trovati: {n_noti})."
            )

        model = LinearRegression()
        model.fit(X_train.loc[mask_target_nota], train_out.loc[mask_target_nota, colonna])

        n_missing_train_prima = int(train_out[colonna].isna().sum())
        n_missing_test_prima = int(test_out[colonna].isna().sum())

        mask_missing_train = train_out[colonna].isna()
        if mask_missing_train.any():
            pred_train = model.predict(X_train.loc[mask_missing_train])
            train_out.loc[mask_missing_train, colonna] = np.clip(pred_train, a_min=0, a_max=None)

        mask_missing_test = test_out[colonna].isna()
        if mask_missing_test.any():
            pred_test = model.predict(X_test.loc[mask_missing_test])
            test_out.loc[mask_missing_test, colonna] = np.clip(pred_test, a_min=0, a_max=None)

        n_missing_train_dopo = int(train_out[colonna].isna().sum())
        n_missing_test_dopo = int(test_out[colonna].isna().sum())

        report = {
            'strategia': 'multivariata_regressione_lineare',
            'colonna': colonna,
            'n_feature_usate': len(feature_cols),
            'feature_usate': feature_cols,
            'n_missing_train_prima': n_missing_train_prima,
            'n_missing_train_dopo': n_missing_train_dopo,
            'n_missing_test_prima': n_missing_test_prima,
            'n_missing_test_dopo': n_missing_test_dopo
        }

        return train_out, test_out, report

    def imputa_knn_predictor(self, train_df, test_df, colonna='age', feature_cols=None, n_neighbors=5):
        """
        Imputazione con KNN Regressor sui predittori numerici.
        """
        if colonna not in train_df.columns or colonna not in test_df.columns:
            raise ValueError(f"La colonna '{colonna}' deve essere presente in train e test.")

        feature_cols = self._seleziona_feature_numeriche(train_df, colonna, feature_cols)
        X_train, X_test = self._prepara_predictor_numerici(train_df, test_df, feature_cols)

        train_out = train_df.copy()
        test_out = test_df.copy()

        mask_target_nota = train_out[colonna].notna()
        n_noti = int(mask_target_nota.sum())

        if n_noti < 2:
            raise ValueError(
                f"Valori noti insufficienti in '{colonna}' per KNN predictor (trovati: {n_noti})."
            )

        n_neighbors_eff = max(1, min(int(n_neighbors), n_noti))

        model = KNeighborsRegressor(
            n_neighbors=n_neighbors_eff,
            weights='distance',
            algorithm='brute',
            n_jobs=-1
        )
        model.fit(X_train.loc[mask_target_nota], train_out.loc[mask_target_nota, colonna])

        n_missing_train_prima = int(train_out[colonna].isna().sum())
        n_missing_test_prima = int(test_out[colonna].isna().sum())

        mask_missing_train = train_out[colonna].isna()
        if mask_missing_train.any():
            pred_train = model.predict(X_train.loc[mask_missing_train])
            train_out.loc[mask_missing_train, colonna] = np.clip(pred_train, a_min=0, a_max=None)

        mask_missing_test = test_out[colonna].isna()
        if mask_missing_test.any():
            pred_test = model.predict(X_test.loc[mask_missing_test])
            test_out.loc[mask_missing_test, colonna] = np.clip(pred_test, a_min=0, a_max=None)

        n_missing_train_dopo = int(train_out[colonna].isna().sum())
        n_missing_test_dopo = int(test_out[colonna].isna().sum())

        report = {
            'strategia': 'knn_predictor',
            'colonna': colonna,
            'n_feature_usate': len(feature_cols),
            'feature_usate': feature_cols,
            'n_neighbors': n_neighbors_eff,
            'n_missing_train_prima': n_missing_train_prima,
            'n_missing_train_dopo': n_missing_train_dopo,
            'n_missing_test_prima': n_missing_test_prima,
            'n_missing_test_dopo': n_missing_test_dopo
        }

        return train_out, test_out, report

    def _estrai_target(self, train_df, train_labels, target_col='damage_grade'):
        """
        Estrae il target allineato al train dataframe.
        """
        if isinstance(train_labels, pd.Series):
            y = train_labels.reset_index(drop=True)
            if len(y) != len(train_df):
                raise ValueError('La Series train_labels non ha la stessa lunghezza del train_df.')
            return y

        if not isinstance(train_labels, pd.DataFrame):
            raise ValueError('train_labels deve essere una Series o DataFrame.')

        if target_col not in train_labels.columns:
            raise ValueError(f"La colonna target '{target_col}' non e presente in train_labels.")

        if 'building_id' in train_labels.columns and 'building_id' in train_df.columns:
            mappa_target = train_labels.set_index('building_id')[target_col]
            y = train_df['building_id'].map(mappa_target)
            return y.reset_index(drop=True)

        y = train_labels[target_col].reset_index(drop=True)
        if len(y) != len(train_df):
            raise ValueError('train_labels non e allineabile a train_df per lunghezza.')

        return y

    def valuta_strategie_con_knn_veloce(
        self,
        train_df,
        train_labels,
        colonna='age',
        target_col='damage_grade',
        strategie=None,
        test_size=0.2,
        random_state=42,
        max_rows=20000,
        n_neighbors_valutazione=5
    ):
        """
        Valuta le strategie di imputazione con un KNN classifier veloce.
        Ritorna una tabella con accuracy per confrontare i metodi.
        """
        if colonna not in train_df.columns:
            raise ValueError(f"La colonna '{colonna}' non e presente nel train_df.")

        if strategie is None:
            strategie = [
                'univariata_media',
                'univariata_mediana',
                'multivariata_regressione_lineare',
                'knn_predictor'
            ]

        y = self._estrai_target(train_df=train_df, train_labels=train_labels, target_col=target_col)
        X_base = train_df.copy().reset_index(drop=True)
        y = y.reset_index(drop=True)

        mask_validi = y.notna()
        X_base = X_base.loc[mask_validi].reset_index(drop=True)
        y = y.loc[mask_validi].astype(int).reset_index(drop=True)

        if max_rows is not None and len(X_base) > int(max_rows):
            X_base, _, y, _ = train_test_split(
                X_base,
                y,
                train_size=int(max_rows),
                stratify=y,
                random_state=random_state
            )
            X_base = X_base.reset_index(drop=True)
            y = y.reset_index(drop=True)

        risultati = []

        for strategia in strategie:
            if strategia == 'univariata_media':
                X_imputato, _, report_imp = self.imputa_univariata_media(
                    train_df=X_base,
                    test_df=X_base,
                    colonna=colonna
                )
            elif strategia == 'univariata_mediana':
                X_imputato, _, report_imp = self.imputa_univariata_mediana(
                    train_df=X_base,
                    test_df=X_base,
                    colonna=colonna
                )
            elif strategia == 'multivariata_regressione_lineare':
                X_imputato, _, report_imp = self.imputa_multivariata_regressione_lineare(
                    train_df=X_base,
                    test_df=X_base,
                    colonna=colonna
                )
            elif strategia == 'knn_predictor':
                X_imputato, _, report_imp = self.imputa_knn_predictor(
                    train_df=X_base,
                    test_df=X_base,
                    colonna=colonna,
                    n_neighbors=n_neighbors_valutazione
                )
            else:
                raise ValueError(f"Strategia non supportata: {strategia}")

            X_model = X_imputato.drop(columns=['building_id'], errors='ignore')
            X_model = pd.get_dummies(X_model, dummy_na=True)

            X_train, X_val, y_train, y_val = train_test_split(
                X_model,
                y,
                test_size=test_size,
                stratify=y,
                random_state=random_state
            )

            scaler = StandardScaler(with_mean=False)
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            n_neighbors_eff = max(1, min(int(n_neighbors_valutazione), len(X_train)))

            knn_clf = KNeighborsClassifier(
                n_neighbors=n_neighbors_eff,
                weights='distance',
                algorithm='brute',
                n_jobs=-1
            )
            knn_clf.fit(X_train_scaled, y_train)

            y_pred = knn_clf.predict(X_val_scaled)
            accuracy = float(accuracy_score(y_val, y_pred))

            risultati.append({
                'strategia': strategia,
                'accuracy': round(accuracy, 6),
                'n_righe_valutate': int(len(X_model)),
                'n_features_model': int(X_model.shape[1]),
                'n_missing_train_dopo': int(report_imp['n_missing_train_dopo'])
            })

        risultati_df = pd.DataFrame(risultati).sort_values('accuracy', ascending=False).reset_index(drop=True)

        return risultati_df



