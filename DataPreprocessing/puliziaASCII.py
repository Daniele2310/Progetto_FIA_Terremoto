"""
Modulo PuliziaASCII - Conversione caratteri ASCII per il dataset terremoto Nepal
"""

import pandas as pd
import numpy as np
import unicodedata
from pathlib import Path


COLONNE_CATEGORICHE = [
    'land_surface_condition',
    'foundation_type',
    'roof_type',
    'ground_floor_type',
    'other_floor_type',
    'position',
    'plan_configuration',
    'legal_ownership_status'
]

CARTELLA_INPUT = 'Data'


class PuliziaASCII:
    """Classe per la conversione dei caratteri ASCII nel dataset."""
    
    def __init__(self, cartella_input=CARTELLA_INPUT):
        self.cartella_input = Path(cartella_input)
        self.dati_originali = {}
        self.dati_puliti = {}
    
    def carica_dati(self):
        """Carica i dati di training e test dal disco."""
        train_values = pd.read_csv(self.cartella_input / 'train_values.csv')
        train_labels = pd.read_csv(self.cartella_input / 'train_labels.csv')
        test_values = pd.read_csv(self.cartella_input / 'test_values.csv')
        
        print(f"Train: {train_values.shape[0]} x {train_values.shape[1]}")
        print(f"Labels: {train_labels.shape[0]}")
        print(f"Test: {test_values.shape[0]} x {test_values.shape[1]}")
        
        self.dati_originali = {
            'train_values': train_values.copy(),
            'train_labels': train_labels.copy(),
            'test_values': test_values.copy()
        }
        
        return train_values, train_labels, test_values
    
    def rimuovi_caratteri_unicode(self, valore):
        """Rimuove caratteri non-ASCII e normalizza il valore."""
        if pd.isna(valore):
            return valore
        
        valore_str = str(valore).strip()
        normalizzato = unicodedata.normalize('NFD', valore_str)
        solo_ascii = normalizzato.encode('ascii', 'ignore').decode('ascii')
        
        return solo_ascii if solo_ascii else np.nan
    
    def pulisci_colonne_categoriche(self, df, colonne_categoriche):
        """Pulisce le colonne categoriche."""
        df_pulito = df.copy()
        
        for col in colonne_categoriche:
            if col in df_pulito.columns:
                df_pulito[col] = df_pulito[col].apply(self.rimuovi_caratteri_unicode)
                df_pulito[col] = df_pulito[col].str.lower().str.strip()
                df_pulito.loc[df_pulito[col] == '', col] = np.nan
        
        return df_pulito
    
    def processa(self, colonne_categoriche):
        """Esegue la conversione ASCII dei dati."""
        train_values, train_labels, test_values = self.carica_dati()
        
        train_values_pulito = self.pulisci_colonne_categoriche(train_values, colonne_categoriche)
        test_values_pulito = self.pulisci_colonne_categoriche(test_values, colonne_categoriche)
        
        self.dati_puliti = {
            'train_values': train_values_pulito,
            'train_labels': train_labels,
            'test_values': test_values_pulito
        }
        
        return train_values_pulito, train_labels, test_values_pulito
