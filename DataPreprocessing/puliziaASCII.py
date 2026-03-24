"""
Modulo PuliziaASCII - Conversione caratteri ASCII per il dataset terremoto Nepal
"""

import pandas as pd
import numpy as np
import unicodedata
from pathlib import Path


# ============================================================================
# CONFIGURAZIONE
# ============================================================================

COLONNE_CATEGORICHE = [
    'land_surface_condition',      # Condizione superficie terreno
    'foundation_type',              # Tipo fondamenta
    'roof_type',                    # Tipo tetto
    'ground_floor_type',            # Tipo pavimento piano terra
    'other_floor_type',             # Tipo costruzione piani superiori
    'position',                     # Posizione edificio
    'plan_configuration',           # Configurazione pianta
    'legal_ownership_status'        # Status proprietà legale
]

CARTELLA_INPUT = 'Data'


# ============================================================================
# CLASSE PRINCIPALE
# ============================================================================

class PuliziaASCII:
    """
    Classe per la conversione dei caratteri ASCII nel dataset
    del terremoto del 2015 in Nepal.
    
    Funzionalità:
    - Rimozione di caratteri non-ASCII
    - Normalizzazione Unicode
    - Conversione a lowercase
    """
    
    def __init__(self, cartella_input=CARTELLA_INPUT):
        """
        Inizializza la classe.
        
        Args:
            cartella_input (str): Percorso cartella con dati grezzi
        """
        self.cartella_input = Path(cartella_input)
        self.dati_originali = {}
        self.dati_puliti = {}
    
    def carica_dati(self):
        """
        Carica i dati di training e test dal disco.
        
        Returns:
            tuple: (train_values, train_labels, test_values)
        """
        print("\n📂 Caricamento dati...")
        
        try:
            train_values = pd.read_csv(self.cartella_input / 'train_values.csv')
            train_labels = pd.read_csv(self.cartella_input / 'train_labels.csv')
            test_values = pd.read_csv(self.cartella_input / 'test_values.csv')
            
            print(f"   ✓ train_values caricato: {train_values.shape[0]} righe, {train_values.shape[1]} colonne")
            print(f"   ✓ train_labels caricato: {train_labels.shape[0]} righe")
            print(f"   ✓ test_values caricato: {test_values.shape[0]} righe, {test_values.shape[1]} colonne")
            
            self.dati_originali = {
                'train_values': train_values.copy(),
                'train_labels': train_labels.copy(),
                'test_values': test_values.copy()
            }
            
            return train_values, train_labels, test_values
            
        except FileNotFoundError as e:
            print(f"   ✗ Errore: file non trovato - {e}")
            raise
    
    def rimuovi_caratteri_unicode(self, valore):
        """
        Rimuove caratteri non-ASCII e normalizza il valore.
        
        Args:
            valore: Valore da pulire
            
        Returns:
            str o NaN: Valore pulito oppure NaN se vuoto
        """
        if pd.isna(valore):
            return valore
        
        valore_str = str(valore).strip()
        
        # Normalizza Unicode (decomposizione NFD: separa accenti da lettere)
        normalizzato = unicodedata.normalize('NFD', valore_str)
        
        # Mantiene solo ASCII (non mantiene accenti e caratteri speciali)
        solo_ascii = normalizzato.encode('ascii', 'ignore').decode('ascii')
        
        # Ritorna il valore pulito oppure NaN se vuoto
        return solo_ascii if solo_ascii else np.nan
    
    def pulisci_colonne_categoriche(self, df, colonne_categoriche):
        """
        Pulisce le colonne categoriche.
        
        Args:
            df (pd.DataFrame): DataFrame da pulire
            colonne_categoriche (list): Lista colonne categoriche da pulire
            
        Returns:
            pd.DataFrame: DataFrame con colonne pulite
        """
        df_pulito = df.copy()
        
        for col in colonne_categoriche:
            if col in df_pulito.columns:
                # Rimuovi caratteri non-ASCII
                df_pulito[col] = df_pulito[col].apply(self.rimuovi_caratteri_unicode)
                
                # Converti in lowercase e rimuovi spazi
                df_pulito[col] = df_pulito[col].str.lower().str.strip()
                
                # Sostituisci stringhe vuote con NaN
                df_pulito.loc[df_pulito[col] == '', col] = np.nan
        
        return df_pulito
    
    def processa(self, colonne_categoriche):
        """
        Esegue la conversione ASCII dei dati.
        
        Args:
            colonne_categoriche (list): Lista colonne categoriche da pulire
            
        Returns:
            tuple: (train_values_pulito, train_labels, test_values_pulito)
        """
        print("\n" + "="*70)
        print("PULIZIA CONVERSIONE CARATTERI ASCII")
        print("="*70)
        
        # Carica dati
        train_values, train_labels, test_values = self.carica_dati()
        
        # Pulisci colonne categoriche
        print("\nPulizia colonne categoriche da non-ASCII...")
        train_values_pulito = self.pulisci_colonne_categoriche(train_values, colonne_categoriche)
        test_values_pulito = self.pulisci_colonne_categoriche(test_values, colonne_categoriche)
        print("   Colonne categoriche pulite")
        
        print("\n" + "="*70)
        print("PULIZIA COMPLETATA!")
        print("="*70 + "\n")
        
        # Salva i dati puliti
        self.dati_puliti = {
            'train_values': train_values_pulito,
            'train_labels': train_labels,
            'test_values': test_values_pulito
        }
        
        return train_values_pulito, train_labels, test_values_pulito
