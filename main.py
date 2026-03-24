"""
Script principale per il preprocessing dati - Terremoto Nepal 2015
"""

import pandas as pd
from DataPreprocessing.puliziaASCII import PuliziaASCII, COLONNE_CATEGORICHE
from DataPreprocessing.missingValues import MissingValuesHandler


def main():
    """Esegue il preprocessing completo."""
    
    print("Preprocessing Terremoto Nepal 2015")
    
    pulizia = PuliziaASCII()
    train_values, train_labels, test_values = pulizia.processa(
        colonne_categoriche=COLONNE_CATEGORICHE
    )
    
    df_merged = pd.merge(train_values, train_labels, on='building_id')
    
    handler = MissingValuesHandler(null_threshold=70)
    report = handler.analizza(df_merged, target_col='damage_grade')
    
    suggerimenti = handler.suggerisci_azioni()
    if suggerimenti:
        print("\nSuggerimenti:")
        for sug in suggerimenti:
            print(f"  - {sug}")
    
    return train_values, train_labels, test_values, report


if __name__ == '__main__':
    train_values, train_labels, test_values, report = main()
