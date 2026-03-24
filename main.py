"""
Script principale per il preprocessing dati - Terremoto Nepal 2015
"""

from DataPreprocessing.puliziaASCII import PuliziaASCII, COLONNE_CATEGORICHE


def main():
    """Esegue la conversione ASCII dei dati."""
    
    print("\n" + "TERREMOTO NEPAL 2015 - CONVERSIONE ASCII")
    
    # Crea l'istanza della classe di pulizia
    pulizia = PuliziaASCII()
    
    # Esegui la conversione ASCII
    train_values, train_labels, test_values = pulizia.processa(
        colonne_categoriche=COLONNE_CATEGORICHE
    )
    
    return train_values, train_labels, test_values


if __name__ == '__main__':
    train_values, train_labels, test_values = main()
