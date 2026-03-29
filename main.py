"""
Script principale per il preprocessing dati - Terremoto Nepal 2015
"""

import pandas as pd
from DataPreprocessing.puliziaASCII import PuliziaASCII, COLONNE_CATEGORICHE
from DataPreprocessing.missingValues import MissingValuesHandler
from DataPreprocessing.data_cleaning import DataQualityHandler
from DataPreprocessing.validation import DataValidator


def main():
    """Esegue il preprocessing completo."""
    
    print("Preprocessing Terremoto Nepal 2015")
    
    pulizia = PuliziaASCII()
    train_values, train_labels, test_values = pulizia.processa(
        colonne_categoriche=COLONNE_CATEGORICHE
    )

    # analisi training set
    train_quality_handler = DataQualityHandler(train_values)
    train_quality_report = train_quality_handler.esegui_controlli(plot=True)
    
    print("\n" + "="*80)
    print("REPORT OUTLIER TRAINING SET")
    print("="*80)
    print(train_quality_report["outliers"])

    # analisi del test set
    test_quality_handler = DataQualityHandler(test_values)
    test_quality_report = test_quality_handler.esegui_controlli(plot=True)
    
    print("\n" + "="*80)
    print("REPORT OUTLIER TEST SET")
    print("="*80)
    print(test_quality_report["outliers"])

    # validazione feature booleane e categoriche (training set)
    validator_train = DataValidator(train_values)
    validation_report_train = validator_train.esegui_validazione(verbose=True)

    # validazione feature booleane e categoriche (test set)
    validator_test = DataValidator(test_values)
    validation_report_test = validator_test.esegui_validazione(verbose=True)

    df_merged = pd.merge(train_values, train_labels, on='building_id')
    
    handler = MissingValuesHandler(null_threshold=70)
    report = handler.analizza(df_merged, target_col='damage_grade')

    
    return train_values, train_labels, test_values, report, train_quality_report, test_quality_report, validation_report_train, validation_report_test


if __name__ == '__main__':
    train_values, train_labels, test_values, report, train_quality_report, test_quality_report, validation_report_train, validation_report_test = main()
