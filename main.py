"""
Script principale per il preprocessing dati - Terremoto Nepal 2015
"""

import pandas as pd
from DataPreprocessing.puliziaASCII import PuliziaASCII, COLONNE_CATEGORICHE
from DataPreprocessing.missingValues import MissingValuesHandler
from DataPreprocessing.data_cleaning import DataQualityHandler
from DataPreprocessing.validation import DataValidator


def main():
    print("Preprocessing Terremoto Nepal 2015")

    # Caricamento e pulizia ASCII
    pulizia = PuliziaASCII()
    train_values, train_labels, test_values = pulizia.processa(
        colonne_categoriche=COLONNE_CATEGORICHE
    )

    # =======================
    # DATA QUALITY TRAIN
    # =======================
    train_quality_handler = DataQualityHandler(train_values)
    train_quality_report = train_quality_handler.esegui_controlli(plot=False)

    # Aggiorno il dataframe con le modifiche effettuate nel handler
    train_values = train_quality_handler.data

    # =======================
    # NORMALIZZAZIONE TRAIN
    # =======================
    scaler = train_quality_handler.fit_normalizzazione()
    train_values = train_quality_handler.applica_normalizzazione(scaler)

    print("\n" + "=" * 80)
    print("REPORT OUTLIER TRAINING SET")
    print("=" * 80)
    print(train_quality_report["outliers"])

    # =======================
    # DATA QUALITY TEST
    # =======================
    test_quality_handler = DataQualityHandler(test_values)
    test_quality_report = test_quality_handler.esegui_controlli(plot=False)

    # Aggiorno il dataframe con le modifiche effettuate nel handler
    test_values = test_quality_handler.data

    # =======================
    # NORMALIZZAZIONE TEST
    # =======================
    test_values = test_quality_handler.applica_normalizzazione(scaler)

    print("\n" + "=" * 80)
    print("REPORT OUTLIER TEST SET")
    print("=" * 80)
    print(test_quality_report["outliers"])

    # =======================
    # VALIDAZIONE
    # =======================
    validator_train = DataValidator(train_values)
    validation_report_train = validator_train.esegui_validazione(verbose=True)

    validator_test = DataValidator(test_values)
    validation_report_test = validator_test.esegui_validazione(verbose=True)

    # =======================
    # MERGE TRAIN + LABELS
    # =======================
    df_merged = pd.merge(train_values, train_labels, on='building_id')

    # =======================
    # MISSING VALUES
    # =======================
    handler = MissingValuesHandler(null_threshold=70)
    report = handler.analizza(df_merged, target_col='damage_grade')

    print("\nPreprocessing completato.")

    return (
        train_values,
        train_labels,
        test_values,
        report,
        train_quality_report,
        test_quality_report,
        validation_report_train,
        validation_report_test
    )


if __name__ == '__main__':
    (
        train_values,
        train_labels,
        test_values,
        report,
        train_quality_report,
        test_quality_report,
        validation_report_train,
        validation_report_test
    ) = main()