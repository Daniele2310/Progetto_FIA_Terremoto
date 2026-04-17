"""
Script principale per il preprocessing dati - Terremoto Nepal 2015
"""

import io
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
from DataPreprocessing.puliziaASCII import PuliziaASCII, COLONNE_CATEGORICHE
from DataPreprocessing.missingValues import MissingValuesHandler
from DataPreprocessing.data_cleaning import DataQualityHandler
from DataPreprocessing.validation import DataValidator


def menu_strategia_imputazione_age():
    """
    Mostra un menu semplice per scegliere la strategia di imputazione di age.
    Default: valuta tutte le strategie con KNN veloce e usa la migliore.
    """
    print("\n" + "=" * 80)
    print("MENU IMPUTAZIONE AGE")
    print("=" * 80)
    print("1) Univariata - Media")
    print("2) Univariata - Mediana")
    print("3) Multivariata - Regressione lineare")
    print("4) KNN predictor")
    print("5) Valuta tutte con KNN veloce (accuracy) e scegli la migliore")

    try:
        scelta = input("Seleziona opzione [1-5] (default=5): ").strip()
    except EOFError:
        scelta = ""

    if scelta not in {"1", "2", "3", "4", "5"}:
        scelta = "5"

    return scelta


def applica_strategia_imputazione_age(missing_handler, train_values, test_values, scelta):
    """Applica la strategia selezionata e ritorna train/test imputati + report."""
    if scelta == "1":
        return missing_handler.imputa_univariata_media(
            train_df=train_values,
            test_df=test_values,
            colonna="age",
        )

    if scelta == "2":
        return missing_handler.imputa_univariata_mediana(
            train_df=train_values,
            test_df=test_values,
            colonna="age",
        )

    if scelta == "3":
        return missing_handler.imputa_multivariata_regressione_lineare(
            train_df=train_values,
            test_df=test_values,
            colonna="age",
        )

    if scelta == "4":
        return missing_handler.imputa_knn_predictor(
            train_df=train_values,
            test_df=test_values,
            colonna="age",
            n_neighbors=5,
        )

    raise ValueError(f"Scelta strategia non valida: {scelta}")


def esegui_silenzioso(funzione, *args, **kwargs):
    """Esegue una funzione sopprimendo le stampe su stdout."""
    with redirect_stdout(io.StringIO()):
        return funzione(*args, **kwargs)


def salva_dataset_preprocessati(train_values, train_labels, test_values, output_dir="DataPreprocessed"):
    """Crea la cartella di output e salva i dataset preprocessati in formato CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_train = output_path / "train_values_preprocessed.csv"
    file_test = output_path / "test_values_preprocessed.csv"
    file_train_con_label = output_path / "train_features_labels_preprocessed.csv"

    train_values.to_csv(file_train, index=False)
    test_values.to_csv(file_test, index=False)
    pd.merge(train_values, train_labels, on="building_id").to_csv(file_train_con_label, index=False)

    print("\n" + "=" * 80)
    print("SALVATAGGIO DATASET PREPROCESSATI")
    print("=" * 80)
    print(f"File salvati in: {output_path.resolve()}")
    print(f"- {file_train.name}")
    print(f"- {file_test.name}")
    print(f"- {file_train_con_label.name}")


def main():
    print("\n" + "=" * 80)
    print("PREPROCESSING TERREMOTO NEPAL 2015")
    print("=" * 80)

    # =======================
    # CARICAMENTO DATI
    # =======================
    pulizia = PuliziaASCII()
    train_values, train_labels, test_values = pulizia.processa(
        colonne_categoriche=COLONNE_CATEGORICHE
    )

    # =======================
    # DATA QUALITY TRAIN
    # =======================
    train_quality_handler = DataQualityHandler(train_values)
    train_quality_report = train_quality_handler.esegui_controlli(plot=False)

    # Aggiorno il dataframe con le modifiche effettuate nell'handler
    train_values = train_quality_handler.data

    # Recupero l'upper bound di age calcolato sul training set
    age_upper_bound_train = train_quality_report["outliers"].loc["age", "upper_bound"]

    print("\n" + "=" * 80)
    print("CREAZIONE NUOVA FEATURE NEL TRAINING SET")
    print("=" * 80)

    # Creo la feature booleana sul train usando la soglia del train
    train_values = train_quality_handler.aggiungi_feature_age_flag(
        upper_bound=age_upper_bound_train
    )


    print("\n" + "=" * 80)
    print("REPORT OUTLIER TRAINING SET")
    print("=" * 80)
    print(train_quality_report["outliers"])

    # =======================
    # DATA QUALITY TEST
    # =======================
    test_quality_handler = DataQualityHandler(test_values)
    test_quality_report = test_quality_handler.esegui_controlli(plot=False)

    # Aggiorno il dataframe con le modifiche effettuate nell'handler
    test_values = test_quality_handler.data

    print("\n" + "=" * 80)
    print("CREAZIONE NUOVA FEATURE NEL TEST SET")
    print("=" * 80)

    # Creo la feature booleana sul test usando LA STESSA soglia del train
    test_values = test_quality_handler.aggiungi_feature_age_flag(
        upper_bound=age_upper_bound_train
    )


    print("\n" + "=" * 80)
    print("REPORT OUTLIER TEST SET")
    print("=" * 80)
    print(test_quality_report["outliers"])

    # =======================
    # IMPUTAZIONE AGE: MENU STRATEGIE
    # =======================
    missing_handler = MissingValuesHandler(null_threshold=70)
    train_values, _, n_sost_train = missing_handler.sostituisci_range_con_nan(
        train_values,
        colonna="age",
        min_val=250,
        max_val=995,
    )
    test_values, _, n_sost_test = missing_handler.sostituisci_range_con_nan(
        test_values,
        colonna="age",
        min_val=250,
        max_val=995,
    )

    scelta_imputazione = menu_strategia_imputazione_age()
    risultati_knn = None
    solo_log_imputazione = True

    if scelta_imputazione == "5":
        risultati_knn = missing_handler.valuta_strategie_con_knn_veloce(
            train_df=train_values,
            train_labels=train_labels,
            colonna="age",
            target_col="damage_grade",
            max_rows=20000,
            n_neighbors_valutazione=5,
        )

        print("\n" + "=" * 80)
        print("VALUTAZIONE STRATEGIE CON KNN VELOCE")
        print("=" * 80)
        print(risultati_knn.to_string(index=False))

        strategia_migliore = risultati_knn.iloc[0]["strategia"]
        print(f"\nStrategia migliore per accuracy KNN: {strategia_migliore}")

        mappa_scelta = {
            "univariata_media": "1",
            "univariata_mediana": "2",
            "multivariata_regressione_lineare": "3",
            "knn_predictor": "4",
        }
        scelta_imputazione = mappa_scelta[strategia_migliore]

    train_values, test_values, age_imputation_report = applica_strategia_imputazione_age(
        missing_handler=missing_handler,
        train_values=train_values,
        test_values=test_values,
        scelta=scelta_imputazione,
    )

    age_imputation_report["range_sostituito"] = [250, 995]
    age_imputation_report["n_valori_sostituiti_train"] = n_sost_train
    age_imputation_report["n_valori_sostituiti_test"] = n_sost_test
    if risultati_knn is not None:
        age_imputation_report["valutazione_knn_veloce"] = risultati_knn.to_dict(orient="records")

    print("\n" + "=" * 80)
    print(f"IMPUTAZIONE AGE - {age_imputation_report['strategia'].upper()}")
    print("=" * 80)
    print(f"Valori in [250, 995] sostituiti con NaN - train: {n_sost_train}, test: {n_sost_test}")
    print(
        f"Missing age train prima/dopo: "
        f"{age_imputation_report['n_missing_train_prima']} -> {age_imputation_report['n_missing_train_dopo']}"
    )
    print(
        f"Missing age test prima/dopo: "
        f"{age_imputation_report['n_missing_test_prima']} -> {age_imputation_report['n_missing_test_dopo']}"
    )

    # =======================
    # VALIDAZIONE TRAIN/TEST (PRIMA DEL ONE-HOT)
    # =======================
    validator_train = DataValidator(train_values)
    if solo_log_imputazione:
        validation_report_train = esegui_silenzioso(validator_train.esegui_validazione, verbose=True)
    else:
        validation_report_train = validator_train.esegui_validazione(verbose=True)

    validator_test = DataValidator(test_values)
    if solo_log_imputazione:
        validation_report_test = esegui_silenzioso(validator_test.esegui_validazione, verbose=True)
    else:
        validation_report_test = validator_test.esegui_validazione(verbose=True)

    # =======================
    # STANDARDIZZAZIONE TRAIN
    # =======================
    train_quality_handler.data = train_values
    if solo_log_imputazione:
        scaler = esegui_silenzioso(train_quality_handler.fit_standardizzazione)
        train_values = esegui_silenzioso(train_quality_handler.applica_standardizzazione, scaler)
    else:
        scaler = train_quality_handler.fit_standardizzazione()
        train_values = train_quality_handler.applica_standardizzazione(scaler)

        print("\n" + "=" * 80)
        print("VERIFICA STANDARDIZZAZIONE TRAIN")
        print("=" * 80)
        cols_to_check = ["count_floors_pre_eq", "age", "area_percentage", "height_percentage", "count_families"]
        print(train_values[cols_to_check].agg(["min", "max"]))

    # =======================
    # STANDARDIZZAZIONE TEST
    # =======================
    test_quality_handler.data = test_values
    if solo_log_imputazione:
        test_values = esegui_silenzioso(test_quality_handler.applica_standardizzazione, scaler)
    else:
        test_values = test_quality_handler.applica_standardizzazione(scaler)

        print("\n" + "=" * 80)
        print("VERIFICA STANDARDIZZAZIONE TEST")
        print("=" * 80)
        print(test_values[cols_to_check].agg(["min", "max"]))

    # =======================
    # ONE-HOT ENCODING
    # =======================
    if not solo_log_imputazione:
        print("\n" + "=" * 80)
        print("ONE-HOT ENCODING (TRAIN & TEST)")
        print("=" * 80)

    # 1. Fit ed applicazione sul TRAIN (Impara le regole e trasforma)
    train_quality_handler.data = train_values
    if solo_log_imputazione:
        ohe_encoder = esegui_silenzioso(train_quality_handler.fit_one_hot_encoding, COLONNE_CATEGORICHE)
        train_values = esegui_silenzioso(
            train_quality_handler.applica_one_hot_encoding,
            ohe_encoder,
            COLONNE_CATEGORICHE,
        )
    else:
        ohe_encoder = train_quality_handler.fit_one_hot_encoding(COLONNE_CATEGORICHE)
        train_values = train_quality_handler.applica_one_hot_encoding(ohe_encoder, COLONNE_CATEGORICHE)

    # 2. Applicazione sul TEST (Usa le regole imparate dal train)
    test_quality_handler.data = test_values
    if solo_log_imputazione:
        test_values = esegui_silenzioso(
            test_quality_handler.applica_one_hot_encoding,
            ohe_encoder,
            COLONNE_CATEGORICHE,
        )
    else:
        test_values = test_quality_handler.applica_one_hot_encoding(ohe_encoder, COLONNE_CATEGORICHE)

    # =======================
    # MERGE TRAIN + LABELS
    # =======================
    df_merged = pd.merge(train_values, train_labels, on="building_id")

    # =======================
    # MISSING VALUES
    # =======================
    handler = MissingValuesHandler(null_threshold=70)
    if solo_log_imputazione:
        report = esegui_silenzioso(handler.analizza, df_merged, target_col="damage_grade")
    else:
        report = handler.analizza(df_merged, target_col="damage_grade")
    report["age_imputation"] = age_imputation_report

    best_accuracy_knn = None
    if risultati_knn is not None and not risultati_knn.empty:
        best_accuracy_knn = float(risultati_knn.iloc[0]["accuracy"])

    print("\n" + "=" * 80)
    print("✅  PREPROCESSING COMPLETATO")
    print("=" * 80)
    print("FINITO PREPROCESSING: scelta dei metodi di imputazione con menu completata.")
    print(f"Metodo imputazione selezionato: {age_imputation_report['strategia']}")
    if best_accuracy_knn is not None:
        print(f"Migliore accuracy KNN veloce: {best_accuracy_knn:.6f}")

    salva_dataset_preprocessati(train_values, train_labels, test_values)

    return (
        train_values,
        train_labels,
        test_values,
        report,
        train_quality_report,
        test_quality_report,
        validation_report_train,
        validation_report_test,
    )


if __name__ == "__main__":
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
