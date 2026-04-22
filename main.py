"""
Script principale per il preprocessing dati - Terremoto Nepal 2015
"""

import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
from DataPreprocessing.puliziaASCII import PuliziaASCII, COLONNE_CATEGORICHE
from DataPreprocessing.missingValues import MissingValuesHandler
from DataPreprocessing.data_cleaning import DataQualityHandler, COLONNE_CONTINUE
from DataPreprocessing.validation import DataValidator


SHOW_DATA_QUALITY_PLOTS = False


def carica_pca_handler():
    """Carica PCAHandler dal modulo in Feature Selection/feature ranking/PCA.py."""
    module_path = (
        Path(__file__).resolve().parent
        / "Feature Selection"
        / "Feature ranking"
        / "PCA.py"
    )

    if not module_path.exists():
        raise FileNotFoundError(f"Modulo PCA non trovato: {module_path}")

    spec = importlib.util.spec_from_file_location("feature_selection_feature_ranking_pca", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossibile caricare il modulo PCA da: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "PCAHandler"):
        raise ImportError("Classe 'PCAHandler' non trovata nel modulo PCA.")

    return module.PCAHandler


PCAHandler = carica_pca_handler()

def menu_strategia_imputazione_outlier_numerici():
    """
    Mostra un menu semplice per scegliere la strategia di imputazione
    delle feature numeriche con outlier sostituiti a NaN.
    Default: valuta tutte le strategie con KNN veloce e usa la migliore.
    """
    print("\n" + "=" * 80)
    print("MENU IMPUTAZIONE FEATURE NUMERICHE OUTLIER")
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


def menu_modalita_pca():
    """
    Permette di scegliere come determinare il numero di componenti PCA.
    Default: scelta manuale tramite scree plot e metodo del gomito.
    """
    print("\n" + "=" * 80)
    print("MENU PCA")
    print("=" * 80)
    print("1) Scelta automatica tramite soglia di varianza cumulativa")
    print("2) Scelta manuale tramite scree plot e metodo del gomito")

    try:
        scelta = input("Seleziona opzione [1-2] (default=2): ").strip()
    except EOFError:
        scelta = ""

    if scelta not in {"1", "2"}:
        scelta = "2"

    return scelta


def leggi_threshold_pca(default=0.95):
    """Legge da input la soglia di varianza cumulativa per la PCA."""
    try:
        valore = input(
            f"Inserisci la soglia di varianza cumulativa desiderata (default={default}): "
        ).strip()
    except EOFError:
        valore = ""

    if not valore:
        return default

    try:
        threshold = float(valore)
    except ValueError as exc:
        raise ValueError("La soglia PCA deve essere un numero compreso tra 0 e 1.") from exc

    if not (0 < threshold <= 1):
        raise ValueError("La soglia PCA deve stare nell'intervallo (0, 1].")

    return threshold


def leggi_n_componenti_pca(max_componenti):
    """Legge da input il numero di componenti PCA da mantenere."""
    try:
        valore = input(
            f"Inserisci il numero di componenti da mantenere [1-{max_componenti}]: "
        ).strip()
    except EOFError:
        valore = ""

    if not valore:
        raise ValueError("Devi inserire il numero di componenti PCA da mantenere.")

    try:
        n_componenti = int(valore)
    except ValueError as exc:
        raise ValueError("Il numero di componenti PCA deve essere un intero.") from exc

    if not (1 <= n_componenti <= max_componenti):
        raise ValueError(f"Il numero di componenti deve stare tra 1 e {max_componenti}.")

    return n_componenti


def applica_strategia_imputazione_colonna(missing_handler, train_values, test_values, scelta, colonna):
    """Applica la strategia selezionata su una colonna e ritorna train/test imputati + report."""
    if scelta == "1":
        return missing_handler.imputa_univariata_media(
            train_df=train_values,
            test_df=test_values,
            colonna=colonna,
        )

    if scelta == "2":
        return missing_handler.imputa_univariata_mediana(
            train_df=train_values,
            test_df=test_values,
            colonna=colonna,
        )

    if scelta == "3":
        return missing_handler.imputa_multivariata_regressione_lineare(
            train_df=train_values,
            test_df=test_values,
            colonna=colonna,
        )

    if scelta == "4":
        return missing_handler.imputa_knn_predictor(
            train_df=train_values,
            test_df=test_values,
            colonna=colonna,
            n_neighbors=5,
        )

    raise ValueError(f"Scelta strategia non valida: {scelta}")


def sostituisci_fuori_bound_con_nan(df, colonna, lower_bound, upper_bound):
    """Sostituisce con NaN i valori fuori dai bound [lower_bound, upper_bound]."""
    if colonna not in df.columns:
        return df, 0

    df_out = df.copy()
    mask_outlier = (df_out[colonna] < lower_bound) | (df_out[colonna] > upper_bound)
    n_sostituiti = int(mask_outlier.sum())
    df_out.loc[mask_outlier, colonna] = pd.NA

    return df_out, n_sostituiti


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
    train_quality_report = train_quality_handler.esegui_controlli(plot=SHOW_DATA_QUALITY_PLOTS)

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
    test_quality_report = test_quality_handler.esegui_controlli(plot=SHOW_DATA_QUALITY_PLOTS)

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
    # IMPUTAZIONE FEATURE NUMERICHE CON OUTLIER
    # =======================
    missing_handler = MissingValuesHandler(null_threshold=70)
    outliers_df = train_quality_report.get("outliers")
    colonne_numeriche = [col for col in COLONNE_CONTINUE if col in train_values.columns and col in test_values.columns]
    colonne_outlier = []

    if outliers_df is not None:
        for col in colonne_numeriche:
            if col in outliers_df.index and float(outliers_df.loc[col, "n_outliers"]) > 0:
                colonne_outlier.append(col)

    if not colonne_outlier:
        print("\nNessuna feature numerica con outlier rilevati: nessuna imputazione outlier necessaria.")

    outlier_replacement_counts = {}
    for col in colonne_outlier:
        lower_bound = float(outliers_df.loc[col, "lower_bound"])
        upper_bound = float(outliers_df.loc[col, "upper_bound"])

        train_values, n_sost_train = sostituisci_fuori_bound_con_nan(
            train_values,
            colonna=col,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        test_values, n_sost_test = sostituisci_fuori_bound_con_nan(
            test_values,
            colonna=col,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        outlier_replacement_counts[col] = {
            "lower_bound_train": lower_bound,
            "upper_bound_train": upper_bound,
            "n_valori_sostituiti_train": n_sost_train,
            "n_valori_sostituiti_test": n_sost_test,
        }

    scelta_imputazione = menu_strategia_imputazione_outlier_numerici()
    risultati_knn = None
    risultati_knn_per_colonna = {}
    solo_log_imputazione = True

    if scelta_imputazione == "5" and colonne_outlier:
        frames_risultati = []
        for col in colonne_outlier:
            risultati_col = missing_handler.valuta_strategie_con_knn_veloce(
                train_df=train_values,
                train_labels=train_labels,
                colonna=col,
                target_col="damage_grade",
                max_rows=20000,
                n_neighbors_valutazione=5,
            )
            risultati_knn_per_colonna[col] = risultati_col

            risultati_col_con_colonna = risultati_col.copy()
            risultati_col_con_colonna["colonna"] = col
            frames_risultati.append(risultati_col_con_colonna)

            print("\n" + "=" * 80)
            print(f"VALUTAZIONE STRATEGIE CON KNN VELOCE - COLONNA: {col}")
            print("=" * 80)
            print(risultati_col.to_string(index=False))

        risultati_knn = pd.concat(frames_risultati, ignore_index=True)
        riepilogo_knn = (
            risultati_knn
            .groupby("strategia", as_index=False)
            .agg(
                accuracy_media=("accuracy", "mean"),
                accuracy_min=("accuracy", "min"),
                accuracy_max=("accuracy", "max"),
                n_colonne=("colonna", "nunique"),
            )
            .sort_values("accuracy_media", ascending=False)
            .reset_index(drop=True)
        )

        print("\n" + "=" * 80)
        print("RIEPILOGO STRATEGIE KNN (MEDIA SU TUTTE LE COLONNE OUTLIER)")
        print("=" * 80)
        print(riepilogo_knn.to_string(index=False))

        strategia_migliore = riepilogo_knn.iloc[0]["strategia"]
        print(f"\nStrategia migliore media per accuracy KNN: {strategia_migliore}")

        mappa_scelta = {
            "univariata_media": "1",
            "univariata_mediana": "2",
            "multivariata_regressione_lineare": "3",
            "knn_predictor": "4",
        }
        scelta_imputazione = mappa_scelta[strategia_migliore]

    imputation_reports = {}
    for col in colonne_outlier:
        train_values, test_values, col_report = applica_strategia_imputazione_colonna(
            missing_handler=missing_handler,
            train_values=train_values,
            test_values=test_values,
            scelta=scelta_imputazione,
            colonna=col,
        )

        col_report.update(outlier_replacement_counts[col])
        if col in risultati_knn_per_colonna:
            col_report["valutazione_knn_veloce"] = risultati_knn_per_colonna[col].to_dict(orient="records")

        imputation_reports[col] = col_report

        print("\n" + "=" * 80)
        print(f"IMPUTAZIONE COLONNA '{col}' - {col_report['strategia'].upper()}")
        print("=" * 80)
        print(
            f"Valori fuori bound train [{col_report['lower_bound_train']:.2f}, {col_report['upper_bound_train']:.2f}] "
            f"sostituiti con NaN - train: {col_report['n_valori_sostituiti_train']}, "
            f"test: {col_report['n_valori_sostituiti_test']}"
        )
        print(
            f"Missing {col} train prima/dopo: "
            f"{col_report['n_missing_train_prima']} -> {col_report['n_missing_train_dopo']}"
        )
        print(
            f"Missing {col} test prima/dopo: "
            f"{col_report['n_missing_test_prima']} -> {col_report['n_missing_test_dopo']}"
        )

        if col_report["strategia"] in {"univariata_media", "univariata_mediana"}:
            nome_valore = "Media" if col_report["strategia"] == "univariata_media" else "Mediana"
            print(
                f"{nome_valore} usata per imputare '{col_report['colonna']}': "
                f"{col_report['valore_imputazione_train']:.2f}"
            )

        if col_report["strategia"] == "multivariata_regressione_lineare":
            def fmt_val(v):
                return f"{v:.6f}" if v is not None else "n/a"

            print(
                "Valori imputati TRAIN (regressione) - "
                f"media: {fmt_val(col_report.get('train_media_imputata'))}, "
                f"mediana: {fmt_val(col_report.get('train_mediana_imputata'))}, "
                f"min: {fmt_val(col_report.get('train_min_imputato'))}, "
                f"max: {fmt_val(col_report.get('train_max_imputato'))}"
            )
            print(
                "Valori imputati TEST (regressione) - "
                f"media: {fmt_val(col_report.get('test_media_imputata'))}, "
                f"mediana: {fmt_val(col_report.get('test_mediana_imputata'))}, "
                f"min: {fmt_val(col_report.get('test_min_imputato'))}, "
                f"max: {fmt_val(col_report.get('test_max_imputato'))}"
            )

        if col_report["strategia"] == "knn_predictor":
            def fmt_val(v):
                return f"{v:.2f}" if v is not None else "n/a"

            print(
                "Strategia usata: KNN predictor - "
                f"n_neighbors: {col_report.get('n_neighbors')}, "
                f"n_feature_usate: {col_report.get('n_feature_usate')}"
            )
            print("Preprocessing interno: mediana solo sui predittori numerici.")
            print(
                "Valori imputati TRAIN (KNN) - valore finale predetto dai vicini: "
                f"media: {fmt_val(col_report.get('train_media_imputata'))}, "
                f"mediana: {fmt_val(col_report.get('train_mediana_imputata'))}, "
                f"min: {fmt_val(col_report.get('train_min_imputato'))}, "
                f"max: {fmt_val(col_report.get('train_max_imputato'))}"
            )
            print(
                "Valori imputati TEST (KNN) - valore finale predetto dai vicini: "
                f"media: {fmt_val(col_report.get('test_media_imputata'))}, "
                f"mediana: {fmt_val(col_report.get('test_mediana_imputata'))}, "
                f"min: {fmt_val(col_report.get('test_min_imputato'))}, "
                f"max: {fmt_val(col_report.get('test_max_imputato'))}"
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
    # PCA
    # =======================
    if train_values.isnull().sum().sum() > 0 or test_values.isnull().sum().sum() > 0:
        raise ValueError("Sono presenti valori NaN: completa imputazione/pulizia prima di applicare PCA.")

    modalita_pca = menu_modalita_pca()
    pca_esplorativa = PCAHandler()
    pca_esplorativa.fit(
        df=train_values,
        exclude_columns=["building_id"],
    )

    explained_full = pca_esplorativa.explained_variance_ratio()
    cumulative_full = pca_esplorativa.cumulative_explained_variance()
    pca_summary = pd.DataFrame(
        {
            "explained_variance_ratio": explained_full.round(6),
            "cumulative_explained_variance": cumulative_full.round(6),
        }
    )

    output_dir = Path("DataPreprocessed")
    output_dir.mkdir(parents=True, exist_ok=True)
    variance_output_path = output_dir / "pca_variance_summary.csv"
    loadings_output_path = output_dir / "pca_loadings.csv"
    scree_plot_output_path = output_dir / "scree_plot.png"

    pca_summary.to_csv(variance_output_path)

    default_threshold_pca = 0.95

    print("\n" + "=" * 80)
    print("TABELLA VARIANZA SPIEGATA PCA")
    print("=" * 80)
    print(pca_summary.to_string())

    threshold_pca = None
    if modalita_pca == "1":
        threshold_pca = leggi_threshold_pca(default=default_threshold_pca)
        pca_esplorativa.plot_scree(
            output_path=scree_plot_output_path,
            threshold=threshold_pca,
            show_plot=True,
        )
        n_componenti = pca_esplorativa.choose_n_components(threshold=threshold_pca)
        metodo_selezione_pca = "threshold"
    else:
        pca_esplorativa.plot_scree(
            output_path=scree_plot_output_path,
            show_plot=True,
        )
        print(
            "\nOsserva lo scree plot e scegli il numero di componenti nel punto di gomito."
        )
        n_componenti = leggi_n_componenti_pca(
            max_componenti=len(pca_summary),
        )
        metodo_selezione_pca = "gomito"

    pca_esplorativa.plot_scree(
        output_path=scree_plot_output_path,
        threshold=threshold_pca,
        selected_n=n_componenti,
        show_plot=False,
    )

    pca_handler = PCAHandler(n_components=n_componenti)
    pca_handler.fit(
        df=train_values,
        exclude_columns=["building_id"],
    )

    train_values = pca_handler.transform(
        train_values,
        preserve_columns=["building_id"],
    )

    test_values = pca_handler.transform(
        test_values,
        preserve_columns=["building_id"],
    )

    pca_handler.get_loadings().to_csv(loadings_output_path)
    pca_report = pca_handler.build_report(threshold=threshold_pca)
    pca_report["selection_method"] = metodo_selezione_pca
    pca_report["n_components_selected"] = int(n_componenti)
    pca_report["variance_table_output_path"] = str(variance_output_path)
    pca_report["loadings_output_path"] = str(loadings_output_path)
    pca_report["scree_plot_output_path"] = str(scree_plot_output_path)

    print("\n" + "=" * 80)
    print("PCA COMPLETATA")
    print("=" * 80)
    print(f"Metodo selezione componenti: {metodo_selezione_pca}")
    print(f"Numero componenti selezionate: {n_componenti}")
    print(
        f"Varianza cumulativa spiegata: "
        f"{list(pca_report['cumulative_explained_variance'].values())[-1]:.4f}"
    )
    print(f"Nuove dimensioni train: {train_values.shape}")
    print(f"Nuove dimensioni test: {test_values.shape}")
    print(f"Tabella varianza salvata in: {variance_output_path}")
    print(f"Loadings salvati in: {loadings_output_path}")
    print(f"Scree plot salvato in: {scree_plot_output_path}")

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
    report["numeric_outlier_imputation"] = imputation_reports
    if "age" in imputation_reports:
        report["age_imputation"] = imputation_reports["age"]
    report["pca"] = pca_report

    best_accuracy_knn = None
    if risultati_knn is not None and not risultati_knn.empty:
        best_accuracy_knn = float(risultati_knn.groupby("strategia")["accuracy"].mean().max())

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETATO")
    print("=" * 80)
    print("FINITO PREPROCESSING: scelta dei metodi di imputazione con menu completata.")
    if imputation_reports:
        strategia_usata = next(iter(imputation_reports.values()))["strategia"]
        print(f"Metodo imputazione selezionato: {strategia_usata}")
        print(f"Colonne numeriche imputate per outlier: {list(imputation_reports.keys())}")
    else:
        print("Nessuna colonna numerica da imputare per outlier.")
    if best_accuracy_knn is not None:
        print(f"Migliore accuracy media KNN veloce: {best_accuracy_knn:.6f}")

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
