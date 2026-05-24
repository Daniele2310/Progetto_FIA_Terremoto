"""
Script principale per la pipeline ML - Terremoto Nepal 2015

MENU PRINCIPALE:
  FASE 1  - Outlier Detection     : IQR oppure DBSCAN
  FASE 2  - Imputazione           : strategia sui NaN generati dagli outlier
                                    [nota: monum_flag aggiunta DOPO outlier+imputazione]
  FASE 3  - Geo-Level Embedding   : rete neurale per feature geografiche
  FASE 4  - Scelta modello        : KNN / Albero Decisionale / Random Forest
                                    [DA IMPLEMENTARE] Multi-Esperto (AdaBoost / SVM)
  FASE 5  - Presentazione risultati (F1-Micro)
             [DA IMPLEMENTARE] Confronto migliore FS per modello
"""

import io
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
from src.preprocessing.clean_ascii import PuliziaASCII, COLONNE_CATEGORICHE
from src.preprocessing.missing_values import MissingValuesHandler
from src.preprocessing.data_cleaning import DataQualityHandler, COLONNE_CONTINUE
from src.preprocessing.validation import DataValidator
from src.preprocessing.imputation_strategies import (
    STRATEGIE_IMPUTAZIONE,
    CODICE_STRATEGIA_DA_NOME_REPORT,
    applica_strategia_imputazione_colonna,
)
from src.preprocessing.outlier_detection.DBSCAN import rileva_outlier_dbscan

SHOW_DATA_QUALITY_PLOTS = False


# ============================================================
# BANNER
# ============================================================

def _banner(titolo: str) -> None:
    print("\n" + "=" * 80)
    print(titolo)
    print("=" * 80)


# ============================================================
# FASE 1 — OUTLIER DETECTION
# ============================================================

def menu_outlier_detection() -> str:
    """
    Fase 1: scelta del metodo di outlier detection.

    Ritorna:
        '1' -> IQR  (Interquartile Range)
        '2' -> DBSCAN (multivariato)
    """
    _banner("FASE 1 — OUTLIER DETECTION")
    print("  1) IQR    — Interquartile Range (per colonna, soglia classica)")
    print("  2) DBSCAN — outlier detection multivariato con hyperparameter tuning")
    print()
    try:
        scelta = input("Seleziona metodo [1-2] (default=1): ").strip()
    except EOFError:
        scelta = ""
    if scelta not in {"1", "2"}:
        scelta = "1"
    print(f"\n>> Metodo selezionato: {'IQR' if scelta == '1' else 'DBSCAN'}")
    return scelta


# ============================================================
# FASE 2 — IMPUTAZIONE
# ============================================================

def menu_strategia_imputazione_outlier_numerici() -> str:
    """
    Fase 2: scelta della strategia di imputazione per i NaN creati
    dalla rimozione degli outlier.

    Ritorna codice stringa '1'-'5'.
    """
    _banner("FASE 2 — IMPUTAZIONE FEATURE NUMERICHE (outlier → NaN)")
    for strategia in STRATEGIE_IMPUTAZIONE.values():
        print(f"  {strategia.codice_menu}) {strategia.nome_menu}")
    print("  5) Valuta TUTTE con KNN veloce e scegli automaticamente la migliore")
    print()
    try:
        scelta = input("Seleziona opzione [1-5] (default=5): ").strip()
    except EOFError:
        scelta = ""
    if scelta not in {"1", "2", "3", "4", "5"}:
        scelta = "5"
    return scelta


# ============================================================
# FASE 3 — GEO EMBEDDING
# ============================================================

def menu_geo_embedding() -> str:
    """
    Fase 3: scelta del numero di hidden layer per la rete neurale
    usata per il geo-level embedding / feature selection.

    Ritorna stringa con numero hidden layer ('1', '2', ...).
    Default = 2.
    """
    _banner("FASE 3 — GEO-LEVEL EMBEDDING (Rete Neurale)")
    print("  Viene addestrata una rete neurale sui geo_level_1/2/3 per")
    print("  generare rappresentazioni dense usabili come feature.")
    print()
    print("  Numero di hidden layer (Feature Selection interna):")
    print("  1) 1 hidden layer  — rete leggera")
    print("  2) 2 hidden layer  — default consigliato")
    print("  3) 3 hidden layer  — rete più espressiva")
    print()
    try:
        scelta = input("Seleziona numero hidden layer [1-3] (default=2): ").strip()
    except EOFError:
        scelta = ""
    if scelta not in {"1", "2", "3"}:
        scelta = "2"
    print(f"\n>> Geo embedding con {scelta} hidden layer selezionato.")
    return scelta


# ============================================================
# FASE 4 — SCELTA MODELLO  [parzialmente DA IMPLEMENTARE]
# ============================================================

def menu_scelta_modello() -> str:
    """
    Fase 4: scelta del modello di classificazione principale.

    Modelli disponibili:
        1 -> KNN
        2 -> Albero Decisionale
        3 -> Random Forest
        4 -> [DA IMPLEMENTARE] Multi-Esperto (AdaBoost / SVM)

    Ritorna stringa '1'-'4'.
    """
    _banner("FASE 4 — SCELTA MODELLO")
    print("  1) KNN               — K-Nearest Neighbors")
    print("  2) Albero Decisionale — Decision Tree")
    print("  3) Random Forest      — ensemble di alberi")
    print("  4) Multi-Esperto      — AdaBoost / SVM  [DA IMPLEMENTARE]")
    print()
    try:
        scelta = input("Seleziona modello [1-4] (default=1): ").strip()
    except EOFError:
        scelta = ""
    if scelta not in {"1", "2", "3", "4"}:
        scelta = "1"

    nomi = {
        "1": "KNN",
        "2": "Albero Decisionale",
        "3": "Random Forest",
        "4": "Multi-Esperto [DA IMPLEMENTARE]",
    }
    print(f"\n>> Modello selezionato: {nomi[scelta]}")

    if scelta == "4":
        print("\n" + "!" * 80)
        print("  SEZIONE MULTI-ESPERTO NON ANCORA IMPLEMENTATA.")
        print("  Seleziona un'altra opzione oppure procedi per testare il placeholder.")
        print("!" * 80)

    return scelta


# ============================================================
# UTILITY
# ============================================================

def sostituisci_outlier_con_nan(df, colonna, lower_bound, upper_bound, metodo="iqr", valori_anomali=None):
    """Sostituisce con NaN gli outlier rilevati con IQR o rarita se IQR=0."""
    if colonna not in df.columns:
        return df, 0
    df_out = df.copy()
    if metodo == "rarity_iqr_zero" and valori_anomali:
        mask_outlier = df_out[colonna].isin(valori_anomali)
    else:
        mask_outlier = (df_out[colonna] < lower_bound) | (df_out[colonna] > upper_bound)

    n_sostituiti = int(mask_outlier.sum())
    df_out.loc[mask_outlier, colonna] = pd.NA
    return df_out, n_sostituiti


def esegui_silenzioso(funzione, *args, **kwargs):
    """Esegue una funzione sopprimendo le stampe su stdout."""
    with redirect_stdout(io.StringIO()):
        return funzione(*args, **kwargs)


def salva_dataset_preprocessati(train_values, train_labels, test_values, output_dir="Data/preprocessed"):
    """Crea la cartella e salva i dataset preprocessati in CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_train = output_path / "train_values_preprocessed.csv"
    file_test = output_path / "test_values_preprocessed.csv"
    file_train_con_label = output_path / "train_features_labels_preprocessed.csv"

    train_values.to_csv(file_train, index=False)
    test_values.to_csv(file_test, index=False)
    pd.merge(train_values, train_labels, on="building_id").to_csv(file_train_con_label, index=False)

    _banner("SALVATAGGIO DATASET PREPROCESSATI")
    print(f"File salvati in: {output_path.resolve()}")
    print(f"  - {file_train.name}")
    print(f"  - {file_test.name}")
    print(f"  - {file_train_con_label.name}")


# ============================================================
# STUB MODELLI  [fasi 4-5, parte DA IMPLEMENTARE]
# ============================================================

def _stub_da_implementare(nome_sezione: str) -> None:
    _banner(f"[DA IMPLEMENTARE] — {nome_sezione}")
    print(f"  La sezione '{nome_sezione}' non è ancora implementata.")
    print("  Verrà integrata nelle prossime iterazioni dello sviluppo.")


def esegui_modello(scelta_modello: str, train_values, train_labels, test_values):
    """
    Dispatcher per l'addestramento del modello selezionato.

    Modelli 1-3 (KNN, Albero, RF): placeholder con messaggio informativo.
    Modello 4 (Multi-Esperto):      stub esplicito DA IMPLEMENTARE.
    """
    _banner("FASE 4 — ADDESTRAMENTO MODELLO")

    if scelta_modello == "4":
        _stub_da_implementare("Multi-Esperto (AdaBoost / SVM)")
        return None

    nomi = {
        "1": "KNN",
        "2": "Albero Decisionale",
        "3": "Random Forest",
    }
    nome = nomi.get(scelta_modello, "Sconosciuto")
    print(f"  Addestramento {nome} in corso...")
    print("  [placeholder — implementazione modello da completare]")
    # TODO: integrare Hyperparameter_Tuning.py per KNN / DT / RF
    return None


def presenta_risultati(scelta_modello: str, risultati_modello) -> None:
    """
    Fase 5: presentazione dei risultati (F1-Micro).

    La selezione del miglior modello per FS è DA IMPLEMENTARE.
    """
    _banner("FASE 5 — PRESENTAZIONE RISULTATI (F1-Micro)")

    if risultati_modello is None:
        print("  Nessun risultato disponibile (modello non addestrato o stub).")
        _stub_da_implementare("Confronto migliore FS per ogni modello (Fase 4 → 5)")
        return

    # TODO: calcolo F1-Micro reale dai risultati del modello
    print("  F1-Micro: [DA CALCOLARE dopo integrazione modelli]")
    _stub_da_implementare("Confronto migliore FS per ogni modello (Fase 4 → 5)")


# ============================================================
# MAIN
# ============================================================

def main():
    _banner("PIPELINE ML — TERREMOTO NEPAL 2015")
    print("  Fasi:")
    print("    1) Outlier Detection  (IQR / DBSCAN)")
    print("    2) Imputazione        (varie strategie)")
    print("    3) Geo Embedding      (rete neurale)")
    print("    4) Scelta Modello     (KNN / DT / RF / Multi-Esperto)")
    print("    5) Risultati          (F1-Micro)")

    # ── FASE 1: Outlier Detection ──────────────────────────
    scelta_outlier = menu_outlier_detection()

    # ── CARICAMENTO DATI ───────────────────────────────────
    _banner("CARICAMENTO DATI")
    pulizia = PuliziaASCII()
    train_values, train_labels, test_values = pulizia.processa(
        colonne_categoriche=COLONNE_CATEGORICHE
    )

    # ── DATA QUALITY TRAIN ─────────────────────────────────
    train_quality_handler = DataQualityHandler(train_values)
    train_quality_report = train_quality_handler.esegui_controlli(plot=SHOW_DATA_QUALITY_PLOTS)
    train_values = train_quality_handler.data

    # Salvo l'upper bound di 'age' calcolato sul train (serve per monum_flag)
    age_upper_bound_train = train_quality_report["outliers"].loc["age", "upper_bound"]

    _banner("REPORT OUTLIER — TRAINING SET")
    print(train_quality_report["outliers"])

    # ── DATA QUALITY TEST ──────────────────────────────────
    test_quality_handler = DataQualityHandler(test_values)
    test_quality_handler.pulisci_nomi_colonne()
    test_quality_handler.controlla_duplicati_building_id()
    test_quality_report = test_quality_handler.report
    test_values = test_quality_handler.data

    # ── SNAPSHOT età ORIGINALE ─────────────────────────────
    # Salvo la colonna 'age' PRIMA di qualsiasi modifica da outlier detection.
    # Serve per costruire monum_flag sul valore reale dell'edificio,
    # non sul valore imputato (che sarebbe sempre ≤ soglia dopo la pulizia).
    age_originale_train = train_values["age"].copy() if "age" in train_values.columns else None
    age_originale_test  = test_values["age"].copy()  if "age" in test_values.columns  else None

    # ── OUTLIER DETECTION E SOSTITUZIONE (solo TRAIN) ──────
    missing_handler = MissingValuesHandler(null_threshold=70)
    colonne_numeriche = [
        col for col in COLONNE_CONTINUE
        if col in train_values.columns and col in test_values.columns
    ]
    colonne_outlier = []
    outlier_replacement_counts = {}

    if scelta_outlier == "1":
        # --- IQR ---
        _banner("OUTLIER DETECTION — IQR (solo training set)")
        outliers_df = train_quality_report.get("outliers")
        if outliers_df is not None:
            for col in colonne_numeriche:
                if col in outliers_df.index and float(outliers_df.loc[col, "n_outliers"]) > 0:
                    colonne_outlier.append(col)

        for col in colonne_outlier:
            lower_bound = float(outliers_df.loc[col, "lower_bound"])
            upper_bound = float(outliers_df.loc[col, "upper_bound"])
            metodo_outlier = outliers_df.loc[col, "metodo"] if "metodo" in outliers_df.columns else "iqr"
            valori_anomali = outliers_df.loc[col, "valori_anomali"] if "valori_anomali" in outliers_df.columns else []
            if not isinstance(valori_anomali, list):
                valori_anomali = []

            train_values, n_sost_train = sostituisci_outlier_con_nan(
                train_values,
                colonna=col,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                metodo=metodo_outlier,
                valori_anomali=valori_anomali,
            )
            outlier_replacement_counts[col] = {
                "lower_bound_train": lower_bound,
                "upper_bound_train": upper_bound,
                "metodo_outlier": metodo_outlier,
                "valori_anomali_train": valori_anomali,
                "n_valori_sostituiti_train": n_sost_train,
            }
            print(
                f"  [{col}] metodo={metodo_outlier} "
                f"bound=[{lower_bound:.2f}, {upper_bound:.2f}]  "
                f"outlier->NaN: {n_sost_train}"
            )

    else:
        # --- DBSCAN ---
        _banner("OUTLIER DETECTION — DBSCAN (solo training set)")

        # (l'IQR è già stato calcolato per il report, qui usiamo solo DBSCAN)
        outliers_df = train_quality_report.get("outliers")

        # Passo 2: DBSCAN multivariato
        print("\n  Avvio DBSCAN per raffinamento multivariato...")
        mask_outlier, info_dbscan = rileva_outlier_dbscan(train_values, COLONNE_CONTINUE)
        n_outlier_totali = int(mask_outlier.sum())

        print(f"\n  DBSCAN — parametri usati: {info_dbscan}")
        print(f"  DBSCAN — righe outlier individuate: {n_outlier_totali}")

        if n_outlier_totali > 0:
            colonne_outlier = list(colonne_numeriche)
            for col in colonne_outlier:
                n_nan_prima = int(train_values[col].isna().sum())
                train_values.loc[mask_outlier, col] = pd.NA
                n_nan_dopo = int(train_values[col].isna().sum())
                n_sostituiti = n_nan_dopo - n_nan_prima
                outlier_replacement_counts[col] = {
                    "lower_bound_train": float("nan"),
                    "upper_bound_train": float("nan"),
                    "n_valori_sostituiti_train": n_sostituiti,
                }
                print(f"  [{col}]  outlier→NaN (DBSCAN): {n_sostituiti}")
        else:
            print("  DBSCAN: nessun outlier individuato.")

    if not colonne_outlier:
        print("\n  Nessuna feature numerica con outlier: nessuna imputazione necessaria.")

    # ── FASE 2: Imputazione ────────────────────────────────
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

            _banner(f"VALUTAZIONE STRATEGIE KNN — colonna: {col}")
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

        _banner("RIEPILOGO STRATEGIE KNN (media su tutte le colonne outlier)")
        print(riepilogo_knn.to_string(index=False))

        strategia_migliore = riepilogo_knn.iloc[0]["strategia"]
        print(f"\n  Strategia migliore: {strategia_migliore}")
        scelta_imputazione = CODICE_STRATEGIA_DA_NOME_REPORT[strategia_migliore]

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

        _banner(f"IMPUTAZIONE '{col}' — {col_report['strategia'].upper()}")
        if scelta_outlier == "1":
            print(
                f"  Bound train [{col_report['lower_bound_train']:.2f}, "
                f"{col_report['upper_bound_train']:.2f}] → NaN: "
                f"{col_report['n_valori_sostituiti_train']}"
            )
        else:
            print(f"  Outlier DBSCAN → NaN: {col_report['n_valori_sostituiti_train']}")
        print(
            f"  Missing {col} train: "
            f"{col_report['n_missing_train_prima']} → {col_report['n_missing_train_dopo']}"
        )
        print(
            f"  Missing {col} test:  "
            f"{col_report['n_missing_test_prima']} → {col_report['n_missing_test_dopo']}"
        )

    # ── NUOVA FEATURE: monum_flag (age_flag) ───────────────
    # La flag viene calcolata sull'età ORIGINALE dell'edificio (pre-outlier),
    # non sul valore imputato. Motivo:
    #   - IQR:   age > 90 viene messo a NaN poi imputato (≤ 90) → flag sempre 0
    #   - DBSCAN: l'intera riga va a NaN, age imputata ≈ media → flag sempre 0
    # Soluzione: sostituisco temporaneamente age con il valore originale,
    # calcolo il flag, poi ripristino i valori imputati nel dataset.

    # TRAIN
    _banner("NUOVA FEATURE: monum_flag (TRAIN)")
    if age_originale_train is not None:
        age_imputata_train = train_values["age"].copy()          # salvo age imputata
        train_values["age"] = age_originale_train.values         # ripristino age originale
        train_quality_handler.data = train_values
        train_values = train_quality_handler.aggiungi_feature_age_flag(upper_bound=age_upper_bound_train)
        train_values["age"] = age_imputata_train.values          # riporto age imputata
    else:
        train_quality_handler.data = train_values
        train_values = train_quality_handler.aggiungi_feature_age_flag(upper_bound=age_upper_bound_train)

    # TEST (usa age originale, non modificata dall'outlier detection)
    _banner("NUOVA FEATURE: monum_flag (TEST)")
    if age_originale_test is not None:
        test_quality_handler.data = test_values
        # Sul test non applichiamo outlier detection, quindi age_originale_test == age corrente.
        # Assegniamo comunque per coerenza col flusso TRAIN.
        test_values["age"] = age_originale_test.values
        test_values = test_quality_handler.aggiungi_feature_age_flag(upper_bound=age_upper_bound_train)
    else:
        test_quality_handler.data = test_values
        test_values = test_quality_handler.aggiungi_feature_age_flag(upper_bound=age_upper_bound_train)

    # ── VALIDAZIONE ────────────────────────────────────────
    validator_train = DataValidator(train_values)
    validation_report_train = esegui_silenzioso(validator_train.esegui_validazione, verbose=True)
    validator_test = DataValidator(test_values)
    validation_report_test = esegui_silenzioso(validator_test.esegui_validazione, verbose=True)

    # ── STANDARDIZZAZIONE ──────────────────────────────────
    train_quality_handler.data = train_values
    scaler = esegui_silenzioso(train_quality_handler.fit_standardizzazione)
    train_values = esegui_silenzioso(train_quality_handler.applica_standardizzazione, scaler)

    test_quality_handler.data = test_values
    test_values = esegui_silenzioso(test_quality_handler.applica_standardizzazione, scaler)

    # ── ONE-HOT ENCODING ───────────────────────────────────
    train_quality_handler.data = train_values
    ohe_encoder = esegui_silenzioso(train_quality_handler.fit_one_hot_encoding, COLONNE_CATEGORICHE)
    train_values = esegui_silenzioso(
        train_quality_handler.applica_one_hot_encoding, ohe_encoder, COLONNE_CATEGORICHE
    )
    test_quality_handler.data = test_values
    test_values = esegui_silenzioso(
        test_quality_handler.applica_one_hot_encoding, ohe_encoder, COLONNE_CATEGORICHE
    )

    # ── MISSING VALUES (log) ───────────────────────────────
    df_merged = pd.merge(train_values, train_labels, on="building_id")
    handler = MissingValuesHandler(null_threshold=70)
    report = esegui_silenzioso(handler.analizza, df_merged, target_col="damage_grade")
    report["numeric_outlier_imputation"] = imputation_reports

    # ── FASE 3: Geo Embedding ──────────────────────────────
    # Il menu viene mostrato PRIMA del salvataggio perché le geo-feature
    # verranno aggiunte al dataset (da implementare) e salvate insieme.
    scelta_geo = menu_geo_embedding()
    n_hidden = int(scelta_geo)
    _banner("FASE 3 — GEO EMBEDDING (esecuzione)")
    print(f"  GeoFeatureEngineer con {n_hidden} hidden layer.")
    print("  [placeholder — integrazione GeoFeatureEngineer + rete neurale da completare]")
    # TODO: instanziare GeoFeatureEngineer e addestrare embedding NN con n_hidden layer
    # TODO: aggiungere le geo-feature a train_values e test_values prima del salvataggio

    # ── SALVATAGGIO (dopo preprocessing completo + geo embedding) ──
    salva_dataset_preprocessati(train_values, train_labels, test_values)

    # ── FASE 4: Scelta Modello ─────────────────────────────
    scelta_modello = menu_scelta_modello()
    risultati_modello = esegui_modello(scelta_modello, train_values, train_labels, test_values)

    # ── FASE 5: Risultati ──────────────────────────────────
    presenta_risultati(scelta_modello, risultati_modello)

    # ── RIEPILOGO FINALE ───────────────────────────────────
    _banner("PIPELINE COMPLETATA")
    print(f"  Outlier detection:    {'IQR' if scelta_outlier == '1' else 'DBSCAN'}")
    if imputation_reports:
        strategia_usata = next(iter(imputation_reports.values()))["strategia"]
        print(f"  Strategia imputazione: {strategia_usata}")
        print(f"  Colonne imputate:      {list(imputation_reports.keys())}")
    else:
        print("  Nessuna colonna numerica imputata.")
    print(f"  Geo embedding hidden:  {n_hidden}")
    nomi_modelli = {
        "1": "KNN",
        "2": "Albero Decisionale",
        "3": "Random Forest",
        "4": "Multi-Esperto [DA IMPLEMENTARE]",
    }
    print(f"  Modello selezionato:   {nomi_modelli.get(scelta_modello, '?')}")

    best_accuracy_knn = None
    if risultati_knn is not None and not risultati_knn.empty:
        best_accuracy_knn = float(risultati_knn.groupby("strategia")["accuracy"].mean().max())
    if best_accuracy_knn is not None:
        print(f"  Migliore accuracy KNN: {best_accuracy_knn:.6f}")

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
    risultato = main()
    if risultato is not None:
        (
            train_values,
            train_labels,
            test_values,
            report,
            train_quality_report,
            test_quality_report,
            validation_report_train,
            validation_report_test,
        ) = risultato
