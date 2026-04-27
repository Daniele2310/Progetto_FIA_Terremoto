"""
Confronto sperimentale: impatto del moltiplicatore IQR (k) sulla classificazione.

Per ogni valore di k in {1.5, 2.0, 2.5, 3.0, 4.0}:
  1. Rileva outlier sulle 5 feature continue
  2. Sostituisce gli outlier con NaN
  3. Imputa con mediana (strategia semplice e neutra)
  4. Addestra un KNN classifier veloce su un campione
  5. Misura accuracy e F1-micro

Output: tabella comparativa stampata a console + CSV salvato.
"""

import sys
from pathlib import Path

# Aggiungiamo la root del progetto al path per gli import
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from DataPreprocessing.puliziaASCII import PuliziaASCII, COLONNE_CATEGORICHE
from DataPreprocessing.data_cleaning import DataQualityHandler, COLONNE_CONTINUE


# -- Configurazione ----------------------------------------------------------
VALORI_K = [1.5, 2.0, 2.5, 3.0, 4.0]
MAX_ROWS = 25_000          # campione per KNN veloce
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_NEIGHBORS = 5
# -----------------------------------------------------------------------------


def carica_dati():
    """Carica e pulisce i dati usando la pipeline esistente."""
    pulizia = PuliziaASCII()
    train_values, train_labels, _ = pulizia.processa(
        colonne_categoriche=COLONNE_CATEGORICHE
    )
    return train_values, train_labels


def analizza_e_imputa(train_values, k):
    """
    Per un dato k:
      - esegue analisi outlier con quel moltiplicatore
      - sostituisce outlier con NaN
      - imputa con mediana del train (strategia neutra)

    Ritorna il dataframe imputato e un dict di statistiche per colonna.
    """
    handler = DataQualityHandler(train_values)
    handler.pulisci_nomi_colonne()
    handler.analizza_outlier(k=k)
    outliers_df = handler.report["outliers"]

    df = handler.data.copy()
    stats = {}

    for col in COLONNE_CONTINUE:
        if col not in df.columns or col not in outliers_df.index:
            continue

        lower = float(outliers_df.loc[col, "lower_bound"])
        upper = float(outliers_df.loc[col, "upper_bound"])
        n_outliers = int(outliers_df.loc[col, "n_outliers"])
        perc_outliers = float(outliers_df.loc[col, "perc_outliers"])

        # Sostituisci outlier con NaN
        mask = (df[col] < lower) | (df[col] > upper)
        df.loc[mask, col] = np.nan

        # Imputa con mediana (strategia neutra per confronto equo)
        mediana = df[col].median()
        n_nan_prima = int(df[col].isna().sum())
        df[col] = df[col].fillna(mediana)

        stats[col] = {
            "lower_bound": lower,
            "upper_bound": upper,
            "n_outliers": n_outliers,
            "perc_outliers": perc_outliers,
            "mediana_imputazione": mediana,
            "n_nan_imputati": n_nan_prima,
        }

    return df, stats


def valuta_con_knn(df, train_labels, max_rows, test_size, random_state, n_neighbors):
    """
    Addestra un KNN classifier su un campione e misura accuracy + F1-micro.
    """
    # Allinea target
    if "building_id" in df.columns and "building_id" in train_labels.columns:
        merged = pd.merge(df, train_labels, on="building_id")
    else:
        merged = pd.concat([df.reset_index(drop=True),
                            train_labels.reset_index(drop=True)], axis=1)

    y = merged["damage_grade"]
    X = merged.drop(columns=["damage_grade", "building_id"], errors="ignore")

    # One-hot encoding per le categoriche
    X = pd.get_dummies(X, dummy_na=True)

    # Campiona per velocita'
    if len(X) > max_rows:
        X, _, y, _ = train_test_split(
            X, y, train_size=max_rows, stratify=y, random_state=random_state
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Fallback NaN residui
    mediane = X_train.median(numeric_only=True)
    X_train = X_train.fillna(mediane).fillna(0)
    X_val = X_val.fillna(mediane).fillna(0)

    scaler = StandardScaler(with_mean=False)
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors, weights="distance",
        algorithm="brute", n_jobs=-1
    )
    knn.fit(X_train_s, y_train)
    y_pred = knn.predict(X_val_s)

    return {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "f1_micro": float(f1_score(y_val, y_pred, average="micro")),
        "n_campioni_train": len(X_train),
        "n_campioni_val": len(X_val),
        "n_features": X_train.shape[1],
    }


def main():
    print("=" * 80)
    print("CONFRONTO SPERIMENTALE: MOLTIPLICATORE IQR (k)")
    print("=" * 80)

    print("\nCaricamento dati...")
    train_values, train_labels = carica_dati()
    print(f"Dataset caricato: {train_values.shape[0]} righe x {train_values.shape[1]} colonne\n")

    risultati = []

    for k in VALORI_K:
        print(f"\n{'-' * 60}")
        print(f"  Testing k = {k}")
        print(f"{'-' * 60}")

        df_imputato, stats_col = analizza_e_imputa(train_values, k)

        totale_outlier = sum(s["n_outliers"] for s in stats_col.values())
        print(f"  Outlier totali rilevati: {totale_outlier}")
        for col, s in stats_col.items():
            print(f"    {col:25s}: {s['n_outliers']:6d} outlier "
                  f"({s['perc_outliers']:5.2f}%)  "
                  f"bounds=[{s['lower_bound']:.1f}, {s['upper_bound']:.1f}]")

        metriche = valuta_con_knn(
            df_imputato, train_labels,
            max_rows=MAX_ROWS, test_size=TEST_SIZE,
            random_state=RANDOM_STATE, n_neighbors=N_NEIGHBORS
        )

        print(f"  Accuracy:  {metriche['accuracy']:.6f}")
        print(f"  F1-micro:  {metriche['f1_micro']:.6f}")

        riga = {"k": k, "n_outlier_totali": totale_outlier, **metriche}
        for col, s in stats_col.items():
            riga[f"outlier_{col}"] = s["n_outliers"]
        risultati.append(riga)

    # -- Tabella riepilogativa ------------------------------------------------
    df_risultati = pd.DataFrame(risultati)

    print("\n" + "=" * 80)
    print("TABELLA RIEPILOGATIVA")
    print("=" * 80)

    # Colonne principali per la stampa
    cols_stampa = ["k", "n_outlier_totali", "accuracy", "f1_micro"]
    cols_outlier = [c for c in df_risultati.columns if c.startswith("outlier_")]
    print(df_risultati[cols_stampa + cols_outlier].to_string(index=False))

    # Miglior k
    best_idx = df_risultati["f1_micro"].idxmax()
    best_k = df_risultati.loc[best_idx, "k"]
    best_f1 = df_risultati.loc[best_idx, "f1_micro"]
    print(f"\n>>> Miglior k per F1-micro: k={best_k}  (F1-micro={best_f1:.6f})")

    # Salva CSV
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "outlier_k_comparison_results.csv"
    df_risultati.to_csv(output_file, index=False)
    print(f"\nRisultati salvati in: {output_file}")


if __name__ == "__main__":
    main()
