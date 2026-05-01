"""
Outlier Detection multivariato utilizzando DBSCAN.

Questo script applica DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
sulle feature continue del dataset per identificare outlier (punti isolati o in regioni a bassa densità).

I passi principali eseguiti sono:
1. Caricamento dei dati: utilizzo della pipeline esistente per caricare e preparare il dataset.
2. Selezione feature continue: DBSCAN lavora sulle distanze (Euclidea di default), per cui
   è ideale applicarlo su feature continue, piuttosto che mischiate con categoriche.
3. Standardizzazione: i dati vengono scalati in modo che ogni feature abbia media 0 e deviazione
   standard 1. Questo passo è cruciale per i metodi basati sulla distanza, in modo che feature con
   range ampi non dominino su altre.
4. Ricerca di Epsilon (eps): viene usato il metodo del K-distance graph per stimare una distanza
   soglia appropriata, trovando la distanza dal k-esimo vicino.
5. Applicazione di DBSCAN: esecuzione dell'algoritmo di clustering. I punti che non riescono ad 
   essere assegnati a nessun cluster vengono etichettati con -1 (gli outlier).
6. Valutazione e Salvataggio: gli outlier vengono quantificati e salvati in un CSV, ed è calcolato
   un profilo medio delle features.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Aggiungiamo la root del progetto al path per importare correttamente i moduli personalizzati
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from DataPreprocessing.puliziaASCII import PuliziaASCII, COLONNE_CATEGORICHE
from DataPreprocessing.data_cleaning import COLONNE_CONTINUE


# -- Configurazione ----------------------------------------------------------
# Se SAMPLE_SIZE è None, processa tutto il dataset. Se è troppo lento, impostare un intero (es. 50000).
SAMPLE_SIZE = None 
MIN_SAMPLES = 10  # Regola empirica: 2 * numero di dimensioni (feature) = 2 * 5 = 10
# -----------------------------------------------------------------------------


def carica_dati():
    """Carica e pulisce i dati usando la pipeline esistente."""
    pulizia = PuliziaASCII()
    train_values, _, _ = pulizia.processa(
        colonne_categoriche=COLONNE_CATEGORICHE
    )
    return train_values

def trova_eps_ottimale(X, min_samples):
    """
    Calcola le distanze dai k-esimi vicini (dove k = min_samples) e 
    salva un grafico (K-distance graph) per analizzare il parametro 'eps'.
    L'eps ottimale corrisponde generalmente al "gomito" della curva.
    """
    print(f"\nCalcolo distanze per K-distance graph (k={min_samples})...")
    neigh = NearestNeighbors(n_neighbors=min_samples, n_jobs=-1)
    nbrs = neigh.fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    # Prendi la distanza dal k-esimo vicino per ciascun punto
    k_distances = distances[:, -1]
    # Ordina le distanze in modo crescente
    k_distances = np.sort(k_distances)
    
    if HAS_MATPLOTLIB:
        # Crea e salva il plot
        plt.figure(figsize=(10, 6))
        plt.plot(k_distances)
        plt.title(f'K-Distance Graph (k={min_samples})')
        plt.xlabel('Punti ordinati per distanza crescente')
        plt.ylabel(f'Distanza dal {min_samples}-esimo vicino')
        plt.grid(True, linestyle="--", alpha=0.6)
        
        output_dir = Path(__file__).resolve().parent
        plot_path = output_dir / "k_distance_graph.png"
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Grafico K-distance salvato in: {plot_path}")
    else:
        print("matplotlib non installato. Grafico K-distance non salvato.")
    
    # Euristica per trovare l'eps "automaticamente" (ad es. basato sul 98° percentile).
    # L'utente può regolare questo valore osservando il grafico.
    eps_stimato = np.percentile(k_distances, 98)
    print(f"Eps stimato automaticamente (98° percentile delle distanze): {eps_stimato:.3f}")
    return eps_stimato

def esegui_dbscan(df_scaled, eps, min_samples):
    """
    Esegue l'algoritmo DBSCAN.
    """
    print(f"\nEsecuzione DBSCAN con eps={eps:.3f} e min_samples={min_samples}...")
    # n_jobs=-1 parallelizza l'esecuzione su tutti i core disponibili della CPU
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    
    # Effettua il clustering. I punti classificati come -1 sono considerati Rumore / Outlier
    labels = dbscan.fit_predict(df_scaled)
    
    n_outliers = np.sum(labels == -1)
    perc_outliers = (n_outliers / len(labels)) * 100
    
    print(f"-> Outlier rilevati: {n_outliers} su {len(labels)} punti ({perc_outliers:.2f}%)")
    
    return labels

def main():
    print("=" * 80)
    print("OUTLIER DETECTION MULTIVARIATO CON DBSCAN")
    print("=" * 80)

    print("\n1. Caricamento dati...")
    train_values = carica_dati()
    
    if SAMPLE_SIZE and SAMPLE_SIZE < len(train_values):
        print(f"Campionamento attivo: selezione di {SAMPLE_SIZE} righe per test...")
        train_values = train_values.sample(n=SAMPLE_SIZE, random_state=42).copy()
    else:
        print(f"Elaborazione dell'intero dataset: {len(train_values)} righe.")
    
    print("\n2. Selezione feature continue...")
    colonne = [col for col in COLONNE_CONTINUE if col in train_values.columns]
    X_num = train_values[colonne].copy()
    print(f"Feature continue da utilizzare ({len(colonne)}): {colonne}")
    
    if X_num.isna().any().any():
        print(" Trovati valori mancanti: imputazione con mediana...")
        X_num = X_num.fillna(X_num.median())
    
    print("\n3. Standardizzazione dei dati (fondamentale per DBSCAN)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)
    
    print("\n4. Ricerca di un eps (Epsilon) appropriato...")
    eps_stimato = trova_eps_ottimale(X_scaled, MIN_SAMPLES)
    
    print("\n5. Rilevamento Outlier con DBSCAN...")
    labels = esegui_dbscan(X_scaled, eps=eps_stimato, min_samples=MIN_SAMPLES)
    
    print("\n6. Analisi dei risultati...")
    train_values["dbscan_cluster"] = labels
    train_values["is_outlier"] = (labels == -1).astype(int)
    
    # Confrontiamo la media delle feature continue per gli inlier (0) e gli outlier (1)
    profilo = train_values.groupby("is_outlier")[colonne].mean().round(2)
    profilo.index = ["Inlier (0)", "Outlier (1)"]
    print("\nProfilo medio (centri) per Inlier e Outlier:")
    print(profilo.to_string())
    
    # Salvataggio su disco
    output_dir = Path(__file__).resolve().parent
    out_file = output_dir / "dbscan_outliers.csv"
    
    col_da_salvare = ["building_id"] + colonne + ["dbscan_cluster", "is_outlier"]
    train_values[col_da_salvare].to_csv(out_file, index=False)
    
    print(f"\nRisultati di clustering e flag outlier salvati in: {out_file}")

if __name__ == "__main__":
    main()
