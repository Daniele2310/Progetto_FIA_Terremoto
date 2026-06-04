"""
Outlier Detection multivariato utilizzando DBSCAN con Hyperparameter Tuning.

Questo script applica DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
sulle feature continue del dataset per identificare outlier (punti isolati o in regioni a bassa densità).

A differenza di un approccio a parametri fissi, viene eseguita una ricerca sistematica (Grid Search)
sulla griglia di iperparametri (eps, min_samples) per trovare la combinazione ottimale.

Poiché DBSCAN è un algoritmo NON supervisionato, non è possibile utilizzare GridSearchCV di scikit-learn
(che richiede un target). Viene invece utilizzata una valutazione basata su metriche interne di clustering:
    - Silhouette Score: misura quanto ogni punto è coerente con il proprio cluster rispetto ai cluster vicini.
      Range: [-1, 1]. Valori più alti indicano cluster ben separati.
    - Percentuale di outlier: combinazioni che producono troppi outlier (>30%) o troppo pochi (<1%)
      vengono penalizzate, poiché indicano parametri non adatti ai dati.
    - Numero di cluster: un risultato con un solo cluster (o nessun cluster) è degenere e viene scartato.

I passi principali eseguiti sono:
1. Caricamento dei dati: utilizzo della pipeline esistente per caricare e preparare il dataset.
2. Selezione feature continue: DBSCAN lavora sulle distanze (Euclidea di default), per cui
   è ideale applicarlo su feature continue, piuttosto che mischiate con categoriche.
3. Standardizzazione: i dati vengono scalati in modo che ogni feature abbia media 0 e deviazione
   standard 1. Questo passo è cruciale per i metodi basati sulla distanza, in modo che feature con
   range ampi non dominino su altre.
4. Stima del range di Epsilon (eps): viene usato il metodo del K-distance graph per calcolare
   un range ragionevole su cui eseguire la Grid Search.
5. Grid Search sugli iperparametri: per ogni combinazione (eps, min_samples) viene eseguito DBSCAN
   e valutato con Silhouette Score e criteri di qualità.
6. Applicazione del modello migliore: esecuzione dell'algoritmo con i parametri ottimali trovati.
7. Valutazione e Salvataggio: gli outlier vengono quantificati e salvati in un CSV, ed è calcolato
   un profilo medio delle features.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use('Agg')  # Backend non-interattivo: evita crash tkinter
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Aggiungiamo la root del progetto al path per importare correttamente i moduli personalizzati
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.clean_ascii import PuliziaASCII, COLONNE_CATEGORICHE
from src.preprocessing.data_cleaning import COLONNE_CONTINUE


# -- Configurazione ----------------------------------------------------------
# Se SAMPLE_SIZE è None, processa tutto il dataset. Se è troppo lento, impostare un intero (es. 50000).
SAMPLE_SIZE = 30000

# Griglia di iperparametri per la Grid Search
# min_samples: regola empirica di partenza = 2 * n_dimensioni. Esploriamo un range intorno.
MIN_SAMPLES_GRID = [5, 8, 10, 15, 20]

# Il range di eps viene calcolato automaticamente dall'analisi delle k-distanze (vedi sotto).
# N_EPS_VALUES controlla quanti valori di eps testare nel range stimato.
N_EPS_VALUES = 10

# Soglie per filtrare combinazioni degeneri
OUTLIER_PCT_MIN = 1.0    # percentuale minima di outlier accettabile
OUTLIER_PCT_MAX = 30.0   # percentuale massima di outlier accettabile
SILHOUETTE_SAMPLE_SIZE = 2000
RANDOM_STATE = 42
# -----------------------------------------------------------------------------


def carica_dati():
    """Carica e pulisce i dati usando la pipeline esistente."""
    pulizia = PuliziaASCII(cartella_input=str(ROOT / 'Data'))
    train_values, _, _ = pulizia.processa(
        colonne_categoriche=COLONNE_CATEGORICHE
    )
    return train_values

def comprimi_punti_duplicati(X):
    """
    Comprimi i punti duplicati mantenendo una mappatura completa verso il dataset
    originale. Con DBSCAN + sample_weight il risultato finale rimane equivalente,
    ma il costo computazionale cala drasticamente quando ci sono molte righe ripetute.

    Returns:
        tuple: (X_unique, inverse_indices, sample_weight)
            - X_unique: array con i punti unici
            - inverse_indices: per ricostruire le label originali via labels[inverse_indices]
            - sample_weight: numero di occorrenze di ciascun punto unico
    """
    X_unique, inverse_indices, sample_weight = np.unique(
        np.asarray(X), axis=0, return_inverse=True, return_counts=True
    )
    return X_unique, inverse_indices, sample_weight.astype(np.int32, copy=False)


def calcola_range_eps(X, min_samples_values, sample_weight=None):
    """
    Calcola un range ragionevole di valori eps basandosi sulle k-distanze.

    Per ogni valore di min_samples nella griglia, calcola le distanze dal
    k-esimo vicino e usa i percentili (dal 90° al 99°) per definire il range
    di ricerca. Questo approccio è più robusto rispetto al fissare un singolo
    percentile, perché il "gomito" della curva k-distance può trovarsi in
    posizioni diverse a seconda dei dati.

    Returns:
        tuple: (eps_min, eps_max, k_distances_dict)
            - eps_min: estremo inferiore del range di eps
            - eps_max: estremo superiore del range di eps
            - k_distances_dict: dizionario {min_samples: k_distances_ordinate}
    """
    print("\nStima del range di eps tramite K-distance analysis:")
    print("-" * 70)

    all_eps_candidates = []
    k_distances_dict = {}

    max_k = max(min_samples_values)
    n_neighbors = min(max_k, len(X))
    if n_neighbors < max_k:
        raise ValueError(
            f"Impossibile calcolare le k-distanze fino a k={max_k}: "
            f"sono disponibili solo {len(X)} punti."
        )

    neigh = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    neigh.fit(X)
    distances, indices = neigh.kneighbors(X)

    if sample_weight is not None:
        neighbor_weights = sample_weight[indices]
        cumulative_weights = np.cumsum(neighbor_weights, axis=1)
        row_idx = np.arange(len(X))

    for k in min_samples_values:
        if sample_weight is None:
            k_dist = np.sort(distances[:, k - 1])
        else:
            if np.any(cumulative_weights[:, -1] < k):
                raise ValueError(
                    f"Impossibile stimare la distanza per k={k}: il grafo dei vicini "
                    f"non accumula abbastanza peso per almeno un punto."
                )
            kth_positions = np.argmax(cumulative_weights >= k, axis=1)
            k_dist_unique = distances[row_idx, kth_positions]
            # Ripetiamo le distanze in base alle occorrenze per mantenere gli stessi
            # percentili che avremmo osservato sul dataset espanso.
            k_dist = np.sort(np.repeat(k_dist_unique, sample_weight))

        k_distances_dict[k] = k_dist

        # Raccogliamo candidati eps dai percentili chiave
        for pct in [90, 92, 94, 95, 96, 97, 98, 99]:
            all_eps_candidates.append(np.percentile(k_dist, pct))

        print(f"  k={k:2d}  min_distanza={k_dist[0]:8.3f}  max_distanza={k_dist[-1]:8.3f}")

    eps_min = max(np.min(all_eps_candidates), 0.01)  # almeno 0.01
    eps_max = np.max(all_eps_candidates)

    print("-" * 70)
    print(f"Range eps stimato: [{eps_min:.4f}, {eps_max:.4f}]")

    return eps_min, eps_max, k_distances_dict


def salva_k_distance_graph(k_distances_dict, output_dir):
    """
    Salva i grafici K-distance per tutti i valori di min_samples testati.
    Utile per analisi visiva del parametro eps.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib non installato. Grafici K-distance non salvati.")
        return
    
    plt.figure(figsize=(12, 7))
    for k, k_dist in k_distances_dict.items():
        plt.plot(k_dist, label=f'k={k}', alpha=0.8)
    
    plt.title('K-Distance Graph (per diversi valori di min_samples)')
    plt.xlabel('Punti ordinati per distanza crescente')
    plt.ylabel('Distanza dal k-esimo vicino')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plot_path = output_dir / "k_distance_graph.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Grafico K-distance salvato in: {plot_path}")


def esegui_grid_search_dbscan(
    X,
    eps_values,
    min_samples_values,
    verbose=True,
    sample_weight=None,
    inverse_indices=None,
    X_full=None,
    silhouette_sample_size=SILHOUETTE_SAMPLE_SIZE,
    random_state=RANDOM_STATE,
):
    """
    Esegue una Grid Search manuale su DBSCAN.

    Per ogni combinazione (eps, min_samples):
        1. Esegue DBSCAN
        2. Verifica che il risultato non sia degenere (almeno 2 cluster, outlier % nel range)
        3. Calcola il Silhouette Score (solo sugli inlier, cioè i punti assegnati a un cluster)
        4. Registra i risultati

    La combinazione con il Silhouette Score più alto viene scelta come migliore.

    Nota: il Silhouette Score viene calcolato solo sugli inlier perché gli outlier (label=-1)
    non appartengono a nessun cluster e distorcerebbero la metrica.

    Args:
        X: array numpy con i dati standardizzati
        eps_values: lista di valori eps da testare
        min_samples_values: lista di valori min_samples da testare
        verbose: se True, stampa i dettagli di ogni combinazione

    Returns:
        tuple: (risultati_df, miglior_combinazione)
            - risultati_df: DataFrame con tutti i risultati della grid search
            - miglior_combinazione: dict con i parametri e metriche della combinazione migliore
    """
    n_combinazioni = len(eps_values) * len(min_samples_values)
    print(f"\nGrid Search DBSCAN")
    print("-" * 80)
    print(f"  Combinazioni da valutare: {len(eps_values)} eps x {len(min_samples_values)} min_samples = {n_combinazioni}")
    print("-" * 80)
    
    risultati = []
    miglior_score = -1
    miglior_combinazione = None

    n_punti_totali = int(sample_weight.sum()) if sample_weight is not None else len(X)

    for i, eps in enumerate(eps_values):
        for j, min_samples in enumerate(min_samples_values):
            idx = i * len(min_samples_values) + j + 1

            combo_start = time.time()

            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels = dbscan.fit_predict(X, sample_weight=sample_weight)

            tempo = time.time() - combo_start

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if sample_weight is None:
                n_outliers = int(np.sum(labels == -1))
            else:
                n_outliers = int(sample_weight[labels == -1].sum())
            pct_outliers = (n_outliers / n_punti_totali) * 100

            # -- Verifica se il risultato è degenere --
            # Caso 1: nessun cluster trovato (tutti outlier)
            if n_clusters == 0:
                if verbose:
                    print(f"  [{idx:3d}/{n_combinazioni}] eps={eps:.4f}, min_samples={min_samples:3d} "
                          f"- SCARTATO (0 cluster, tutti outlier)")
                risultati.append({
                    "eps": eps, "min_samples": min_samples,
                    "n_clusters": 0, "n_outliers": n_outliers,
                    "pct_outliers": pct_outliers, "silhouette": np.nan,
                    "tempo_s": tempo, "valido": False, "motivo_scarto": "0 cluster"
                })
                continue

            # Caso 2: un solo cluster (tutti nello stesso cluster, nessuna separazione)
            if n_clusters == 1:
                if verbose:
                    print(f"  [{idx:3d}/{n_combinazioni}] eps={eps:.4f}, min_samples={min_samples:3d} "
                          f"- SCARTATO (1 solo cluster, outlier={pct_outliers:.1f}%)")
                risultati.append({
                    "eps": eps, "min_samples": min_samples,
                    "n_clusters": 1, "n_outliers": n_outliers,
                    "pct_outliers": pct_outliers, "silhouette": np.nan,
                    "tempo_s": tempo, "valido": False, "motivo_scarto": "1 solo cluster"
                })
                continue

            # Caso 3: troppi o troppo pochi outlier
            if pct_outliers < OUTLIER_PCT_MIN or pct_outliers > OUTLIER_PCT_MAX:
                motivo = f"outlier {pct_outliers:.1f}% fuori range [{OUTLIER_PCT_MIN}, {OUTLIER_PCT_MAX}]%"
                if verbose:
                    print(f"  [{idx:3d}/{n_combinazioni}] eps={eps:.4f}, min_samples={min_samples:3d} "
                          f"- SCARTATO ({motivo})")
                risultati.append({
                    "eps": eps, "min_samples": min_samples,
                    "n_clusters": n_clusters, "n_outliers": n_outliers,
                    "pct_outliers": pct_outliers, "silhouette": np.nan,
                    "tempo_s": tempo, "valido": False, "motivo_scarto": motivo
                })
                continue

            # -- Calcolo Silhouette Score (solo sugli inlier) --
            mask_inlier = labels != -1

            if mask_inlier.sum() < 2:
                risultati.append({
                    "eps": eps, "min_samples": min_samples,
                    "n_clusters": n_clusters, "n_outliers": n_outliers,
                    "pct_outliers": pct_outliers, "silhouette": np.nan,
                    "tempo_s": time.time() - combo_start,
                    "valido": False,
                    "motivo_scarto": "meno di 2 inlier"
                })
                continue

            labels_full = labels if inverse_indices is None else labels[inverse_indices]
            X_silhouette = X if X_full is None else X_full
            mask_inlier_full = labels_full != -1
            labels_inlier = labels_full[mask_inlier_full]

            if len(np.unique(labels_inlier)) < 2:
                risultati.append({
                    "eps": eps, "min_samples": min_samples,
                    "n_clusters": n_clusters, "n_outliers": n_outliers,
                    "pct_outliers": pct_outliers, "silhouette": np.nan,
                    "tempo_s": time.time() - combo_start,
                    "valido": False,
                    "motivo_scarto": "silhouette non calcolabile (meno di 2 cluster inlier)"
                })
                continue

            try:
                X_inlier = X_silhouette[mask_inlier_full]
                sil_kwargs = {}
                if len(labels_inlier) > silhouette_sample_size:
                    sil_kwargs = {
                        "sample_size": silhouette_sample_size,
                        "random_state": random_state,
                    }
                sil_score = silhouette_score(X_inlier, labels_inlier, **sil_kwargs)
            except ValueError as exc:
                risultati.append({
                    "eps": eps, "min_samples": min_samples,
                    "n_clusters": n_clusters, "n_outliers": n_outliers,
                    "pct_outliers": pct_outliers, "silhouette": np.nan,
                    "tempo_s": time.time() - combo_start,
                    "valido": False,
                    "motivo_scarto": f"silhouette non calcolabile: {exc}"
                })
                continue

            tempo = time.time() - combo_start
            risultati.append({
                "eps": eps, "min_samples": min_samples,
                "n_clusters": n_clusters, "n_outliers": n_outliers,
                "pct_outliers": pct_outliers, "silhouette": sil_score,
                "tempo_s": tempo, "valido": True, "motivo_scarto": ""
            })

            if verbose:
                status = "MIGLIORE" if sil_score > miglior_score else "OK"
                print(f"  [{idx:3d}/{n_combinazioni}] eps={eps:.4f}, min_samples={min_samples:3d} "
                      f"| cluster={n_clusters}, outlier={pct_outliers:.1f}%, "
                      f"silhouette={sil_score:.4f}, tempo={tempo:.1f}s  ({status})")

            # Aggiorna il migliore
            if sil_score > miglior_score:
                miglior_score = sil_score
                miglior_combinazione = {
                    "eps": eps, "min_samples": min_samples,
                    "n_clusters": n_clusters, "n_outliers": n_outliers,
                    "pct_outliers": pct_outliers, "silhouette": sil_score,
                }

    risultati_df = pd.DataFrame(risultati)

    return risultati_df, miglior_combinazione


def salva_risultati_grid_search(risultati_df, output_dir):
    """
    Salva tutti i risultati della Grid Search in un CSV e genera
    un grafico heatmap (se matplotlib è disponibile) per visualizzare
    il Silhouette Score per ogni combinazione di parametri.
    """
    # Salva CSV con tutti i risultati
    csv_path = output_dir / "dbscan_grid_search_results.csv"
    risultati_df.to_csv(csv_path, index=False)
    print(f"\nRisultati Grid Search salvati in: {csv_path}")
    
    if not HAS_MATPLOTLIB:
        return
    
    # Heatmap del Silhouette Score
    validi = risultati_df[risultati_df["valido"] == True].copy()
    if validi.empty:
        print("Nessuna combinazione valida trovata per generare la heatmap.")
        return
    
    pivot = validi.pivot_table(
        values="silhouette", index="min_samples", columns="eps", aggfunc="first"
    )
    
    plt.figure(figsize=(14, 6))
    
    im = plt.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                    interpolation="nearest")
    
    plt.colorbar(im, label="Silhouette Score")
    
    # Etichette asse
    plt.xticks(range(len(pivot.columns)), [f"{v:.3f}" for v in pivot.columns], rotation=45)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("eps")
    plt.ylabel("min_samples")
    plt.title("Grid Search DBSCAN - Silhouette Score")
    
    # Annota i valori nella heatmap
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.3f}", ha="center", va="center",
                         fontsize=8, fontweight="bold",
                         color="white" if val < 0.3 else "black")
    
    plt.tight_layout()
    heatmap_path = output_dir / "dbscan_grid_search_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap Grid Search salvata in: {heatmap_path}")


def esegui_dbscan(df_scaled, eps, min_samples, sample_weight=None, inverse_indices=None):
    """
    Esegue l'algoritmo DBSCAN.
    """
    print(f"\nEsecuzione DBSCAN con eps={eps:.3f} e min_samples={min_samples}...")
    # n_jobs=-1 parallelizza l'esecuzione su tutti i core disponibili della CPU
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)

    # Effettua il clustering. I punti classificati come -1 sono considerati Rumore / Outlier
    labels = dbscan.fit_predict(df_scaled, sample_weight=sample_weight)
    labels_finali = labels if inverse_indices is None else labels[inverse_indices]

    if sample_weight is None:
        n_outliers = int(np.sum(labels == -1))
        n_punti = len(labels_finali)
    else:
        n_outliers = int(sample_weight[labels == -1].sum())
        n_punti = int(sample_weight.sum())
    perc_outliers = (n_outliers / n_punti) * 100

    print(f"-> Outlier rilevati: {n_outliers} su {n_punti} punti ({perc_outliers:.2f}%)")

    return labels_finali

def rileva_outlier_dbscan(train_values, colonne_continue=COLONNE_CONTINUE):
    """
    Esegue outlier detection multivariato con DBSCAN e hyperparameter tuning
    sul training set, restituendo una maschera booleana degli outlier.

    A differenza di main(), questa funzione:
    - Accetta i dati già caricati (non ricarica da file)
    - Non esegue campionamento: elabora tutto il training set
    - Restituisce la maschera outlier e le informazioni del modello
      per l'integrazione nella pipeline di preprocessing del main

    Args:
        train_values: DataFrame del training set (già pulito)
        colonne_continue: lista delle colonne continue su cui applicare DBSCAN

    Returns:
        tuple: (mask_outlier, info_dbscan)
            - mask_outlier: pd.Series booleana (True = outlier), stesso index di train_values
            - info_dbscan: dict con eps, min_samples, silhouette, n_clusters, n_outliers, ecc.
    """
    print("\n" + "=" * 80)
    print("OUTLIER DETECTION MULTIVARIATO CON DBSCAN + HYPERPARAMETER TUNING")
    print("=" * 80)

    print(f"\nElaborazione dell'intero dataset: {len(train_values)} righe.")

    colonne = [col for col in colonne_continue if col in train_values.columns]
    X_num = train_values[colonne].copy()
    print(f"Feature continue da utilizzare ({len(colonne)}): {colonne}")

    if X_num.isna().any().any():
        print("  Trovati valori mancanti: imputazione con mediana...")
        X_num = X_num.fillna(X_num.median())

    print("\nStandardizzazione dei dati (fondamentale per DBSCAN)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    X_unique, inverse_indices, sample_weight = comprimi_punti_duplicati(X_scaled)
    riduzione_pct = 100 * (1 - len(X_unique) / len(X_scaled))
    print(
        f"\nCompressione duplicati: {len(X_scaled)} righe -> {len(X_unique)} punti unici "
        f"({riduzione_pct:.1f}% in meno)"
    )

    print("\nStima del range di eps tramite K-distance analysis...")
    eps_min, eps_max, k_distances_dict = calcola_range_eps(
        X_unique, MIN_SAMPLES_GRID, sample_weight=sample_weight
    )

    eps_values = np.linspace(eps_min, eps_max, N_EPS_VALUES)
    print(f"   Valori eps da testare: {[f'{v:.3f}' for v in eps_values]}")
    print(f"   Valori min_samples da testare: {MIN_SAMPLES_GRID}")

    output_dir = Path(__file__).resolve().parent
    salva_k_distance_graph(k_distances_dict, output_dir)

    print("\nGrid Search sugli iperparametri DBSCAN...")
    start_gs = time.time()
    risultati_df, miglior_comb = esegui_grid_search_dbscan(
        X_unique,
        eps_values,
        MIN_SAMPLES_GRID,
        sample_weight=sample_weight,
        inverse_indices=inverse_indices,
        X_full=X_scaled,
    )
    tempo_gs = time.time() - start_gs

    salva_risultati_grid_search(risultati_df, output_dir)

    n_valide = risultati_df["valido"].sum()
    n_totali = len(risultati_df)
    print(f"\n{'=' * 80}")
    print("RISULTATI GRID SEARCH DBSCAN")
    print(f"{'=' * 80}")
    print(f"   Combinazioni totali testate: {n_totali}")
    print(f"   Combinazioni valide:         {n_valide}")
    print(f"   Tempo totale Grid Search:    {tempo_gs:.1f}s")

    if miglior_comb is None:
        print("\nAVVERTENZA: Nessuna combinazione valida trovata!")
        print("\nSuggerimenti:")
        print("  - Ampliare il range di OUTLIER_PCT_MIN / OUTLIER_PCT_MAX")
        print("  - Modificare MIN_SAMPLES_GRID o N_EPS_VALUES")
        print("  - Verificare i dati in input")
        return pd.Series(False, index=train_values.index), {
            "metodo": "dbscan",
            "n_outliers": 0,
            "nota": "nessuna combinazione valida trovata",
        }

    print(f"\nMIGLIORE COMBINAZIONE TROVATA:")
    print("-" * 70)
    print(f"  eps          = {miglior_comb['eps']:.4f}")
    print(f"  min_samples  = {miglior_comb['min_samples']}")
    print(f"  Silhouette   = {miglior_comb['silhouette']:.4f}")
    print(f"  N. cluster   = {miglior_comb['n_clusters']}")
    print(f"  Outlier      = {miglior_comb['n_outliers']} ({miglior_comb['pct_outliers']:.1f}%)")
    print("-" * 70)

    validi = risultati_df[risultati_df["valido"] == True].sort_values(
        "silhouette", ascending=False
    ).head(5)
    print(f"\nTop 5 combinazioni migliori:")
    print(validi[["eps", "min_samples", "n_clusters", "pct_outliers", "silhouette"]].to_string(index=False))

    print(f"\nApplicazione DBSCAN con i parametri ottimali...")
    best_eps = miglior_comb["eps"]
    best_min_samples = miglior_comb["min_samples"]
    labels = esegui_dbscan(
        X_unique,
        eps=best_eps,
        min_samples=best_min_samples,
        sample_weight=sample_weight,
        inverse_indices=inverse_indices,
    )

    mask_outlier = pd.Series(labels == -1, index=train_values.index)

    # Profilo medio inlier vs outlier
    temp_df = train_values[colonne].copy()
    temp_df["is_outlier"] = mask_outlier.astype(int)
    profilo = temp_df.groupby("is_outlier")[colonne].mean().round(2)
    profilo.index = ["Inlier (0)", "Outlier (1)"]
    print("\nProfilo medio (centri) per Inlier e Outlier:")
    print(profilo.to_string())

    # Salvataggio risultati su disco
    out_file = output_dir / "dbscan_outliers.csv"
    save_df = train_values[["building_id"] + colonne].copy()
    save_df["dbscan_cluster"] = labels
    save_df["is_outlier"] = mask_outlier.astype(int)
    save_df.to_csv(out_file, index=False)
    print(f"\nRisultati di clustering e flag outlier salvati in: {out_file}")

    best_params_file = output_dir / "dbscan_best_params.txt"
    with open(best_params_file, "w", encoding="utf-8") as f:
        f.write("DBSCAN - Migliori Iperparametri (Grid Search)\n")
        f.write("=" * 50 + "\n")
        f.write(f"eps          = {best_eps:.4f}\n")
        f.write(f"min_samples  = {best_min_samples}\n")
        f.write(f"Silhouette   = {miglior_comb['silhouette']:.4f}\n")
        f.write(f"N. cluster   = {miglior_comb['n_clusters']}\n")
        f.write(f"N. outlier   = {miglior_comb['n_outliers']} ({miglior_comb['pct_outliers']:.1f}%)\n")
    print(f"Migliori iperparametri salvati in: {best_params_file}")

    info_dbscan = {
        "metodo": "dbscan",
        "eps": best_eps,
        "min_samples": best_min_samples,
        "silhouette": miglior_comb["silhouette"],
        "n_clusters": miglior_comb["n_clusters"],
        "n_outliers": int(mask_outlier.sum()),
        "pct_outliers": miglior_comb["pct_outliers"],
        "colonne_usate": colonne,
    }

    return mask_outlier, info_dbscan


def main():
    print("=" * 80)
    print("OUTLIER DETECTION MULTIVARIATO CON DBSCAN + HYPERPARAMETER TUNING")
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

    X_unique, inverse_indices, sample_weight = comprimi_punti_duplicati(X_scaled)
    riduzione_pct = 100 * (1 - len(X_unique) / len(X_scaled))
    print(
        f"\n4. Compressione duplicati: {len(X_scaled)} righe -> {len(X_unique)} punti unici "
        f"({riduzione_pct:.1f}% in meno)"
    )

    print("\n5. Stima del range di eps tramite K-distance analysis...")
    eps_min, eps_max, k_distances_dict = calcola_range_eps(
        X_unique, MIN_SAMPLES_GRID, sample_weight=sample_weight
    )

    # Genera la griglia di valori eps (distribuiti uniformemente nel range stimato)
    eps_values = np.linspace(eps_min, eps_max, N_EPS_VALUES)
    print(f"   Valori eps da testare: {[f'{v:.3f}' for v in eps_values]}")
    print(f"   Valori min_samples da testare: {MIN_SAMPLES_GRID}")
    
    output_dir = Path(__file__).resolve().parent
    
    # Salva il grafico K-distance
    salva_k_distance_graph(k_distances_dict, output_dir)
    
    print("\n5. Grid Search sugli iperparametri DBSCAN...")
    start_gs = time.time()
    risultati_df, miglior_comb = esegui_grid_search_dbscan(
        X_unique,
        eps_values,
        MIN_SAMPLES_GRID,
        sample_weight=sample_weight,
        inverse_indices=inverse_indices,
        X_full=X_scaled,
    )
    tempo_gs = time.time() - start_gs
    
    # Salva risultati e heatmap della grid search
    salva_risultati_grid_search(risultati_df, output_dir)
    
    # Report Grid Search
    n_valide = risultati_df["valido"].sum()
    n_totali = len(risultati_df)
    print(f"\n{'=' * 80}")
    print(f"RISULTATI GRID SEARCH DBSCAN")
    print(f"{'=' * 80}")
    print(f"   Combinazioni totali testate: {n_totali}")
    print(f"   Combinazioni valide:         {n_valide}")
    print(f"   Tempo totale Grid Search:    {tempo_gs:.1f}s")
    
    if miglior_comb is None:
        print("\nAVVERTENZA: Nessuna combinazione valida trovata!")
        print("\nSuggerimenti:")
        print("  - Ampliare il range di OUTLIER_PCT_MIN / OUTLIER_PCT_MAX")
        print("  - Modificare MIN_SAMPLES_GRID o N_EPS_VALUES")
        print("  - Verificare i dati in input")
        return
    
    print(f"\nMIGLIORE COMBINAZIONE TROVATA:")
    print("-" * 70)
    print(f"  eps          = {miglior_comb['eps']:.4f}")
    print(f"  min_samples  = {miglior_comb['min_samples']}")
    print(f"  Silhouette   = {miglior_comb['silhouette']:.4f}")
    print(f"  N. cluster   = {miglior_comb['n_clusters']}")
    print(f"  Outlier      = {miglior_comb['n_outliers']} ({miglior_comb['pct_outliers']:.1f}%)")
    print("-" * 70)
    
    # Top 5 combinazioni
    validi = risultati_df[risultati_df["valido"] == True].sort_values(
        "silhouette", ascending=False
    ).head(5)
    print(f"\nTop 5 combinazioni migliori:")
    print(validi[["eps", "min_samples", "n_clusters", "pct_outliers", "silhouette"]].to_string(index=False))
    
    print(f"\n6. Applicazione DBSCAN con i parametri ottimali...")
    best_eps = miglior_comb["eps"]
    best_min_samples = miglior_comb["min_samples"]
    labels = esegui_dbscan(
        X_unique,
        eps=best_eps,
        min_samples=best_min_samples,
        sample_weight=sample_weight,
        inverse_indices=inverse_indices,
    )
    
    print("\n7. Analisi dei risultati...")
    train_values["dbscan_cluster"] = labels
    train_values["is_outlier"] = (labels == -1).astype(int)
    
    # Confrontiamo la media delle feature continue per gli inlier (0) e gli outlier (1)
    profilo = train_values.groupby("is_outlier")[colonne].mean().round(2)
    profilo.index = ["Inlier (0)", "Outlier (1)"]
    print("\nProfilo medio (centri) per Inlier e Outlier:")
    print(profilo.to_string())
    
    # Salvataggio su disco
    out_file = output_dir / "dbscan_outliers.csv"
    
    col_da_salvare = ["building_id"] + colonne + ["dbscan_cluster", "is_outlier"]
    train_values[col_da_salvare].to_csv(out_file, index=False)
    
    print(f"\nRisultati di clustering e flag outlier salvati in: {out_file}")

    # Salva anche i migliori iperparametri in un file separato per reference
    best_params_file = output_dir / "dbscan_best_params.txt"
    with open(best_params_file, "w", encoding="utf-8") as f:
        f.write("DBSCAN - Migliori Iperparametri (Grid Search)\n")
        f.write("=" * 50 + "\n")
        f.write(f"eps          = {best_eps:.4f}\n")
        f.write(f"min_samples  = {best_min_samples}\n")
        f.write(f"Silhouette   = {miglior_comb['silhouette']:.4f}\n")
        f.write(f"N. cluster   = {miglior_comb['n_clusters']}\n")
        f.write(f"N. outlier   = {miglior_comb['n_outliers']} ({miglior_comb['pct_outliers']:.1f}%)\n")
    print(f"Migliori iperparametri salvati in: {best_params_file}")

if __name__ == "__main__":
    main()
