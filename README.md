# Progetto FIA - Terremoto (Richter's Predictor)

Predizione del livello di danno agli edifici colpiti dal terremoto del 2015 Gorkha in Nepal.

**Competizione**: [DrivenData: Richter's Predictor](https://www.drivendata.org/competitions/57/nepal-earthquake/)  
**Partecipanti**: 8.653 iscritti | **Difficoltà**: Intermediate Practice

---

## Obiettivo

L'obiettivo del progetto è **classificare il grado di danno** di edifici colpiti dal terremoto di Gorkha (2015) sulla base di caratteristiche strutturali, costruttive e geografiche.

### Classificazione Ordinale
La variabile target `damage_grade` è **ordinale** con 3 classi:
- **Grado 1**: Low damage (danno basso)
- **Grado 2**: Medium amount of damage (danno medio)
- **Grado 3**: Almost complete destruction (distruzione quasi totale)

L'ordine delle classi è significativo (1 < 2 < 3), quindi è un problema di **Ordinal Regression**.

### Metrica di Valutazione
**F1 Score (Micro-averaged)**:
```
F_micro = 2 × P_micro × R_micro / (P_micro + R_micro)
```

Calcolabile con:
```python
from sklearn.metrics import f1_score
f1_score(y_true, y_pred, average='micro')
```

---

## Dataset

### Fonte e Dimensione
Il dataset è stato raccolto da:
- **Kathmandu Living Labs**
- **Central Bureau of Statistics** (Nepal)

Uno dei più grandi dataset post-disastro mai raccolti (260.601 campioni di training).

### File Disponibili
| File | Descrizione |
|------|-------------|
| `Data/raw/train_values.csv` | 260.601 edifici × 39 colonne (1 building_id + 38 feature) |
| `Data/raw/train_labels.csv` | Etichette di danno per i 260.601 edifici |
| `Data/raw/test_values.csv` | ~86.868 edifici da predire |
| `Data/raw/submission_format.csv` | Template CSV per le submissions |

### Caratteristiche del Dataset

**Importanti**:
- Variabili categoriche sono offuscate con caratteri ASCII casuali (es. `a`, `b`, `c`)
- L'apparizione dello stesso carattere in colonne diverse **NON** implica lo stesso valore
- Sono presenti **valori mancanti** (NaN) distribuiti nelle feature

---

## Feature del Dataset (38 Feature)

### Geografiche (3)
| Feature | Tipo | Valori | Descrizione |
|---------|------|--------|-------------|
| `geo_level_1_id` | int | 0-30 | Regione geografica di livello 1 (più grande) |
| `geo_level_2_id` | int | 0-1.427 | Regione geografica di livello 2 |
| `geo_level_3_id` | int | 0-12.567 | Regione geografica di livello 3 (più specifica) |

### Strutturali (5)
| Feature | Tipo | Descrizione |
|---------|------|-------------|
| `count_floors_pre_eq` | int | Numero di piani prima del terremoto |
| `age` | int | Età dell'edificio in anni |
| `area_percentage` | int | Area normalizzata della base |
| `height_percentage` | int | Altezza normalizzata della base |
| `land_surface_condition` | cat | Condizione della superficie (n, o, t) |

### Costruzione (9)
| Feature | Tipo | Descrizione |
|---------|------|-------------|
| `foundation_type` | cat | Tipo di fondamenta (h, i, r, u, w) |
| `roof_type` | cat | Tipo di tetto (n, q, x) |
| `ground_floor_type` | cat | Tipo di pavimento al piano terra (f, m, v, x, z) |
| `other_floor_type` | cat | Tipo di piani superiori (j, q, s, x) |
| `position` | cat | Posizione dell'edificio (j, o, s, t) |
| `plan_configuration` | cat | Configurazione della pianta (a, c, d, f, m, n, o, q, s, u) |

### Superstructure/Materiali (11)
Indicatori binari (0/1) per tipologie di sovrastruttura:
- `has_superstructure_adobe_mud` — Adobe/Fango
- `has_superstructure_mud_mortar_stone` — Pietra con mortaio di fango
- `has_superstructure_stone_flag` — Pietra
- `has_superstructure_cement_mortar_stone` — Pietra con mortaio di cemento
- `has_superstructure_mud_mortar_brick` — Mattone con mortaio di fango
- `has_superstructure_cement_mortar_brick` — Mattone con mortaio di cemento
- `has_superstructure_timber` — Legno
- `has_superstructure_bamboo` — Bambù
- `has_superstructure_rc_engineered` — Cemento armato (engineered)
- `has_superstructure_rc_non_engineered` — Cemento armato (non-engineered)
- `has_superstructure_other` — Altri materiali

### Proprietà e Utilizzo (8+)
| Feature | Tipo | Descrizione |
|---------|------|-------------|
| `legal_ownership_status` | cat | Status di proprietà legale (a, r, v, w) |
| `count_families` | int | Numero di famiglie residenti |
| `has_secondary_use` | bin | Utilizzo secondario |
| **Secondary Uses** | bin | Agricoltura, Hotel, Affitti, Istituzione, Scuola, Industria, Centro sanitario, Ufficio governativo, Stazione di polizia, Altro |

---

## Preprocessing dei Dati

### Moduli Implementati

#### 1. Data Cleaning (`src/preprocessing/data_cleaning.py`)
- Verifica integrità dataset e rimozione duplicati
- Validazione range feature numeriche
- Rilevamento outlier con metodo **IQR parametrico** (moltiplicatore `k=3.0` — Extreme IQR, validato sperimentalmente come ottimale tra k∈{1.5, 2.0, 2.5, 3.0, 4.0})
- Aggiunta feature `monum_flag`: flag booleano per edifici con età anomala ma non sentinel

#### 2. Outlier Detection Multivariata con DBSCAN (`src/preprocessing/outlier_detection/DBSCAN.py`)

DBSCAN viene proposto come alternativa multivariata all'IQR. Invece di analizzare
ogni colonna separatamente, considera simultaneamente le feature continue e
identifica come outlier le righe collocate in regioni a bassa densità.

La procedura implementata è la seguente:

1. **Selezione delle feature continue**: il clustering utilizza solo le variabili
   numeriche continue, perché DBSCAN si basa sulla distanza tra osservazioni.
2. **Preparazione temporanea dei dati**: eventuali valori mancanti vengono
   riempiti con la mediana esclusivamente per consentire il calcolo delle
   distanze durante la detection.
3. **Standardizzazione**: `StandardScaler` porta le feature sulla stessa scala,
   evitando che le variabili con range più ampio dominino la distanza euclidea.
4. **Compressione dei duplicati**: i punti numerici identici vengono rappresentati
   una sola volta e associati a un `sample_weight`. DBSCAN mantiene così il peso
   reale di ogni configurazione, riducendo il costo computazionale.
5. **Stima automatica di `eps`**: per ciascun valore candidato di `min_samples`
   vengono calcolate le distanze dal k-esimo vicino. I percentili alti delle
   k-distanze definiscono un intervallo ragionevole di valori `eps`.
6. **Grid search non supervisionata**: vengono provate le combinazioni tra i
   valori di `eps` stimati e la griglia di `min_samples`. Non essendoci un target,
   la selezione non usa `GridSearchCV`.
7. **Filtro delle soluzioni degeneri**: vengono scartate le configurazioni che
   producono meno di due cluster oppure una percentuale di outlier esterna
   all'intervallo accettato dalla pipeline.
8. **Scelta dei parametri**: tra le configurazioni valide viene selezionata quella
   con il migliore Silhouette Score, calcolato soltanto sugli inlier perché i
   punti con etichetta DBSCAN `-1` rappresentano rumore.

La configurazione selezionata viene applicata all'intero training set. Le righe
con label `-1` formano la maschera degli outlier. Nel flusso principale, per
queste righe le feature continue vengono convertite in `NaN` e successivamente
ricostruite mediante la strategia di imputazione scelta. La detection non viene
fittata sul test set.

#### 3. Missing Values (`src/preprocessing/missing_values.py`)
Gestione NaN con strategie di imputazione selezionabili via menu interattivo:

**Outlier Handling**:
- Valori `age` nel range [250, 995] → convertiti in NaN (valori sentinel)

**Strategie di Imputazione** (selezionabili via menu):
1. **Univariata - Media**
2. **Univariata - Mediana**
3. **Multivariata - Regressione Lineare** (mediana per gruppi geografici gerarchici)
4. **KNN Predictor**


#### 4. Pattern Strategy per Imputazione (`src/preprocessing/imputation_strategies.py`)
Implementazione del **Design Pattern Strategy** per isolare la logica di selezione dell'algoritmo dal flusso principale:
- Interfaccia astratta `ImputationStrategy`
- Quattro strategie concrete intercambiabili
- `ImputationContext` come punto di delega
- Registry `STRATEGIE_IMPUTAZIONE` per selezione a runtime

#### 5. Codifica Categorica
**OneHotEncoder** di scikit-learn con `handle_unknown='ignore'`, scelto rispetto a `get_dummies()` per:
- Allineamento automatico Train/Test (stesse colonne garantite)
- Gestione di categorie mai viste nel test set
- Prevenzione del data leakage

#### 6. Standardizzazione
**StandardScaler** fittato sul train e applicato a train e test separatamente.

#### 7. ASCII Cleaning (`src/preprocessing/clean_ascii.py`)
Normalizzazione encoding caratteri categorici offuscati.

---

## Feature Selection

Sono stati implementati **7 metodi di feature selection** con approcci diversi, poi confrontati in un benchmark rigoroso.

### Metodi di Ranking

#### Information Gain (Entropia)
- **File**: `src/feature_selection/feature_ranking/uncertainty_information_gain_ranking.py`
- **Top Features**: `geo_level_3_id` (IG=0.482), `geo_level_2_id` (IG=0.346), `geo_level_1_id` (IG=0.190)
- **Meno informative**: feature `has_secondary_use_*` (IG≈0)

#### RELIEF
- **File**: `src/feature_selection/feature_ranking/relief_ranking.py`
- **Top Features**: `geo_level_3_id`, `geo_level_2_id`, `has_superstructure_cement_mortar_stone`
- Coerente con Information Gain: la geografia domina il ranking supervisionato

#### Correlazione di Pearson
- **File**: `src/feature_selection/feature_ranking/pairwise_correlation_ranking.py`
- **Top correlate col target**: `foundation_type_r` (0.343), `ground_floor_type_v` (0.319)
- **Correlazioni negative forti** tra dummy della stessa variabile categorica (atteso con OHE)

### Metodi di Subset Selection

#### Sequential Forward Selection (SFS)
- **File**: `src/feature_selection/subset_selection/sfs.py`
- **Approccio**: Forward — parte da 0 feature, aggiunge iterativamente la migliore
- **Estimator**: LogisticRegression o KNeighborsClassifier
- **Stop**: quando lo score smette di crescere

#### Sequential Backward Selection (SBS)
- **File**: `src/feature_selection/subset_selection/sbs_subset_selection.py`
- **Approccio**: Backward — parte da tutte le feature, rimuove iterativamente la peggiore


#### Stepwise Bidirectional Selection
- **File**: `src/feature_selection/subset_selection/bidirectional_subset_selection.py`
- **Approccio**: Alternanza di step forward e backward per ciclo
- **Stop**: quando un intero ciclo non produce miglioramenti

#### Best First Search
- **File**: `src/feature_selection/subset_selection/best_first.py`
- **Approccio**: Priority queue con espansione greedy, patience k=5

#### Max-Min Subset Selection
- **File**: `src/feature_selection/subset_selection/max_min_subset_selection.py`
- **Formula**: `score(f) = |corr(f, target)| - max|corr(f, selected_set)|`


#### Embedded Lasso Regression
- **File**: `src/feature_selection/embedded/lasso_feature_selection.py`
- **Approccio**: Regolarizzazione L1 durante l'addestramento
- **Alpha**: selezionabile (LassoCV automatico o valore fisso)


#### PCA
- **File**: `src/feature_selection/feature_ranking/pca.py`
- **Approccio**: Riduzione dimensionale non supervisionata
- **Feature escluse dal fit**: `building_id`, `geo_level_*_id`, `damage_grade`


### Benchmark Rigoroso di Feature Selection
Benchmark finale con campionamento bilanciato (~30.000 campioni), GridSearchCV con K∈[3,5,9,15,21], K ottimale trovato = 21.

**Classifica finale (Top 3)**:

| Posizione | Metodo | F1-Micro | Feature Selezionate |
|-----------|--------|----------|---------------------|
| 1 | Sequential Backward Selection (SBS) | 0.5450 | 30 |
| 2 | Best First Search | 0.5417 | 17 |
| 3 | Relief Ranking | 0.5385 | 15 (taglio prefissato) |

I tre metodi confermati per l'integrazione nella pipeline principale sono **SBS**, **Best First Search** e **Relief**.

---


## Analisi delle Feature Geografiche

Le variabili `geo_level_1_id`, `geo_level_2_id` e `geo_level_3_id` sono codici
categorici organizzati secondo una gerarchia territoriale. La pipeline permette
di trasformarle scegliendo tra due approcci alternativi:

1. **Geo-feature statiche e aggregate**, costruite tramite statistiche
   gerarchiche supervisionate.
2. **Embedding neurale supervisionato**, appreso mediante una rete PyTorch.

In entrambi i casi le feature geografiche originali vengono mantenute e le nuove
rappresentazioni vengono aggiunte al dataset.

### Geo-Feature Statiche e Aggregate (`src/preprocessing/geo_features.py`)

`GeoFeatureEngineer` trasforma gli ID geografici in statistiche numeriche che
descrivono frequenza, rarità e distribuzione del danno nelle diverse aree. La
struttura gerarchica viene rappresentata usando le chiavi:

- livello 1: `geo_level_1_id`;
- livello 2: coppia `(geo_level_1_id, geo_level_2_id)`;
- livello 3: terna `(geo_level_1_id, geo_level_2_id, geo_level_3_id)`.

Per ogni livello vengono generate:

- **count**: numero di edifici appartenenti all'area;
- **frequency**: frequenza relativa dell'area nel training set;
- **is_rare**: flag attivo quando il numero di osservazioni è inferiore alla
  soglia configurata;
- **target mean smoothed**: media regolarizzata di `damage_grade`;
- **probabilità smoothed per classe**: stima regolarizzata della probabilità di
  ciascun grado di danno.

Le statistiche supervisionate utilizzano uno **smoothing gerarchico**:

```text
valore_smoothed =
    (numero_osservazioni × valore_area + smoothing × valore_padre)
    / (numero_osservazioni + smoothing)
```

Il valore padre del livello 3 è la statistica del livello 2; quello del livello
2 deriva dal livello 1; il livello 1 usa la statistica globale. In questo modo,
le aree con pochi edifici vengono regolarizzate verso un contesto geografico più
ampio invece di produrre stime instabili.

La stessa gerarchia viene usata in trasformazione come meccanismo di fallback:
se un'area di livello 3 non è conosciuta, si utilizza il livello 2, poi il
livello 1 e infine il valore globale. Vengono inoltre prodotte feature finali
come `geo_hierarchical_target_mean` e
`geo_hierarchical_class_<classe>_prob`.

Per evitare target leakage, le geo-feature del training set vengono costruite
con `fit_transform_oof`: il train viene diviso tramite `StratifiedKFold` e ogni
riga riceve statistiche calcolate esclusivamente sugli altri fold. Terminata la
generazione OOF, l'engineer viene fittato sull'intero train e utilizzato per
trasformare il test set.

### Embedding Neurale Supervisionato (`src/geo_embedding/embedding_extractor.py`)

Gli identificativi geografici sono codici categorici gerarchici, non misure
numeriche continue. La distanza aritmetica tra due ID non descrive quindi una
reale vicinanza geografica. La rete neurale apprende una rappresentazione densa
dei tre livelli usando `damage_grade` come segnale supervisionato.

Il flusso implementato è composto dai seguenti passaggi:

1. **Mapping delle categorie**: per ogni `geo_level_*_id`, le categorie osservate
   nel train vengono convertite in indici consecutivi a partire da `1`.
   L'indice `0` è riservato a valori mancanti o categorie del test non viste nel
   training set.
2. **Embedding separati per livello**: ciascun livello geografico possiede una
   propria matrice `nn.Embedding`. Le dimensioni predefinite sono rispettivamente
   `4`, `8` e `16`, così i livelli con maggiore cardinalità possono apprendere
   rappresentazioni più ricche.
3. **Fusione gerarchica**: i tre vettori di embedding vengono concatenati in un
   unico vettore che rappresenta contemporaneamente regione, sotto-regione e
   area locale dell'edificio.
4. **Encoder denso**: il vettore concatenato attraversa un livello lineare,
   `BatchNorm`, attivazione `ReLU` e `Dropout`. L'uscita dell'encoder è il codice
   geografico compatto; la sua dimensione è selezionabile dalla pipeline.
5. **Obiettivo supervisionato**: un classificatore lineare trasforma il codice
   nascosto nei logits delle classi di danno. La rete viene addestrata con
   `CrossEntropyLoss` e ottimizzatore `AdamW`.
6. **Arresto anticipato**: durante l'addestramento viene conservato lo stato con
   loss media più bassa; se la loss non migliora per il numero configurato di
   epoche, il training viene interrotto e viene ripristinato lo stato migliore.
7. **Estrazione delle feature**: terminato il training, il classificatore finale
   non viene usato come modello della pipeline. Vengono invece estratti i vettori
   prodotti dall'encoder e aggiunti ai dataset come colonne
   `geo_hidden_0`, `geo_hidden_1`, ..., mantenendo anche le feature originali.

Nel `main.py` è possibile scegliere la dimensione del codice nascosto tra `12`,
`16` e `20`. La rete viene addestrata sul training set e lo stesso mapping
geografico appreso sul train viene riutilizzato per generare le feature del test,
senza apprendere nuove categorie da quest'ultimo.

---

## Struttura del Progetto

```
Progetto_FIA_Terremoto/
│
├── README.md
├── DocumentoDiBordo.txt               # Diario di lavoro dettagliato (cronologico)
├── main.py                            # Pipeline principale di preprocessing
│
├── Data/
│   ├── train_values.csv
│   ├── train_labels.csv
│   ├── test_values.csv
│   └── submission_format.csv
│
├── Data/
│   ├── raw/
│   │   ├── train_values.csv
│   │   ├── train_labels.csv
│   │   ├── test_values.csv
│   │   └── submission_format.csv
│   └── preprocessed/
│       ├── train_values_preprocessed.csv
│       ├── test_values_preprocessed.csv
│       └── train_features_labels_preprocessed.csv
│
├── src/
│   ├── preprocessing/
│   │   ├── clean_ascii.py
│   │   ├── data_cleaning.py
│   │   ├── data_selection.py
│   │   ├── imputation_strategies.py   # Pattern Strategy
│   │   ├── missing_values.py
│   │   ├── geo_features.py
│   │   ├── validation.py
│   │   └── outlier_detection/
│   │       ├── DBSCAN.py
│   │       └── outlier_k_comparison.py
│   │
│   ├── geo_embedding/
│   │   └── embedding_extractor.py
│   │
│   ├── feature_selection/
│   │   ├── Hyperparameter_Tuning.py
│   │   ├── feature_ranking/
│   │   │   ├── pairwise_correlation_ranking.py
│   │   │   ├── relief_ranking.py
│   │   │   ├── uncertainty_information_gain_ranking.py
│   │   │   └── pca.py
│   │   ├── subset_selection/
│   │   │   ├── sfs.py
│   │   │   ├── sbs_subset_selection.py
│   │   │   ├── bidirectional_subset_selection.py
│   │   │   ├── best_first.py
│   │   │   └── max_min_subset_selection.py
│   │   └── embedded/
│   │       └── lasso_feature_selection.py
│
├── experiments/
│   └── evaluate_feature_selection.py
│
└── requirements.txt
```

---

## Quick Start

### Setup Iniziale

```bash
# Attivare ambiente virtuale
.\venv\Scripts\Activate.ps1   # Windows
source venv/bin/activate       # Linux/Mac

# Installare dipendenze
pip install -r requirements.txt
```

### Preprocessing Completo

```bash
python main.py
```

### Feature Selection Standalone

```bash
# Sequential Forward Selection
python src/feature_selection/subset_selection/sfs.py --estimator knn --scoring f1_micro

# Sequential Backward Selection
python src/feature_selection/subset_selection/sbs_subset_selection.py --min-features 20

# Bidirectional Stepwise
python src/feature_selection/subset_selection/bidirectional_subset_selection.py

# Best First Search
python src/feature_selection/subset_selection/best_first.py --patience 5

# Lasso Embedded
python src/feature_selection/embedded/lasso_feature_selection.py --alpha 0.002

# Max-Min Strategy
python src/feature_selection/subset_selection/max_min_subset_selection.py --max-features 15

# PCA
python src/feature_selection/feature_ranking/pca.py
```

### Benchmark e Valutazione

```bash
# Benchmark feature selection (solo KNN)
python experiments/evaluate_feature_selection.py

# Benchmark completo (KNN + RF + DT)
python experiments/evaluate_feature_selection.py --full-tuning

# Valutazione sistema multi-esperto
python experiments/evaluate_multi_expert.py

# Hyperparameter tuning
python experiments/tune_multi_expert_hyperparameters.py
```

---

## Decisioni Implementative Chiave

**Pattern Strategy per imputazione** — separa la logica di scelta algoritmo dal main; estendibile aggiungendo solo una nuova classe concreta.

**OneHotEncoder vs get_dummies()** — garantisce allineamento Train/Test e gestione categorie sconosciute senza data leakage.

**Esclusione Geographic IDs da PCA** — i geo_level_id alteravano la scala della varianza rendendo lo scree plot illeggibile; vengono riallegati dopo la trasformazione.

**k=3.0 per IQR** — validato sperimentalmente come ottimale tra k∈{1.5, 2.0, 2.5, 3.0, 4.0}; rimuove solo gli outlier genuini senza perdere dati validi.

---

## Documentazione Estesa

Per dettagli completi su ogni decisione, esperimento e iterazione consultare:
```
DocumentoDiBordo.txt
```
Diario cronologico che documenta scelte metodologiche, esperimenti comparativi, output di ogni fase e note sulle problematiche risolte.


