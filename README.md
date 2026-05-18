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
| `Data/train_values.csv` | 260.601 edifici × 39 colonne (1 building_id + 38 feature) |
| `Data/train_labels.csv` | Etichette di danno per i 260.601 edifici |
| `Data/test_values.csv` | ~86.868 edifici da predire |
| `Data/submission_format.csv` | Template CSV per le submissions |

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

#### 2. Missing Values (`src/preprocessing/missing_values.py`)
Gestione NaN con strategie di imputazione selezionabili via menu interattivo:

**Outlier Handling**:
- Valori `age` nel range [250, 995] → convertiti in NaN (valori sentinel)

**Strategie di Imputazione** (selezionabili via menu):
1. **Univariata - Media**
2. **Univariata - Mediana**
3. **Multivariata - Regressione Lineare** (mediana per gruppi geografici gerarchici)
4. **KNN Predictor**
5. **Valutazione Automatica**: confronto rapido con KNN veloce e selezione del metodo migliore

**Validazione delle strategie** (confronto su holdout 80/20):

| Strategia | Accuracy | F1 Macro | Balanced Accuracy |
|-----------|----------|----------|-------------------|
| Univariata media | 0.5785 | — | — |
| KNN predictor | 0.5785 | — | — |
| Univariata mediana | 0.5783 | — | — |
| Multivariata regressione lineare | 0.5783 | — | — |

La strategia migliore viene selezionata automaticamente (opzione 5).

#### 3. Pattern Strategy per Imputazione (`src/preprocessing/imputation_strategies.py`)
Implementazione del **Design Pattern Strategy** per isolare la logica di selezione dell'algoritmo dal flusso principale:
- Interfaccia astratta `ImputationStrategy`
- Quattro strategie concrete intercambiabili
- `ImputationContext` come punto di delega
- Registry `STRATEGIE_IMPUTAZIONE` per selezione a runtime

#### 4. Codifica Categorica
**OneHotEncoder** di scikit-learn con `handle_unknown='ignore'`, scelto rispetto a `get_dummies()` per:
- Allineamento automatico Train/Test (stesse colonne garantite)
- Gestione di categorie mai viste nel test set
- Prevenzione del data leakage

#### 5. Standardizzazione
**StandardScaler** fittato sul train e applicato a train e test separatamente.

#### 6. ASCII Cleaning (`src/preprocessing/clean_ascii.py`)
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
- **Risultati**: 7 feature selezionate (age, area_percentage, has_superstructure_stone_flag, has_superstructure_rc_engineered, foundation_type_h, foundation_type_i, roof_type_x)
- **Stop**: quando lo score smette di crescere

#### Sequential Backward Selection (SBS)
- **File**: `src/feature_selection/subset_selection/sbs_subset_selection.py`
- **Approccio**: Backward — parte da tutte le feature, rimuove iterativamente la peggiore
- **Risultati**: 68 → 28 feature; accuracy 0.533 → 0.622 (+8.8%)
- **Costo**: O(p²), 1.969 modelli valutati in 29.26s

#### Stepwise Bidirectional Selection
- **File**: `src/feature_selection/subset_selection/bidirectional_subset_selection.py`
- **Approccio**: Alternanza di step forward e backward per ciclo
- **Risultati**: 68 → 9 feature; accuracy 0.533 → 0.655 (+12.2%)
- **Stop**: quando un intero ciclo non produce miglioramenti

#### Best First Search
- **File**: `src/feature_selection/subset_selection/best_first.py`
- **Approccio**: Priority queue con espansione greedy, patience k=5
- **Valutazione**: Decision Tree con 5-fold CV
- **Risultati**: **82.6% di riduzione** (69 → 12 feature), accuracy **0.6913**
- **Nota**: riduzione significativa con miglioramento delle prestazioni rispetto alla baseline del 57%

#### Max-Min Subset Selection
- **File**: `src/feature_selection/subset_selection/max_min_subset_selection.py`
- **Formula**: `score(f) = |corr(f, target)| - max|corr(f, selected_set)|`
- **Risultati**: 6 feature selezionate (stop su score negativo)

#### Embedded Lasso Regression
- **File**: `src/feature_selection/embedded/lasso_feature_selection.py`
- **Approccio**: Regolarizzazione L1 durante l'addestramento
- **Alpha**: selezionabile (LassoCV automatico o valore fisso)
- **Valore consigliato**: `alpha=0.002` → 49 feature, prestazioni stabili
- **Feature stabili**: presenti in tutte le prove (count_families, foundation_type_r, ground_floor_type_v, has_superstructure_mud_mortar_stone, ...)

#### PCA
- **File**: `src/feature_selection/feature_ranking/pca.py`
- **Approccio**: Riduzione dimensionale non supervisionata
- **Feature escluse dal fit**: `building_id`, `geo_level_*_id`, `damage_grade`
- **Output**: scree plot + tabella varianza per scelta manuale del gomito

### Analisi Monotonia e Branch-and-Bound
- **File**: `tests/test_monotonia_fast.py`
- **Risultato**: Ipotesi di monotonia **NON RISPETTATA** (40% di violazioni su 20 trial)
- **Implicazione**: Branch-and-Bound **non applicabile** su questo dataset
- **Cause**: ridondanza tra feature, interazioni non lineari, overfitting locale di KNN

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

## Sistema Multi-Esperto ed Ensemble

### Architettura (`src/ensemble/multi_expert_system.py`)
Sistema multi-esperto tradizionale basato su **decision profile** con aggregazione configurabile: `mean`, `weighted_mean`, `median`, `product`, `majority_vote`.

### Esperti Valutati
- **KNN**: su feature set da Lasso, Relief e SBS
- **Decision Tree**: su feature set Max-Min e Best First
- **Random Forest**: su full features e Information Gain top-25
- **Logistic Regression**: su Lasso top-25
- **AdaBoost** e **HistGradientBoosting**: su full features e Lasso top-25

### Hyperparameter Tuning (`src/feature_selection/Hyperparameter_Tuning.py`)
GridSearchCV con 3-fold CV su 2.000 campioni per classe. Migliori parametri trovati:

| Modello | Parametri Ottimali |
|---------|-------------------|
| Logistic Regression (lasso_top25) | C=0.3, class_weight=balanced |
| Random Forest (full) | n_estimators=200, max_depth=16, max_features=log2, min_samples_leaf=2 |
| HistGradientBoosting (full) | learning_rate=0.06, max_iter=160, max_leaf_nodes=15 |
| HistGradientBoosting (lasso_top25) | learning_rate=0.04, max_iter=160, max_leaf_nodes=15, l2=0.1 |
| AdaBoost (lasso_top25) | n_estimators=180, learning_rate=0.3, max_depth=3 |

### Risultati Finali MES

| Configurazione | F1-Micro | F1-Macro | Accuracy |
|----------------|----------|----------|----------|
| mes_top4_product | **0.5803** | 0.5800 | 0.5803 |
| mes_diverse4_product | 0.5803 | 0.5800 | 0.5803 |
| mes_top4_mean | 0.5800 | 0.5793 | 0.5800 |
| logistic_lasso_top25 (singolo) | 0.5793 | 0.5785 | 0.5793 |

**Miglior modello**: `mes_top4_product` — sistema multi-esperto con i 4 migliori esperti (logistic_lasso_top25, hist_gradient_boosting_full, hist_gradient_boosting_lasso_top25, random_forest_full) e regola `product` sul decision profile.

**Classification report del miglior MES**:

| Classe | Precision | Recall | F1 |
|--------|-----------|--------|----|
| 1 | 0.765 | 0.664 | 0.711 |
| 2 | 0.463 | 0.411 | 0.436 |
| 3 | 0.535 | 0.666 | 0.593 |

La classe 2 rimane la più difficile, spesso confusa con la classe 3.

---

## Analisi delle Feature Geografiche

Le variabili `geo_level_1_id`, `geo_level_2_id` e `geo_level_3_id` risultano le più informative in tutti i ranking (IG, Relief). Sono state identificate diverse strategie per sfruttarle meglio:

**Priorità alta**:
- Target/CatBoost Encoding out-of-fold dei geo_level_id
- Feature aggregate gerarchiche con smoothing (count edifici per area, probabilità danno per zona, entropia locale, gestione aree rare)

**Priorità media**:
- Collasso delle aree rare al livello geografico superiore + flag `rare_geo`
- Embedding denso dei tre livelli (autoencoder)

**Priorità bassa**:
- Geo3 Rollup embedding (più sperimentale, costo/beneficio peggiore)

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
│   │   ├── validation.py
│   │   └── outlier_detection/
│   │       ├── DBSCAN.py
│   │       └── outlier_k_comparison.py
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
│   │
│   └── ensemble/
│       └── multi_expert_system.py
│
├── experiments/
│   ├── evaluate_feature_selection.py
│   ├── evaluate_multi_expert.py
│   └── tune_multi_expert_hyperparameters.py
│
├── tests/
│   ├── test_best_first.py
│   ├── test_monotonia_fast.py
│   └── validate_bidirectional.py
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

Il menu interattivo guida attraverso: imputazione outlier, scelta strategia per `age`, OHE, standardizzazione e PCA opzionale.

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

# Test monotonia (prerequisito per branch-and-bound)
python tests/test_monotonia_fast.py --num-trials 20 --max-rows 5000
```

---

## Riepilogo Risultati

### Baseline (68 feature, KNN k=5)
- Accuracy: 0.592 | F1 Macro: 0.463 | Balanced Accuracy: 0.455

### Feature Selection (confronto su ~30.000 campioni bilanciati, K ottimale=21)

| Metodo | F1-Micro | Feature |
|--------|----------|---------|
| SBS | 0.545 | 30 |
| Best First Search | 0.542 | 17 |
| Relief | 0.539 | 15 |

### Sistema Multi-Esperto (5.000 campioni per classe, 15.000 totali)

| Configurazione | F1-Micro |
|----------------|----------|
| mes_top4_product | **0.580** |
| HistGradientBoosting (full, tuned) | 0.586 *(singolo esperto)* |
| Random Forest (full, tuned) | 0.575 *(singolo esperto)* |

### Feature Stabili in Tutte le Selezioni
Strutturali e materiali: `count_floors_pre_eq`, `count_families`, `foundation_type_r/i/w`, `ground_floor_type_v`, `has_superstructure_mud_mortar_stone`, `has_superstructure_cement_mortar_brick`, `has_superstructure_rc_engineered`, `roof_type_q/x`, `position_s/t`

---

## Decisioni Implementative Chiave

**Pattern Strategy per imputazione** — separa la logica di scelta algoritmo dal main; estendibile aggiungendo solo una nuova classe concreta.

**OneHotEncoder vs get_dummies()** — garantisce allineamento Train/Test e gestione categorie sconosciute senza data leakage.

**Esclusione Geographic IDs da PCA** — i geo_level_id alteravano la scala della varianza rendendo lo scree plot illeggibile; vengono riallegati dopo la trasformazione.

**k=3.0 per IQR** — validato sperimentalmente come ottimale tra k∈{1.5, 2.0, 2.5, 3.0, 4.0}; rimuove solo gli outlier genuini senza perdere dati validi.

**Branch-and-bound escluso** — ipotesi di monotonia violata nel 40% dei trial; i metodi euristici (SFS, SBS, Bidirectional, Best First) rimangono le scelte corrette.

---

## Documentazione Estesa

Per dettagli completi su ogni decisione, esperimento e iterazione consultare:
```
DocumentoDiBordo.txt
```
Diario cronologico che documenta scelte metodologiche, esperimenti comparativi, output di ogni fase e note sulle problematiche risolte.


