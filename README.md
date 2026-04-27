# Progetto FIA - Terremoto (Richter's Predictor)

Predizione del livello di danno agli edifici colpiti dal terremoto del 2015 Gorkha in Nepal.

**Competizione**: [DrivenData: Richter's Predictor](https://www.drivendata.org/competitions/57/nepal-earthquake/)  
**Partecipanti**: 8,653 iscritti | **Difficoltà**: Intermediate Practice

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
- `has_superstructure_adobe_mud` - Adobe/Fango
- `has_superstructure_mud_mortar_stone` - Pietra con mortaio di fango
- `has_superstructure_stone_flag` - Pietra
- `has_superstructure_cement_mortar_stone` - Pietra con mortaio di cemento
- `has_superstructure_mud_mortar_brick` - Mattone con mortaio di fango
- `has_superstructure_cement_mortar_brick` - Mattone con mortaio di cemento
- `has_superstructure_timber` - Legno
- `has_superstructure_bamboo` - Bambù
- `has_superstructure_rc_engineered` - Cemento armato (engineered)
- `has_superstructure_rc_non_engineered` - Cemento armato (non-engineered)
- `has_superstructure_other` - Altri materiali

### Proprietà e Utilizzo (8)
| Feature | Tipo | Descrizione |
|---------|------|-------------|
| `legal_ownership_status` | cat | Status di proprietà legale (a, r, v, w) |
| `count_families` | int | Numero di famiglie residenti |
| `has_secondary_use` | bin | Utilizzo secondario |
| **Secondary Uses** | bin | Agricoltura, Hotel, Affitti, Istituzione, Scuola, Industria, Centro sanitario, Ufficio governativo, Stazione di polizia, Altro |

---

## Preprocessing dei Dati

### Moduli Implementati

#### 1. **Data Cleaning** (`DataPreprocessing/data_cleaning.py`)
- Verifica integrità dataset
- Rimozione righe duplicate
- Validazione range feature numeriche
- Conversione tipi dati

#### 2. **Missing Values** (`DataPreprocessing/missingValues.py`)
Gestione NaN con strategie di imputazione per la feature `age`:

**Outlier Handling**:
- Valori `age` nel range [250, 995] → convertiti in NaN (valori sentinel)

**Strategie di Imputazione** (selezionabili via menu):
1. **Univariata - Media**: mediana globale del training set
2. **Univariata - Mediana**: mediana globale del training set
3. **Multivariata - Regressione Lineare**: mediana per gruppi geografici in gerarchia
4. **KNN Predictor**: K-nearest neighbors per imputazione
5. **Valutazione Automatica**: confronto rapido con KNN e selezione del metodo migliore

**Motivo della scelta multivariata** (attualmente implementata):
- Mantiene informazione geografica locale
- Riduce l'appiattimento tipico della mediana globale
- Leggero miglioramento su F1 macro (+0.0006) e balanced accuracy (+0.0004)

#### 3. **Codifica Categorica** (`DataPreprocessing/data_cleaning.py`)
- **OneHotEncoder** di scikit-learn con `handle_unknown='ignore'`
- Vantaggi rispetto a `get_dummies()`:
  - Allineamento automatico Train/Test (stesse colonne)
  - Gestione di categorie mai viste nel test set
  - Prevenzione del data leakage

#### 4. **Standardizzazione**
- **StandardScaler** per feature numeriche
- Essenziale per modelli sensibili alla scala (KNN, Lasso, etc.)

#### 5. **ASCII Cleaning** (`DataPreprocessing/puliziaASCII.py`)
- Normalizzazione encoding caratteri categorici
- Risoluzione problemi con charset non UTF-8

---

## Feature Selection

### Metodi Implementati

Sono stati implementati **7 metodi di feature selection** con approcci diversi:

#### 1. **Sequential Forward Selection (SFS)**
- **File**: `Feature Selection/subset selection/sfs.py`
- **Approccio**: Forward. Parte da 0 feature, aggiunge iterativamente la migliore
- **Metrica**: Accuracy o F1-micro
- **Estimator**: LogisticRegression o KNeighborsClassifier
- **Risultati**: Selezionate 7 feature
- **Output**: `sfs_history.csv`, `sfs_selected_features.csv`, `sfs_score_history.png`

#### 2. **Sequential Backward Selection (SBS)**
- **File**: `Feature Selection/subset selection/sbs_subset_selection.py`
- **Approccio**: Backward. Parte da tutte le feature, rimuove iterativamente la peggiore
- **Risultati**: Riduzione 68 → 28 feature; accuracy baseline 0.533 → 0.622 (+8.8%)
- **Costo**: O(p²), 1.969 modelli valutati in 29.26s

#### 3. **Bidirectional Stepwise Selection**
- **File**: `Feature Selection/subset selection/bidirectional_subset_selection.py`
- **Approccio**: Alternanza di step forward e backward
- **Risultati**: Riduzione 68 → 9 feature; accuracy 0.533 → 0.655 (+12.2%)
- **Stop**: Quando un intero ciclo forward+backward non produce miglioramenti
- **Output**: `bidirectional_history.csv`, `bidirectional_selected_features.csv`, `bidirectional_score_history.png`

#### 4. **Best First Search**
- **File**: `Feature Selection/subset selection/best_first.py`
- **Approccio**: Priority queue, esplorazione greedy guidata da uno score
- **Pazienza**: k=5 (stop se 5 espansioni consecutive non migliorano il best score)
- **Risultati**: **82.6% di riduzione dimensionale** (69 → 12 feature)
  - **Accuracy: 0.6913 (~69.1%)**
  - Baseline (full features): 57%
- **Costo computazionale**: Ragionevole rispetto alla ricerca esaustiva
- **Validazione**: Script test dedotto `test_best_first.py`

#### 5. **Max-Min Subset Selection**
- **File**: `Feature Selection/subset selection/max_min_subset_selection.py`
- **Approccio**: Greedy, seleziona feature per massimo compromesso rilevanza-ridondanza
- **Formula**: `score(f) = rilevanza(f, target) - ridondanza(f, selected_set)`
- **Risultati**: Selezionate 6 feature con stop su score negativo

#### 6. **Embedded Lasso Regression**
- **File**: `Feature Selection/Embedded/lasso_feature_selection.py`
- **Approccio**: Feature selection durante l'addestramento (regolarizzazione L1)
- **Alpha**: Selezionabile via menu (LassoCV automatico o valore fisso)
- **Parametri**: `alpha=0.002` è un buon compromesso
- **Risultati con α=0.002**: 68 → 49 feature, mantenimento prestazioni baseline
- **Output**: `lasso_selected_features.csv`, `lasso_all_coefficients.csv`

#### 7. **PCA - Analisi delle Componenti Principali**
- **File**: `Feature Selection/feature ranking/PCA.py`
- **Approccio**: Riduzione dimensionale non supervisionata
- **Feature Escluse**: `building_id`, `geo_level_*_id`, `damage_grade`
- **Output**: 
  - `scree_plot.png` - per scelta manuale del gomito
  - `pca_variance_summary.csv` - varianza per componente
  - `train_values_preprocessed.csv` - dataset con componenti PCA

### Ranking e Importanza

#### Information Gain (Entropia)
- **File**: `Feature Selection/feature ranking/uncertainty_information_gain_ranking.py`
- **Top Features**: `geo_level_3_id` (IG=0.482), `geo_level_2_id` (IG=0.346), `geo_level_1_id` (IG=0.190)
- **Meno Informative**: `has_secondary_use_*police` (IG≈0)

#### RELIEF (Relevance)
- **File**: `Feature Selection/feature ranking/relief_ranking.py`
- **Approccio**: Misura locale per feature relevance
- **Top Features**: `geo_level_3_id`, `has_superstructure_cement_mortar_stone`, `geo_level_2_id`

#### Pearson Correlation
- **File**: `Feature Selection/feature ranking/pairwise_correlation_ranking.py`
- **Top Correlate con Target**: `foundation_type_r` (0.343), `ground_floor_type_v` (0.319)

### Test Monotonia
- **File**: `Feature Selection/test_monotonia_fast.py`
- **Risultato**: Ipotesi di monotonia **NON RISPETTATA** (40% violazioni)
- **Implicazione**: **Branch-and-bound NOT applicabile** su questo dataset
- **Motivi**: Ridondanza feature, interazioni non-lineari, overfitting

---

## Struttura del Progetto

```
Progetto_FIA_Terremoto/
│
├── README.md                          # Questo file
├── DocumentoDiBordo.txt               # Diario di lavoro dettagliato
├── main.py                            # Script principale di preprocessing
│
├── Data/
│   ├── train_values.csv               # Dataset training (features)
│   ├── train_labels.csv               # Etichette danno
│   ├── test_values.csv                # Dataset test
│   ├── submission_format.csv          # Template submission
│   └── Puliti/                        # Dataset puliti intermediari
│
├── DataPreprocessed/
│   ├── train_values_preprocessed.csv               # Features preprocessate
│   ├── test_values_preprocessed.csv                # Test preprocessato
│   ├── train_features_labels_preprocessed.csv      # Train + labels preprocessati
│   ├── pca_variance_summary.csv                    # Varianza componenti PCA
│   ├── pca_loadings.csv                            # Loadings PCA
│   └── scree_plot.png                              # Grafico scree plot
│
├── DataPreprocessing/
│   ├── __init__.py
│   ├── data_cleaning.py               # Pulizia dati e codifica
│   ├── imputation_strategies.py       # Pattern Strategy per imputazione
│   ├── missingValues.py               # Gestione valori mancanti
│   ├── puliziaASCII.py                # Cleaning encoding ASCII
│   └── validation.py                  # Validazione consistenza
│
├── Feature Selection/
│   ├── test_monotonia_fast.py          # Test ipotesi monotonia
│   │
│   ├── feature ranking/
│   │   ├── uncertainty_information_gain_ranking.py
│   │   ├── relief_ranking.py
│   │   ├── pairwise_correlation_ranking.py
│   │   ├── PCA.py
│   │   └── outputs/
│   │       ├── pairwise_ranking.csv
│   │       ├── uncertainty_information_gain_ranking.csv
│   │       ├── relief_ranking.csv
│   │       └── ...
│   │
│   ├── subset selection/
│   │   ├── sfs.py                                # Sequential Forward Selection
│   │   ├── sbs_subset_selection.py              # Sequential Backward Selection
│   │   ├── bidirectional_subset_selection.py    # Bidirectional Stepwise
│   │   ├── best_first.py                        # Best First Search
│   │   ├── max_min_subset_selection.py          # Max-Min Strategy
│   │   ├── test_best_first.py                   # Validazione Best First
│   │   ├── validate_bidirectional.py            # Validazione Bidirectional
│   │   └── outputs/
│   │       ├── sfs_history.csv
│   │       ├── sbs_selected_features.csv
│   │       ├── bidirectional_summary.json
│   │       ├── best_first_summary.json
│   │       └── ...
│   │
│   ├── Embedded/
│   │   ├── lasso_feature_selection.py
│   │   └── outputs/
│   │       ├── lasso_selected_features.csv
│   │       ├── lasso_all_coefficients.csv
│   │       └── lasso_summary.json
│   │
│   └── outputs/
│
├── experiments/                       # Modelli di experiment
│
├── venv/                              # Ambiente virtuale Python
│
└── requirements.txt                   # Dipendenze Python
```

---

## Quick Start

### Setup Iniziale

```powershell
# 1. Attivare ambiente virtuale
.\venv\Scripts\Activate.ps1

# 2. Installare dipendenze (se necessario)
pip install -r requirements.txt
```

### Esecuzione Preprocessing Completo

```bash
# Run il main menu interattivo
python main.py
```

**Opzioni disponibili nel menu**:
1. Imputazione di feature numeriche con outlier
2. Selezione metodo imputazione `age`
3. Codifica OneHot delle categoriche
4. Standardizzazione features
5. Pipeline PCA

### Feature Selection Standalone

```bash
# Sequential Forward Selection
python "Feature Selection\subset selection\sfs.py" --estimator knn --scoring f1_micro

# Sequential Backward Selection
python "Feature Selection\subset selection\sbs_subset_selection.py" --max-features 20

# Bidirectional Stepwise
python "Feature Selection\subset selection\bidirectional_subset_selection.py"

# Best First Search
python "Feature Selection\subset selection\best_first.py" --patience 5

# Lasso Embedded
python "Feature Selection\Embedded\lasso_feature_selection.py" --alpha 0.002

# Max-Min Strategy
python "Feature Selection\subset selection\max_min_subset_selection.py" --max-features 15

# PCA
python "Feature Selection\feature ranking\PCA.py"
```

### Test Monotonia

```bash
python "Feature Selection\test_monotonia_fast.py" --num-trials 20 --max-rows 5000
```

---

## Risultati Principali

### Baseline
- **Features**: 68 (dopo one-hot encoding)
- **Accuracy**: 0.5921
- **F1 Macro**: 0.4632
- **Balanced Accuracy**: 0.4552

### Best First Search
- **Feature ridotte**: 69 → 12 feature (-82.6%)
- **Accuracy**: 0.6913 (+16.8%)
- **Messaggio**: Riduzione significativa con mantenimento/miglioramento prestazioni

### Bidirectional Selection
- **Feature ridotte**: 68 → 9 feature
- **Accuracy**: 0.6550 (+12.2%)
- **Modelli valutati**: 688 in 17.61s

### SBS (Sequential Backward)
- **Feature ridotte**: 68 → 28 feature
- **Accuracy**: 0.6217 (+8.8%)
- **Modelli valutati**: 1.969 in 29.26s

### Lasso (α=0.002)
- **Feature ridotte**: 68 → 49 feature
- **Accuracy**: ~0.59 (stabilità baseline)
- **Vantaggio**: Embedded, veloce, interpretabile

### Feature Stabili (presenti in tutte le selezioni)
- Caratteristiche geografiche: `geo_level_*_id`
- Strutturali: `count_floors_pre_eq`, `count_families`
- Materiali: Indicatori di superstructure (adobe, stone, brick, timber, bamboo, RC)
- Posizione e tipo edificio

---

## Decisioni Implementative Chiave

### 1. Pattern Strategy per Imputazione
- **File**: `DataPreprocessing/imputation_strategies.py`
- **Motivo**: Separare logica di selezione algoritmo dalla pipeline
- **Vantaggi**: Estendibilità, chiarezza, manutenibilità

### 2. OneHotEncoder vs get_dummies()
- **Motivo**: Allineamento automatico Train/Test, gestione categorie sconosciute
- **Implementazione**: `handle_unknown='ignore'`

### 3. Esclusione Geographic IDs da PCA
- **Motivo**: Alteravano scala varianza, rendevano scree plot ininterpretabile
- **Soluzione**: Esclusione dal fit, riattacco post-PCA

### 4. Test Monotonia per Branch-and-Bound
- **Ritratto**: Ipotesi violata 40% dei trial
- **Conclusione**: Branch-and-bound non applicabile; mantenere metodi euristici

---

## Prossimi Passi

1. **Modellazione Avanzata**: Usare subset di feature migliore (Best First 12 feat.) per addestrare modelli complessi (XGBoost, LightGBM, Ensemble)
2. **Hyperparameter Tuning**: GridSearch/RandomSearch sui modelli scelti
3. **Ensemble Methods**: Combinare predizioni di modelli diversi
4. **Ordinality Constraint**: Exploitare natura ordinale del target (damage_grade: 1 < 2 < 3)
5. **Stacked Generalization**: Meta-modello su predizioni di base learners

---

## Documentazione Estesa

Per dettagli completi su ogni decisione, esperimento e iterazione, consultare:
```
DocumentoDiBordo.txt
```

Diario cronologico che documenta:
- Scelte metodologiche e motivazioni
- Esperimenti comparativi
- Output e risultati di ogni fase
- Note problematiche e risoluzioni

---


