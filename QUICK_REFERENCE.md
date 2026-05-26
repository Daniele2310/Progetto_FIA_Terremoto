# 🚀 Quick Reference - Pipeline Completa

## Esecuzione Rapida

### 1. Esegui Pipeline Completa (Interattiva)
```bash
python main.py
```

**Ti chiederà**:
1. Outlier Detection: `1` (IQR) o `2` (DBSCAN)
2. Geo Embedding: `1` (statico) o `2` (neurale)
3. Hidden layers (se neurale): `1`, `2`, o `3`
4. Feature Selection: `1` (nessuna), `2` (Relief), `3` (Info Gain)
5. Modello: `1` (KNN), `2` (DT), `3` (RF), `4` (Multi-Expert)

**Output**: Report completo con:
- ✅ Tutte le scelte effettuate
- ✅ Metriche su validation set
- ✅ Feature selezionate
- ✅ Riepilogo finale

---

### 2. Testa L'Integrazione
```bash
python test_pipeline_integration.py
```

**Verifica**:
- ✅ Caricamento dati
- ✅ Geo-embedding (sample 5k righe)
- ✅ Feature selection (Relief)
- ✅ Model training (KNN)
- ✅ Tutto integrato correttamente

---

## 📊 Flusso Semplificato

```
INPUT: Scelte utente
  ↓
OUTLIER DETECTION (Fase 1)
  ↓
IMPUTATION (Fase 2)
  ↓
GEO-EMBEDDING (Fase 3) ← NUOVO: Implementato con GeoFeatureEngineer
  ↓
FEATURE SELECTION (Fase 3.5) ← NUOVO: Relief o Information Gain
  ↓
MODEL TRAINING (Fase 4) ← AGGIORNATO: Usa geo-features + FS
  ↓
RESULTS (Fase 5) ← AGGIORNATO: Mostra tutte le scelte
```

---

## 🎯 Cosa è Stato Fatto

### ✅ Geo-Embedding (Fase 3) - IMPLEMENTATO
- Utilizza `GeoFeatureEngineer` dal modulo preprocessing
- Metodo OOF (Out-Of-Fold) per evitare data leakage
- Aggiunge geo-feature ai dataset train e test
- Scegliibile: Embedding statico o Rete neurale

### ✅ Feature Selection (Fase 3.5) - NUOVO
- Menu per scegliere metodo
- Opzioni: Nessuna, Relief Ranking, Information Gain
- Seleziona top 30 feature (configurabile)
- Passa feature selezionate al modello

### ✅ Model Training (Fase 4) - INTEGRATO
- Accetta geo-features dai dati
- Accetta selected_features da Feature Selection
- Split train/validation: 80/20 stratificato
- Chiama train_knn(), train_randomforest(), train_decisiontree()
- Ritorna metriche su validation set

### ✅ Results (Fase 5) - COMPLETO
- Mostra tutte le scelte (outlier, imputation, geo, FS, modello)
- Mostra metriche dettagliate
- Mostra feature selezionate
- Layout ordinato con emoji per chiarezza

---

## 🔧 Parametri Configurabili

Se vuoi modificare i parametri delle funzioni helper, trovate:

**In `main.py`**:

```python
# Linea ~720: Geo embedding
train_values, test_values = applica_geo_embedding(
    train_values=train_values,
    test_values=test_values,
    tipo=tipo_geo_embedding,
    smoothing=20.0,           # ← Modifica qui
    rare_threshold=10,        # ← O qui
    n_splits=5                # ← O qui
)

# Linea ~735: Feature selection
selected_features = esegui_feature_selection(
    X_train=X_train_fs,
    y_train=y_train_fs,
    metodo=metodo_fs,
    top_k=50                  # ← Modifica qui per più/meno feature
)
```

---

## 📋 Struttura File Creati

```
Progetto_FIA_Terremoto/
├── main.py                              ← MODIFICATO (pipeline completa)
├── test_pipeline_integration.py         ← NUOVO (test)
├── PIPELINE_INTEGRATION_DOCS.md        ← NUOVO (doc completa)
├── README_TECHNICAL_INTEGRATION.md     ← NUOVO (sintesi tecnica)
├── QUICK_REFERENCE.md                  ← Questo file
└── ...
```

---

## 🧪 Esempi di Utilizzo

### Esempio 1: Baseline (Nessuna Feature Selection)
```
1. Outlier: IQR (1)
2. Geo: Embedding statico (1)
3. Feature Selection: None (1)  ← Usa tutte le feature
4. Modello: KNN (1)

Risultato: KNN con tutte le geo-feature
```

### Esempio 2: Con Feature Selection Aggressiva
```
1. Outlier: DBSCAN (2)
2. Geo: Rete neurale (2) → Hidden: 2 (2)
3. Feature Selection: Relief (2)  ← Top 30
4. Modello: Random Forest (3)

Risultato: RF con 30 feature selezionate via Relief
```

### Esempio 3: Information Gain
```
1. Outlier: IQR (1)
2. Geo: Embedding statico (1)
3. Feature Selection: Info Gain (3)  ← Entropia
4. Modello: Decision Tree (2)

Risultato: DT con feature selezionate via Information Gain
```

---

## 📈 Metriche Riportate

Alla fine della Fase 5 vedrai:

```
================================================================================
  RIEPILOGO SCELTE EFFETTUATE:

  🔹 Outlier Detection:      DBSCAN
  🔹 Imputazione:            KNN Strategy
  🔹 Geo Embedding:          Embedding statico
  🔹 Feature Selection:      RELIEF
  🔹 Modello:                K-Nearest Neighbors (KNN)
  🔹 Feature usate:          30

📊 RISULTATI MODELLO:

  Miglior configurazione iperparametri:
    - n_neighbors: 5
    - weights: uniform
    - metric: euclidean

  📈 Metriche su VALIDATION set:
    - F1 Micro:              0.7234
    - Accuracy:              0.7891

🎯 FEATURE SELECTION (30 feature selezionate):

    1. feature_x
    2. feature_y
    ...
    10. feature_z
    ... e 20 altri

================================================================================
```

---

## ⚡ Veloce Setup

Se vuoi usare i parametri di default senza menu interattivo:

**Script wrapper**:
```python
# quick_pipeline.py
import subprocess
import sys

# Simula input automatico
inputs = """1
1
2
2
1
"""

proc = subprocess.Popen(
    [sys.executable, 'main.py'],
    stdin=subprocess.PIPE,
    text=True
)
proc.communicate(input=inputs)
```

Esegui:
```bash
python quick_pipeline.py
```

---

## 🔒 Protezione Contro Data Leakage

La pipeline è progettata per evitare data leakage:

| Fase | Protezione |
|------|-----------|
| Outlier | Solo su TRAIN |
| Imputation | Fit su TRAIN, apply su TEST |
| Geo-Embedding | OOF fit su TRAIN, transform su TEST |
| Feature Selection | Ranking su TRAIN only |
| Model Training | Split 80/20 STRATIFICATO |

---

## 📝 Note Importanti

1. **Train/Validation Split**: 
   - 80% per training
   - 20% per validation (metriche finali)
   - Stratificato (mantiene distribuzione classi)

2. **Geo-Features**:
   - Aggiunte automaticamente ai dataset
   - Smoothing gerarchico (geo_level_1 → geo_level_2 → geo_level_3)
   - Anti-leakage tramite OOF

3. **Feature Selection**:
   - Relief: Basa su distanza ai vicini
   - Information Gain: Basa su entropia
   - Nessuna: Usa tutte le feature

4. **Modelli**:
   - KNN: Fast, memory-efficient
   - Decision Tree: Interpretable
   - Random Forest: Robust, ensemble

---

## 🐛 Troubleshooting

### Problema: "ModuleNotFoundError: No module named 'xxx'"
```bash
# Installa le dipendenze
pip install -r requirements.txt
```

### Problema: "KeyError: 'damage_grade'"
```
Assicurati che train_labels sia passato correttamente.
Il modulo internal aggiunge damage_grade a train_values automaticamente.
```

### Problema: "ValueError: too many values to unpack"
```
Verifica che esegui_modello() sia chiamato con 4 parametri:
✗ esegui_modello(choice, train_values, labels, test_values)
✓ esegui_modello(choice, train_values, labels, selected_features)
```

---

## 📞 Per Saperne Di Più

Leggi:
- `PIPELINE_INTEGRATION_DOCS.md` - Documentazione completa
- `README_TECHNICAL_INTEGRATION.md` - Dettagli tecnici
- `test_pipeline_integration.py` - Codice di test

---

**Status**: ✅ PIPELINE COMPLETATA E TESTATA
**Versione**: 1.0
**Data**: 2024
