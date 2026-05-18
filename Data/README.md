# Dati del Progetto

## Data/raw/
Contiene i dati grezzi del progetto:
- `train_values.csv` - Features di training (260.601 edifici × 39 colonne)
- `train_labels.csv` - Etichette di danno corrispondenti
- `test_values.csv` - Features di test (~86.868 edifici)
- `submission_format.csv` - Formato di submission

**Nota:** I file CSV sono troppo grandi per essere tracciati in Git. Scaricarli dal dataset Kaggle "Earthquake Damage Prediction".

## Data/preprocessed/
Contiene i dati preprocessati generati da `main.py`:
- `train_values_preprocessed.csv` - Training preprocessato
- `test_values_preprocessed.csv` - Test preprocessato
- `train_features_labels_preprocessed.csv` - Training + labels uniti

Questi file vengono generati automaticamente dal pipeline di preprocessing.
