# Dati del Progetto

## Data/raw/
Contiene i dati grezzi del progetto:

### File Necessari:
- `train_values.csv` - Features di training (260.601 edifici × 39 colonne)
- `train_labels.csv` - Etichette di danno corrispondenti (260.601 edifici)
- `test_values.csv` - Features di test (~86.868 edifici)
- `submission_format.csv` - Formato di submission

**Note:** 
- I file CSV sono troppo grandi per essere tracciati in Git
- Scaricarli dal dataset Kaggle: [Earthquake Damage Prediction - Nepal 2015](https://www.kaggle.com/c/earthquake-damage-prediction)
- Una volta scaricati, inserirli in questa cartella

## Data/preprocessed/
Contiene i dati preprocessati generati da `main.py`:

- `train_values_preprocessed.csv` - Training preprocessato
- `test_values_preprocessed.csv` - Test preprocessato  
- `train_features_labels_preprocessed.csv` - Training + labels uniti

**Generazione:** Questi file vengono generati automaticamente eseguendo:
```bash
python main.py
```
