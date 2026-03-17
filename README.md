# Progetto FIA - Terremoto (Richter's Predictor)

## 📋 Descrizione Progetto

Questo progetto partecipa alla competizione **DrivenData: Richter's Predictor - Modeling Earthquake Damage**.

L'obiettivo è **predire il livello di danno agli edifici** causato dal **terremoto del 2015 Gorkha in Nepal** sulla base di:
- **Aspetti della posizione** dell'edificio
- **Caratteristiche costruttive** dell'edificio

### Dataset
Il dataset è stato raccolto attraverso sondaggi di:
- **Kathmandu Living Labs**
- **Central Bureau of Statistics** (Nepal)

È uno dei più grandi dataset post-disastro mai raccolti, contenente informazioni preziose su impatti sismici, condizioni abitative e statistiche socio-economico-demografiche.

## 📊 Dati

- `train_values.csv` - Features di training (variabili indipendenti)
- `train_labels.csv` - Etichette di training (livello di danno)
- `test_values.csv` - Features di test (da predire)
- `submission_format.csv` - Formato richiesto per le submissions

## 🔗 Link Competizione

- **Competizione**: https://www.drivendata.org/competitions/57/nepal-earthquake/
- **Partecipanti**: 8,653 iscritti

## 🚀 Quick Start

1. Attivare l'ambiente virtuale:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. Installare dipendenze (se necessario):
   ```bash
   pip install -r requirements.txt
   ```

3. Eseguire il modello e generare le predizioni

## 📈 Struttura Progetto

```
├── train_values.csv          # Dati di training
├── train_labels.csv          # Etichette di training
├── test_values.csv           # Dati di test
├── submission_format.csv     # Formato submission
├── venv/                     # Ambiente virtuale
└── [scripts Python]          # Modelli e analisi
```
