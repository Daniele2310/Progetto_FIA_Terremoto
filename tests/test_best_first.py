"""
Test della implementazione Best First Search
Eseguito sul dataset preprocessato del progetto Terremoto
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_selection.subset_selection.best_first import BestFirstSelector
from src.preprocessing.data_selection import get_stratified_sample


def carica_best_first_selector():
    """Restituisce BestFirstSelector dal modulo canonico del progetto."""
    return BestFirstSelector


def main():
    print("\n" + "=" * 80)
    print("TEST BEST FIRST SEARCH - TERREMOTO NEPAL 2015")
    print("=" * 80)

    # Caricamento della classe dal modulo canonico del progetto
    BestFirstSelector = carica_best_first_selector()

    # ==========================================
    # 1. Caricamento Dataset
    # ==========================================
    print("\n[1/6] Caricamento dataset preprocessato...")

    # Usa la root del progetto per avere percorsi sicuri ovunque esegui lo script
    project_root = PROJECT_ROOT
    data_preprocessed_path = project_root / "Data" / "preprocessed" / "train_features_labels_preprocessed.csv"

    if not data_preprocessed_path.exists():
        print(f"❌ File non trovato: {data_preprocessed_path}")
        print("Esegui prima main.py dal modulo src.preprocessing")
        return

    df = pd.read_csv(data_preprocessed_path)
    print(f"✓ Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne")

    # ==========================================
    # 2. Campionamento Non Polarizzato (Stratificato)
    # ==========================================
    print("\n[2/6] Campionamento non polarizzato (stratificato)...")
    
    df_sampled = get_stratified_sample(
        df, 
        target_col='damage_grade',
        n_samples=25000,  # Campione stratificato di 25k righe
        random_state=42
    )
    print(f"✓ Dataset campionato: {df_sampled.shape[0]} righe")
    print(f"✓ Distribuzione classi:\n{df_sampled['damage_grade'].value_counts().sort_index().to_string()}")

    # ==========================================
    # 3. Split Train / Validation (80/20)
    # ==========================================
    print("\n[3/6] Split train / validation (80/20)...")
    
    X = df_sampled.drop(columns=['building_id', 'damage_grade'])
    y = df_sampled['damage_grade'].astype(int)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"✓ Train: {X_train.shape[0]} righe")
    print(f"✓ Validation: {X_val.shape[0]} righe")
    print(f"✓ Feature totali: {X_train.shape[1]}")
    print(f"✓ Target classes: {np.unique(y_train)}")

    # ==========================================
    # 4. Istanziazione Best First Selector
    # ==========================================
    print("\n[4/6] Configurazione Best First Search...")

    selector = BestFirstSelector(
        patience=5,  # Fermarsi dopo 5 iterazioni senza miglioramento
        random_state=42
    )
    print(f"✓ Patience: {selector.patience}")
    print(f"✓ Random state: {selector.random_state}")

    # ==========================================
    # 5. Selezione Feature su TRAIN Set
    # ==========================================
    print("\n[5/6] Esecuzione Best First Search su TRAIN set...")
    print("(Questo potrebbe richiedere qualche minuto...)\n")

    result = selector.select(
        x=X_train,
        y=y_train.to_numpy(),
        max_rows=None  # Usa tutto il train set per la selezione
    )

    # ==========================================
    # 6. Valutazione su VALIDATION Set
    # ==========================================
    print("\n[6/6] Valutazione delle feature selezionate su VALIDATION set...\n")

    selected_features_list = result['selected_features']['selected_feature'].tolist()
    
    if selected_features_list:
        # Valuta il modello sulle feature selezionate
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import classification_report
        
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train[selected_features_list], y_train)
        y_pred = model.predict(X_val[selected_features_list])
        
        accuracy_val = accuracy_score(y_val, y_pred)
        f1_val = f1_score(y_val, y_pred, average='micro')
        
        print(f"✓ Accuracy su Validation: {accuracy_val:.4f}")
        print(f"✓ F1-Micro su Validation: {f1_val:.4f}")
    else:
        print("⚠ Nessuna feature selezionata!")
        accuracy_val = 0.0
        f1_val = 0.0

    # ==========================================
    # 7. Analisi Risultati
    # ==========================================
    print("\n" + "=" * 80)
    print("RISULTATI FINALI")
    print("=" * 80)
    print("\n📋 FLUSSO DI VALIDATION CORRETTO:")
    print(f"   1. Dataset campionato (stratificato): {df_sampled.shape[0]} righe")
    print(f"   2. Split train: {X_train.shape[0]} righe ({100*X_train.shape[0]/(X_train.shape[0]+X_val.shape[0]):.0f}%)")
    print(f"   3. Split validation: {X_val.shape[0]} righe ({100*X_val.shape[0]/(X_train.shape[0]+X_val.shape[0]):.0f}%)")
    print(f"   4. Selezione feature: ADDESTRATA su train set")
    print(f"   5. Valutazione modello: TESTATA su validation set")
    print(f"   6. Accuracy su validation: {accuracy_val:.4f}")
    print(f"   7. F1-Micro su validation: {f1_val:.4f}")
    print("\n" + "=" * 80)

    # Summary
    summary = result['summary']
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key:.<40} {value:.4f}" if 'score' in key or 'elapsed' in key else f"  {key:.<40} {value}")
        else:
            print(f"  {key:.<40} {value}")

    # Selected Features
    selected_features_df = result['selected_features']
    print("\n" + "=" * 80)
    print("FEATURE SELEZIONATE")
    print("=" * 80)
    selected_list = selected_features_df['selected_feature'].tolist()
    print(f"Numero: {len(selected_list)}")
    for i, feat in enumerate(selected_list, 1):
        print(f"  {i:2d}. {feat}")

    # History - Top 10 step
    history_df = result['history'].copy()
    print("\n" + "=" * 80)
    print("STORIA DELL'ESPANSIONE (Primi 10 step)")
    print("=" * 80)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)

    print(history_df.head(10).to_string(index=False))

    # ==========================================
    # Statistiche finali
    # ==========================================
    print("\n" + "=" * 80)
    print("STATISTICHE FINALI")
    print("=" * 80)

    best_step = history_df.loc[history_df['global_best_score'].idxmax()]
    print(f"Step migliore: #{best_step['step']}")
    print(f"Score migliore raggiunto: {best_step['global_best_score']:.4f}")

    # Tronca la stampa se le feature sono troppe
    expanded_subset_str = best_step['expanded_subset']
    if len(expanded_subset_str) > 60:
        expanded_subset_str = expanded_subset_str[:60] + "..."
    print(f"Feature nel subset migliore: {expanded_subset_str}")

    print(f"\nModelli valutati: {summary['evaluated_models']}")
    print(f"Tempo totale: {summary['elapsed_seconds']:.2f} secondi")
    print(f"Motivo stop: {summary['stop_reason']}")

    # ==========================================
    # Salva i risultati
    # ==========================================
    print("\n" + "=" * 80)
    print("SALVATAGGIO RISULTATI")
    print("=" * 80)

    output_dir = project_root / "tests" / "outputs" / "best_first_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Salva summary
    summary_df = pd.DataFrame([summary])
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary salvato: {summary_path}")

    # Salva history
    history_path = output_dir / "history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"✓ History salvato: {history_path}")

    # Salva selected features
    selected_path = output_dir / "selected_features.csv"
    selected_features_df.to_csv(selected_path, index=False)
    print(f"✓ Selected features salvato: {selected_path}")

    # ==========================================
    # Confronto
    # ==========================================
    print("\n" + "=" * 80)
    print("CONFRONTO RIDUZIONE FEATURES")
    print("=" * 80)
    print(f"Features iniziali:  {summary['n_features_initial']}")
    print(f"Features finali:    {summary['n_features_final']}")
    reduction_pct = (1 - summary['n_features_final'] / summary['n_features_initial']) * 100
    print(f"Riduzione:          {reduction_pct:.1f}%")
    print(f"Score mantenuto:    {summary['best_score_final']:.4f}")

    print("\n" + "=" * 80)
    print("✅ TEST COMPLETATO CON SUCCESSO!")
    print("=" * 80)


if __name__ == "__main__":
    main()
