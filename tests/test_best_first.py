"""
Test della implementazione Best First Search
Eseguito sul dataset preprocessato del progetto Terremoto
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import importlib.util


def carica_best_first_selector():
    """Carica BestFirstSelector dalla stessa cartella in cui si trova questo script."""
    # Visto che test e script sono nella stessa cartella, basta cercare "best_first.py" qui
    module_path = Path(__file__).resolve().parent / "best_first.py"

    if not module_path.exists():
        raise FileNotFoundError(f"Modulo Best First non trovato: {module_path}")

    spec = importlib.util.spec_from_file_location("best_first_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossibile caricare il modulo da: {module_path}")

    module = importlib.util.module_from_spec(spec)

    # Registra il modulo nel sistema per far funzionare le dataclass
    import sys
    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    if not hasattr(module, "BestFirstSelector"):
        raise ImportError("Classe 'BestFirstSelector' non trovata nel modulo.")

    return module.BestFirstSelector


def main():
    print("\n" + "=" * 80)
    print("TEST BEST FIRST SEARCH - TERREMOTO NEPAL 2015")
    print("=" * 80)

    # Caricamento dinamico della classe
    BestFirstSelector = carica_best_first_selector()

    # ==========================================
    # 1. Caricamento Dataset
    # ==========================================
    print("\n[1/5] Caricamento dataset preprocessato...")

    # Usa Path(__file__).resolve().parent per avere percorsi sicuri ovunque esegui lo script
    project_root = Path(__file__).resolve().parents[2]
    data_preprocessed_path = project_root / "DataPreprocessed" / "processed" / "train_features_labels_preprocessed.csv"

    if not data_preprocessed_path.exists():
        print(f"❌ File non trovato: {data_preprocessed_path}")
        print("Esegui prima main.py dal modulo src.preprocessing")
        return

    df = pd.read_csv(data_preprocessed_path)
    print(f"✓ Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne")

    # ==========================================
    # 2. Preparazione X e y
    # ==========================================
    print("\n[2/5] Preparazione X (features) e y (target)...")

    # Rimuovi building_id e damage_grade
    X = df.drop(columns=['building_id', 'damage_grade'])
    y = df['damage_grade'].values

    print(f"✓ X shape: {X.shape}")
    print(f"✓ y shape: {y.shape}")
    print(f"✓ Feature names: {list(X.columns[:5])}... (prime 5 di {len(X.columns)})")
    print(f"✓ Target classes: {np.unique(y)}")

    # ==========================================
    # 3. Istanziazione Best First Selector
    # ==========================================
    print("\n[3/5] Configurazione Best First Search...")

    selector = BestFirstSelector(
        patience=5,  # Fermarsi dopo 5 iterazioni senza miglioramento
        random_state=42
    )
    print(f"✓ Patience: {selector.patience}")
    print(f"✓ Random state: {selector.random_state}")

    # ==========================================
    # 4. Esecuzione Best First
    # ==========================================
    print("\n[4/5] Esecuzione Best First Search...")
    print("(Questo potrebbe richiedere qualche minuto...)\n")

    result = selector.select(
        x=X,
        y=y,
        max_rows=15000  # Usa max 15k righe per velocità
    )

    # ==========================================
    # 5. Analisi Risultati
    # ==========================================
    print("\n[5/5] Analisi Risultati\n")

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

    output_dir = project_root / "src.feature_selection" / "Best First Results"
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