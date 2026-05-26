"""
Test di integrazione per la pipeline completata.

Verifica:
1. Caricamento dati
2. Applicazione geo-embedding
3. Feature selection
4. Addestramento modello
5. Visualizzazione risultati
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.clean_ascii import PuliziaASCII, COLONNE_CATEGORICHE
from src.preprocessing.geo_features import GeoFeatureEngineer
from src.feature_selection.feature_ranking.relief_ranking import ReliefRanker
from src.Modelli.knn import train_knn

def test_step(step_name: str):
    """Decorator per tracciare i test step"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"\n{'='*80}")
            print(f"  TEST: {step_name}")
            print(f"{'='*80}")
            try:
                result = func(*args, **kwargs)
                print(f"  ✅ PASSED\n")
                return result
            except Exception as e:
                print(f"  ❌ FAILED: {e}\n")
                import traceback
                traceback.print_exc()
                return None
        return wrapper
    return decorator

@test_step("Caricamento dati")
def test_load_data():
    pulizia = PuliziaASCII()
    train_values, train_labels, test_values = pulizia.processa(
        colonne_categoriche=COLONNE_CATEGORICHE
    )
    
    print(f"  Train shape: {train_values.shape}")
    print(f"  Labels shape: {train_labels.shape}")
    print(f"  Test shape: {test_values.shape}")
    print(f"  Colonne: {train_values.columns.tolist()[:5]}...")
    
    # Assicura che damage_grade sia in train_values per geo embedding
    if "damage_grade" not in train_values.columns:
        print("  Aggiunta damage_grade a train_values da train_labels...")
        if isinstance(train_labels, pd.Series):
            train_values["damage_grade"] = train_labels.values
        else:
            train_values["damage_grade"] = train_labels.iloc[:, 0].values
    
    return train_values, train_labels, test_values

@test_step("Geo-Embedding")
def test_geo_embedding(train_values, test_values):
    print("  Applicazione geo-embedding...")
    
    # Utilizza campione di 20000 righe per velocità (maggiori dimensioni = più elementi per classe)
    train_sample = train_values.iloc[:20000].copy()
    test_sample = test_values.iloc[:5000].copy()
    
    # Usa n_splits=5 (richiesto da GeoFeatureEngineer)
    geo_engineer = GeoFeatureEngineer(
        geo_columns=("geo_level_1_id", "geo_level_2_id", "geo_level_3_id"),
        target_col="damage_grade",
        smoothing=20.0,
        rare_threshold=10,
        n_splits=5,
        random_state=42,
        append_original=True,
    )
    
    print(f"  Sample train: {train_sample.shape}")
    print(f"  Sample test: {test_sample.shape}")
    
    # Nel test usiamo fit_transform (veloce)
    # Nel main.py usiamo fit_transform_oof (anti-leakage con dataset completo)
    train_with_geo = geo_engineer.fit_transform(
        train_sample,
        train_sample["damage_grade"]
    )
    test_with_geo = geo_engineer.transform(test_sample)
    
    print(f"  Train con geo: {train_with_geo.shape}")
    print(f"  Test con geo: {test_with_geo.shape}")
    print(f"  Nuove feature: {train_with_geo.shape[1] - train_sample.shape[1]}")
    
    return train_with_geo, test_with_geo

@test_step("Feature Selection")
def test_feature_selection(train_with_geo):
    print("  Feature selection con Relief Ranking...")
    
    # Estrai X e y
    X = train_with_geo.drop(columns=["building_id", "damage_grade"], errors="ignore")
    y = train_with_geo["damage_grade"]
    
    print(f"  Feature disponibili: {X.shape[1]}")
    
    ranker = ReliefRanker(n_neighbors=5, n_iterations=100, random_state=42)
    result = ranker.rank(X, label_column=y)
    
    # Estrai ranking
    if "relief_ranking" in result:
        ranking_df = result["relief_ranking"]
    else:
        ranking_df = result.get(list(result.keys())[0])
    
    print(f"  Ranking type: {type(ranking_df)}")
    print(f"  Ranking shape: {ranking_df.shape}")
    
    top_k = 30
    selected_features = ranking_df["feature"].head(top_k).tolist()
    
    print(f"  Feature selezionate: {len(selected_features)}")
    print(f"  Prime 10: {selected_features[:10]}")
    
    return selected_features

@test_step("Model Training")
def test_model_training(train_with_geo, selected_features):
    print("  Addestramento KNN...")
    
    # Preparazione dati
    X = train_with_geo[selected_features].copy()
    y = train_with_geo["damage_grade"].copy()
    
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Classi: {sorted(y.unique())}")
    
    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Train: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    
    # Train
    result = train_knn(X_train, y_train, X_val, y_val, verbose=False)
    
    if result:
        print(f"  Model trained: {result.get('model')}")
        if "metrics" in result:
            print(f"  Metriche: {result['metrics']}")
        print(f"  Best params: {result.get('best_params', {})}")
    
    return result

def main():
    print("\n" + "="*80)
    print("  TEST INTEGRAZIONE PIPELINE COMPLETATA")
    print("="*80)
    
    # Step 1: Load data
    data = test_load_data()
    if data is None:
        print("Interruzione: caricamento dati fallito")
        return
    
    train_values, train_labels, test_values = data
    
    # Step 2: Geo embedding
    geo_data = test_geo_embedding(train_values, test_values)
    if geo_data is None:
        print("Interruzione: geo embedding fallito")
        return
    
    train_with_geo, test_with_geo = geo_data
    
    # Step 3: Feature selection
    selected_features = test_feature_selection(train_with_geo)
    if selected_features is None:
        print("Interruzione: feature selection fallito")
        return
    
    # Step 4: Model training
    model_result = test_model_training(train_with_geo, selected_features)
    if model_result is None:
        print("Interruzione: model training fallito")
        return
    
    # Resoconto finale
    print("\n" + "="*80)
    print("  RESOCONTO FINALE")
    print("="*80)
    print(f"  ✅ Caricamento dati: SUCCESS")
    print(f"  ✅ Geo embedding: SUCCESS (shape {train_with_geo.shape})")
    print(f"  ✅ Feature selection: SUCCESS ({len(selected_features)} feature)")
    print(f"  ✅ Model training: SUCCESS")
    print("\n  La pipeline è stata integrata e testata con successo!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
