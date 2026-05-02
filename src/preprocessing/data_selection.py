import pandas as pd
from typing import Optional

def get_balanced_sample(df: pd.DataFrame, target_col: str, max_per_class: int, random_state: int = 42) -> pd.DataFrame:
    """
    Campionamento Polarizzato (Bilanciato).
    Estrae un campione bilanciato dal dataset originale, forzando le classi ad avere 
    lo stesso numero di campioni (limitato da max_per_class o dalla numerosità della classe minoritaria).
    Utile per benchmark rigorosi dove si vuole evitare che la classe maggioritaria domini.
    
    Args:
        df: Il DataFrame Pandas originale.
        target_col: La colonna target contenente le classi.
        max_per_class: Il numero massimo di campioni da estrarre per ogni classe.
        random_state: Seed per la riproducibilità.
        
    Returns:
        Un DataFrame pandas campionato e rimescolato.
    """
    if target_col not in df.columns:
        raise ValueError(f"Colonna '{target_col}' non trovata nel DataFrame.")
        
    sampled_dfs = []
    for cls in df[target_col].unique():
        cls_df = df[df[target_col] == cls]
        n_samples = min(len(cls_df), max_per_class)
        sampled_dfs.append(cls_df.sample(n=n_samples, random_state=random_state))
        
    # Concateniamo e mescoliamo le righe risultanti (frac=1)
    return pd.concat(sampled_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)

def get_stratified_sample(df: pd.DataFrame, target_col: str, n_samples: int, random_state: int = 42) -> pd.DataFrame:
    """
    Campionamento Non Polarizzato (Stratificato / Proporzionale).
    Estrae un sottoinsieme di n_samples mantenendo inalterata la distribuzione 
    delle classi presente nel dataset originale.
    Utile per test veloci dove si vuole mantenere la distribuzione originaria intatta.
    
    Args:
        df: Il DataFrame Pandas originale.
        target_col: La colonna target contenente le classi.
        n_samples: Il numero totale di campioni desiderati nel DataFrame finale.
        random_state: Seed per la riproducibilità.
        
    Returns:
        Un DataFrame pandas campionato proporzionalmente.
    """
    if target_col not in df.columns:
        raise ValueError(f"Colonna '{target_col}' non trovata nel DataFrame.")
    
    total_rows = len(df)
    if n_samples >= total_rows:
        return df.copy()
        
    # Calcoliamo quanti campioni estrarre per ogni classe mantenendo le proporzioni
    class_counts = df[target_col].value_counts(normalize=True)
    
    sampled_dfs = []
    for cls, proportion in class_counts.items():
        cls_df = df[df[target_col] == cls]
        # Calcoliamo i campioni teorici e facciamo in modo che non superino quelli reali
        n_cls_samples = int(round(proportion * n_samples))
        n_cls_samples = min(len(cls_df), n_cls_samples)
        
        if n_cls_samples > 0:
            sampled_dfs.append(cls_df.sample(n=n_cls_samples, random_state=random_state))
            
    return pd.concat(sampled_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
