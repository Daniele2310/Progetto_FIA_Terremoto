"""
Modulo MissingValues - Controllo valori nulli e bilanciamento labels
"""

import pandas as pd


class MissingValuesHandler:
    """Classe per il controllo e la gestione dei valori nulli nel dataset."""
    
    def __init__(self, null_threshold=70):
        self.null_threshold = null_threshold
        self.report = {}
    
    def analizza(self, df, target_col='damage_grade'):
        """Esegue l'analisi dei valori nulli e del bilanciamento."""
        self.report = {
            'total_nulls': 0,
            'sparse_cols': [],
            'sparse_rows': [],
            'label_distribution': None,
            'label_percent': None
        }
        
        total_nulls = df.isnull().sum().sum()
        self.report['total_nulls'] = total_nulls
        print(f"Valori nulli totali: {total_nulls}")
        
        null_percent_cols = (df.isnull().sum() / len(df)) * 100
        sparse_cols = null_percent_cols[null_percent_cols > self.null_threshold]
        
        if not sparse_cols.empty:
            print(f"Colonne con >{self.null_threshold}% nulli:")
            for col, percent in sparse_cols.items():
                print(f"  {col}: {percent:.1f}%")
            self.report['sparse_cols'] = sparse_cols.to_dict()
        
        null_percent_rows = (df.isnull().sum(axis=1) / df.shape[1]) * 100
        sparse_rows = null_percent_rows[null_percent_rows > self.null_threshold]
        
        if not sparse_rows.empty:
            print(f"Righe con >{self.null_threshold}% nulli: {len(sparse_rows)}")
            self.report['sparse_rows'] = sparse_rows.index.tolist()
        
        if target_col in df.columns:
            label_counts = df[target_col].value_counts().sort_index()
            label_percent = (df[target_col].value_counts(normalize=True) * 100).round(1).sort_index()
            
            print("\nDistribuzione label:")
            for label, count in label_counts.items():
                percent = label_percent[label]
                print(f"  Classe {label}: {count} ({percent}%)")
            
            self.report['label_distribution'] = label_counts.to_dict()
            self.report['label_percent'] = label_percent.to_dict()
        
        return self.report
    

