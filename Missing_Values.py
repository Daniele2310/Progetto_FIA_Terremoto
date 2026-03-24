import pandas as pd

# 1. Caricamento del dataset
df_values = pd.read_csv('train_values.csv')
df_labels = pd.read_csv('train_labels.csv')

df_merged = pd.merge(df_values, df_labels, on='building_id')

# 3. Creiamo la copia su cui lavorare per evitare di toccare df_merged
df = df_merged.copy()
# ==========================================
# CONTROLLO VALORI NULLI
# ==========================================

# Verifica presenza generica di valori nulli
total_nulls = df.isnull().sum().sum()
print(f"Ci sono in totale {total_nulls} valori nulli nel dataset.\n")

# -- Controllo sulle Feature --
# Calcolo la percentuale di valori nulli per ogni colonna
null_percent_cols = (df.isnull().sum() / len(df)) * 100

# Filtro le colonne che superano la soglia del 70%
cols_to_drop = null_percent_cols[null_percent_cols > 70]

if not cols_to_drop.empty:
    print("Colonne con più del 70% di valori nulli:")
    print(cols_to_drop)
else:
    print("Nessuna colonna ha più del 70% di valori nulli.\n")

# -- Controllo sui Record --
# Calcolo la percentuale di valori nulli per ogni riga (lungo l'asse 1)
null_percent_rows = (df.isnull().sum(axis=1) / df.shape[1]) * 100

# Filtro le righe che superano la soglia del 70%
rows_to_drop = null_percent_rows[null_percent_rows > 70]

if not rows_to_drop.empty:
    print(f"Ci sono {len(rows_to_drop)} righe con più del 70% di valori nulli.")
    # Mostra gli indici delle righe problematiche
    print("Indici delle prime 5 righe:", rows_to_drop.head().index.tolist())
else:
    print("Nessuna riga ha più del 70% di valori nulli.\n")

# ==========================================
# CONTROLLO BILANCIAMENTO DELLE LABELS
# ==========================================

target_col = 'damage_grade'

if target_col in df.columns:
    print("--- Bilanciamento delle Labels ---")

    # Conteggio assoluto ordinato per nome della classe (1, 2, 3)
    label_counts = df[target_col].value_counts().sort_index()
    print("Conteggio per classe:\n", label_counts)

    # Percentuale approssimata a 1 cifra decimale e ordinata per classe (1, 2, 3)
    label_percent = (df[target_col].value_counts(normalize=True) * 100).round(1).sort_index()
    print("\nPercentuale per classe (%):\n", label_percent)
else:
    print(f"Attenzione: La colonna target '{target_col}' non è presente nel DataFrame.")
    print("Assicurati di aver fatto il merge con il file delle labels.")




