"""
Test di monotonia VELOCIZZATO per il Branch-and-Bound.
Versione rapida che usa un campione del dataset per verificare l'ipotesi.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class FastMonotonicityTester:
    """Test veloce della monotonia su un campione del dataset."""
    
    def __init__(self, random_state: int = 42, n_trials: int = 20, sample_size: int = 5000):
        self.random_state = random_state
        self.n_trials = n_trials
        self.sample_size = sample_size
    
    @staticmethod
    def _load_data(project_root: Path) -> tuple[np.ndarray, np.ndarray]:
        """Carica il dataset preprocessato."""
        preprocessed_path = project_root / "Data" / "preprocessed" / "train_features_labels_preprocessed.csv"
        
        if preprocessed_path.exists():
            df = pd.read_csv(preprocessed_path)
            exclude_cols = {'building_id', 'damage_grade'}
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            X = df[feature_cols].astype(float).to_numpy()
            y = (df['damage_grade'].astype(int) - 1).to_numpy()  # Classi 0, 1, 2
            
            return X, y
        
        raise FileNotFoundError(f"Dataset non trovato in {preprocessed_path}")
    
    def _evaluate_subset(self, X_train, X_val, y_train, y_val, feature_indices, scoring='accuracy'):
        """Valuta un subset di feature."""
        if len(feature_indices) == 0:
            return 0.0
        
        X_train_subset = X_train[:, feature_indices]
        X_val_subset = X_val[:, feature_indices]
        
        model = LogisticRegression(max_iter=500, random_state=self.random_state, n_jobs=1)
        model.fit(X_train_subset, y_train)
        y_pred = model.predict(X_val_subset)
        
        if scoring == 'accuracy':
            return float(accuracy_score(y_val, y_pred))
        elif scoring == 'f1_micro':
            return float(f1_score(y_val, y_pred, average='micro', zero_division=0))
        
        return float(accuracy_score(y_val, y_pred))
    
    def run(self, X, y):
        """Esegui il test veloce."""
        print("\n" + "=" * 80)
        print("TEST DI MONOTONIA - VERSIONE VELOCIZZATA")
        print("=" * 80)
        
        print(f"\nCampionamento: {self.sample_size} su {X.shape[0]} campioni")
        
        # Campiona i dati
        indices = np.random.RandomState(self.random_state).choice(len(y), size=min(self.sample_size, len(y)), replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
        
        n_features = X_sample.shape[1]
        print(f"Feature da valutare: {n_features}")
        print(f"Trial: {self.n_trials}")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_sample, y_sample,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y_sample
        )
        
        violations_acc = 0
        violations_f1 = 0
        total_tests = 0
        
        rng = np.random.default_rng(self.random_state)
        
        print("\nEsecuzione trial...")
        for trial_idx in range(self.n_trials):
            if trial_idx % 5 == 0:
                print(f"  Trial {trial_idx}/{self.n_trials}")
            
            # Subset casuale
            k = rng.integers(1, max(2, n_features - 1))
            selected = rng.choice(n_features, size=k, replace=False)
            
            score_k_acc = self._evaluate_subset(X_train, X_val, y_train, y_val, selected, 'accuracy')
            score_k_f1 = self._evaluate_subset(X_train, X_val, y_train, y_val, selected, 'f1_micro')
            
            # Aggiungi feature casuale
            available = np.setdiff1d(np.arange(n_features), selected)
            if len(available) == 0:
                continue
            
            new_feat = rng.choice(available)
            extended = np.append(selected, new_feat)
            
            score_k1_acc = self._evaluate_subset(X_train, X_val, y_train, y_val, extended, 'accuracy')
            score_k1_f1 = self._evaluate_subset(X_train, X_val, y_train, y_val, extended, 'f1_micro')
            
            total_tests += 1
            
            if score_k1_acc < score_k_acc - 1e-6:
                violations_acc += 1
            
            if score_k1_f1 < score_k_f1 - 1e-6:
                violations_f1 += 1
        
        # Stampa risultati
        print("\n" + "=" * 80)
        print("RISULTATI")
        print("=" * 80)
        
        acc_compliance = 100.0 * (1 - violations_acc / max(1, total_tests))
        f1_compliance = 100.0 * (1 - violations_f1 / max(1, total_tests))
        
        print(f"\nAccuracy:")
        print(f"  Test eseguiti: {total_tests}")
        print(f"  Violazioni: {violations_acc}")
        print(f"  Conformità: {acc_compliance:.1f}%")
        
        print(f"\nF1-Micro:")
        print(f"  Test eseguiti: {total_tests}")
        print(f"  Violazioni: {violations_f1}")
        print(f"  Conformità: {f1_compliance:.1f}%")
        
        print("\n" + "=" * 80)
        print("CONCLUSIONE PER BRANCH-AND-BOUND")
        print("=" * 80)
        
        if acc_compliance > 95 and f1_compliance > 95:
            print("\n✓ IPOTESI DI MONOTONIA RISPETTATA")
            print("  → Il branch-and-bound È APPLICABILE")
            print("  → Potete usare la monotonia per potare l'albero di ricerca")
            print("  → Bound valido per tutte le metriche")
            return True
        elif acc_compliance > 80 and f1_compliance > 80:
            print("\n⚠ IPOTESI DI MONOTONIA PARZIALMENTE RISPETTATA")
            print("  → Il branch-and-bound È APPLICABILE CON CAUTELA")
            print("  → Il bound potrebbe essere debole in alcuni casi")
            print("  → Aggiungere margini di sicurezza ai bound")
            return True
        else:
            print("\n✗ IPOTESI DI MONOTONIA NON RISPETTATA")
            print("  → Il branch-and-bound NON È APPLICABILE")
            print("  → Preferite SFS, SBS o algoritmi euristici alternativi")
            print("  → Il problema ha caratteristiche non-convesse")
            return False


def main():
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[1]
    
    print("Caricamento dataset...")
    try:
        X, y = FastMonotonicityTester._load_data(project_root)
        print(f"✓ Dataset caricato: {X.shape} campioni, {X.shape[1]} feature")
    except FileNotFoundError as e:
        print(f"✗ Errore: {e}")
        return
    
    tester = FastMonotonicityTester(n_trials=20, sample_size=5000)
    result = tester.run(X, y)
    
    # Salva risultato
    report_path = project_root / "src.feature_selection" / "outputs" / "monotonia_report_fast.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(f"BRANCH-AND-BOUND APPLICABILE: {result}\n")
    
    print(f"\n✓ Report salvato in: {report_path}")


if __name__ == "__main__":
    main()
