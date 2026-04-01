"""
Modulo di validazione per il dataset terremoto Nepal
Verifica:
1. Le feature booleane contengono solo valori ammissibili (configurabili)
2. Le feature categoriche contengono solo valori ammissibili noti da DocumentoDiBordo.txt
"""

import pandas as pd
import numpy as np


# Valori ammissibili per le feature categoriche
CATEGORICAL_VALID_VALUES = {
    'land_surface_condition': {'n', 'o', 't'},
    'foundation_type': {'h', 'i', 'r', 'u', 'w'},
    'roof_type': {'n', 'q', 'x'},
    'ground_floor_type': {'f', 'm', 'v', 'x', 'z'},
    'other_floor_type': {'j', 'q', 's', 'x'},
    'position': {'j', 'o', 's', 't'},
    'plan_configuration': {'a', 'c', 'd', 'f', 'm', 'n', 'o', 'q', 's', 'u'},
    'legal_ownership_status': {'a', 'r', 'v', 'w'}
}

# Feature booleane (dalla documentazione)
BOOLEAN_FEATURES = [
    'has_superstructure_adobe_mud',
    'has_superstructure_mud_mortar_stone',
    'has_superstructure_stone_flag',
    'has_superstructure_cement_mortar_stone',
    'has_superstructure_mud_mortar_brick',
    'has_superstructure_cement_mortar_brick',
    'has_superstructure_timber',
    'has_superstructure_bamboo',
    'has_superstructure_rc_engineered',
    'has_superstructure_rc_non_engineered',
    'has_superstructure_other',
    'has_secondary_use',
    'has_secondary_use_agriculture',
    'has_secondary_use_hotel',
    'has_secondary_use_rental',
    'has_secondary_use_institution',
    'has_secondary_use_school',
    'has_secondary_use_industry',
    'has_secondary_use_health_post',
    'has_secondary_use_gov_office',
    'has_secondary_use_use_police',
    'has_secondary_use_other',
    'age_flag'
]

# Valori ammissibili di default per le feature booleane
DEFAULT_BOOLEAN_VALID_VALUES = {0, 1, 0.0, 1.0}


class DataValidator:
    """Classe per la validazione del dataset."""
    
    def __init__(self, data: pd.DataFrame, boolean_valid_values=None):
        self.data = data.copy()
        # Permette di configurare i valori ammissibili per le feature booleane
        self.boolean_valid_values = boolean_valid_values or DEFAULT_BOOLEAN_VALID_VALUES
        self.validation_report = {
            "boolean_validation": None,
            "categorical_validation": None,
            "has_errors": False
        }
    
    def valida_feature_booleane(self, verbose=True):
        """
        Verifica che le feature booleane contengano solo valori ammissibili.
        Di default sono 0 e 1, ma è configurabile tramite boolean_valid_values.
        
        Returns:
            dict: Report con i risultati della validazione
        """
        boolean_report = {
            "n_features_checked": 0,
            "n_features_valid": 0,
            "invalid_features": {},
            "summary_by_feature": {}
        }
        
        for feature in BOOLEAN_FEATURES:
            if feature not in self.data.columns:
                continue
            
            boolean_report["n_features_checked"] += 1
            
            # Ottieni i valori unici (escludi NaN)
            unique_vals = self.data[feature].dropna().unique()
            unique_vals_set = set(unique_vals)
            
            boolean_report["summary_by_feature"][feature] = {
                "unique_values": sorted(list(unique_vals_set)),
                "dtype": str(self.data[feature].dtype),
                "n_total": len(self.data[feature]),
                "n_not_null": (self.data[feature].notna()).sum(),
                "n_null": (self.data[feature].isna()).sum()
            }
            
            if unique_vals_set.issubset(self.boolean_valid_values):
                boolean_report["n_features_valid"] += 1
            else:
                invalid_vals = unique_vals_set - self.boolean_valid_values
                boolean_report["invalid_features"][feature] = {
                    "invalid_values": sorted(list(invalid_vals)),
                    "count": (self.data[feature].isin(invalid_vals)).sum()
                }
                self.validation_report["has_errors"] = True
        
        self.validation_report["boolean_validation"] = boolean_report
        
        if verbose:
            self._print_boolean_validation_report(boolean_report)
        
        return boolean_report
    
    def valida_feature_categoriche(self, verbose=True):
        """
        Verifica che le feature categoriche contengano solo valori ammissibili.
        
        Returns:
            dict: Report con i risultati della validazione
        """
        categorical_report = {
            "n_features_checked": 0,
            "n_features_valid": 0,
            "invalid_features": {},
            "summary_by_feature": {}
        }
        
        for feature, allowed_values in CATEGORICAL_VALID_VALUES.items():
            if feature not in self.data.columns:
                continue
            
            categorical_report["n_features_checked"] += 1
            
            # Ottieni i valori unici (escludi NaN)
            unique_vals = self.data[feature].dropna().unique()
            unique_vals_set = set(unique_vals)
            
            categorical_report["summary_by_feature"][feature] = {
                "unique_values": sorted(list(unique_vals_set)),
                "dtype": str(self.data[feature].dtype),
                "n_total": len(self.data[feature]),
                "n_not_null": (self.data[feature].notna()).sum(),
                "n_null": (self.data[feature].isna()).sum()
            }
            
            if unique_vals_set.issubset(allowed_values):
                categorical_report["n_features_valid"] += 1
            else:
                invalid_vals = unique_vals_set - allowed_values
                categorical_report["invalid_features"][feature] = {
                    "expected_values": sorted(list(allowed_values)),
                    "invalid_values": sorted(list(invalid_vals)),
                    "count": (self.data[feature].isin(invalid_vals)).sum()
                }
                self.validation_report["has_errors"] = True
        
        self.validation_report["categorical_validation"] = categorical_report
        
        if verbose:
            self._print_categorical_validation_report(categorical_report)
        
        return categorical_report
    
    def esegui_validazione(self, verbose=True):
        """Esegue tutte le validazioni."""
        print("\n" + "="*80)
        print("VALIDAZIONE DATASET")
        print("="*80)
        
        print("\n[1/2] Validazione feature booleane...")
        self.valida_feature_booleane(verbose=verbose)
        
        print("\n[2/2] Validazione feature categoriche...")
        self.valida_feature_categoriche(verbose=verbose)
        
        self._print_summary()
        
        return self.validation_report
    
    def _print_boolean_validation_report(self, report):
        """Stampa il report della validazione delle feature booleane."""
        print(f"\n  Feature booleane controllate: {report['n_features_checked']}")
        print(f"  Feature valide: {report['n_features_valid']}/{report['n_features_checked']}")
        
        if report['invalid_features']:
            print(f"\n  ❌ ERRORI RILEVATI - Feature con valori non ammissibili:")
            for feature, details in report['invalid_features'].items():
                print(f"    - {feature}:")
                print(f"      Valori non ammissibili: {details['invalid_values']}")
                print(f"      Numero di occorrenze: {details['count']}")
        else:
            print(f"\n  ✓ Tutte le feature booleane sono valide")
    
    def _print_categorical_validation_report(self, report):
        """Stampa il report della validazione delle feature categoriche."""
        print(f"\n  Feature categoriche controllate: {report['n_features_checked']}")
        print(f"  Feature valide: {report['n_features_valid']}/{report['n_features_checked']}")
        
        if report['invalid_features']:
            print(f"\n  ❌ ERRORI RILEVATI - Feature con categorie non ammissibili:")
            for feature, details in report['invalid_features'].items():
                print(f"    - {feature}:")
                print(f"      Valori ammissibili: {details['expected_values']}")
                print(f"      Categorie non ammissibili trovate: {details['invalid_values']}")
                print(f"      Numero di occorrenze: {details['count']}")
        else:
            print(f"\n  ✓ Tutte le feature categoriche sono valide")
    
    def _print_summary(self):
        """Stampa il riassunto della validazione."""
        print("\n" + "-"*80)
        if self.validation_report["has_errors"]:
            print("⚠️  ATTENZIONE: Sono stati rilevati degli errori durante la validazione!")
        else:
            print("✓ VALIDAZIONE COMPLETATA: Nessun errore rilevato!")
        print("-"*80)
