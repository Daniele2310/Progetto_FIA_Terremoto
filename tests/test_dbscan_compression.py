import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN as SklearnDBSCAN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.outlier_detection.DBSCAN import (
    calcola_range_eps,
    comprimi_punti_duplicati,
    esegui_dbscan,
)


class TestDbscanCompression(unittest.TestCase):
    def setUp(self):
        # Due cluster compatti con molti duplicati e un outlier isolato.
        self.X = np.array([
            [0.00, 0.00],
            [0.00, 0.00],
            [0.00, 0.00],
            [0.05, 0.05],
            [0.05, 0.05],
            [5.00, 5.00],
            [5.10, 5.10],
            [5.20, 5.20],
            [20.0, 20.0],
        ])
        self.params = {"eps": 0.30, "min_samples": 3, "n_jobs": -1}

    def test_weighted_dbscan_preserves_outlier_mask(self):
        labels_full = SklearnDBSCAN(**self.params).fit_predict(self.X)

        X_unique, inverse_indices, sample_weight = comprimi_punti_duplicati(self.X)
        labels_compressed = esegui_dbscan(
            X_unique,
            eps=self.params["eps"],
            min_samples=self.params["min_samples"],
            sample_weight=sample_weight,
            inverse_indices=inverse_indices,
        )

        np.testing.assert_array_equal(labels_full == -1, labels_compressed == -1)

    def test_weighted_k_distance_matches_expanded_dataset(self):
        min_samples_values = [2, 3]

        eps_min_full, eps_max_full, k_dist_full = calcola_range_eps(
            self.X, min_samples_values
        )

        X_unique, _, sample_weight = comprimi_punti_duplicati(self.X)
        eps_min_weighted, eps_max_weighted, k_dist_weighted = calcola_range_eps(
            X_unique, min_samples_values, sample_weight=sample_weight
        )

        self.assertAlmostEqual(eps_min_full, eps_min_weighted, places=10)
        self.assertAlmostEqual(eps_max_full, eps_max_weighted, places=10)
        for k in min_samples_values:
            np.testing.assert_allclose(k_dist_full[k], k_dist_weighted[k])


if __name__ == "__main__":
    unittest.main()
