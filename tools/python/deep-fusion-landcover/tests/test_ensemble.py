"""test_ensemble.py — Tests for EnsembleClassifier."""
from __future__ import annotations

import numpy as np
import pytest

from deep_fusion_landcover.ensemble import EnsembleClassifier, ClassificationResult
from deep_fusion_landcover.constants import NUM_CLASSES


@pytest.fixture(scope="module")
def trained_ensemble(small_feature_array, small_labels):
    """Lightweight ensemble (no quantum/CNN) fitted on toy data."""
    ens = EnsembleClassifier(use_quantum=False, use_cnn=False, use_obia=False,
                              n_estimators_rf=20)
    ens.fit(small_feature_array, small_labels)
    return ens


class TestEnsembleClassifier:
    def test_fit_does_not_raise(self, small_feature_array, small_labels):
        ens = EnsembleClassifier(use_quantum=False, use_cnn=False, use_obia=False,
                                  n_estimators_rf=10)
        ens.fit(small_feature_array, small_labels)  # should not raise

    def test_predict_returns_classification_result(self, trained_ensemble):
        H, W, N = 8, 8, 50
        rng = np.random.default_rng(1)
        feature_map = rng.random((H, W, N), dtype=np.float32)
        result = trained_ensemble.predict(feature_map, year=2020)

        assert isinstance(result, ClassificationResult)
        assert result.class_map.shape == (H, W)
        assert result.class_probs.shape == (H, W, NUM_CLASSES)
        assert result.confidence.shape == (H, W)
        assert result.year == 2020

    def test_class_labels_in_valid_range(self, trained_ensemble):
        H, W, N = 4, 4, 50
        rng = np.random.default_rng(2)
        feature_map = rng.random((H, W, N), dtype=np.float32)
        result = trained_ensemble.predict(feature_map, year=2010)
        # Class map must be in [1, NUM_CLASSES] (1-indexed)
        valid = result.class_map > 0
        assert int(result.class_map[valid].min()) >= 1
        assert int(result.class_map[valid].max()) <= NUM_CLASSES

    def test_probs_sum_to_one(self, trained_ensemble):
        H, W, N = 4, 4, 50
        rng = np.random.default_rng(3)
        feature_map = rng.random((H, W, N), dtype=np.float32)
        result = trained_ensemble.predict(feature_map, year=2000)
        row_sums = result.class_probs.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)

    def test_confidence_matches_max_probs(self, trained_ensemble):
        H, W, N = 4, 4, 50
        rng = np.random.default_rng(4)
        feature_map = rng.random((H, W, N), dtype=np.float32)
        result = trained_ensemble.predict(feature_map, year=2005)
        expected_conf = result.class_probs.max(axis=-1)
        np.testing.assert_allclose(result.confidence, expected_conf, atol=1e-5)

    def test_save_load_roundtrip(self, tmp_path, small_feature_array, small_labels):
        ens = EnsembleClassifier(use_quantum=False, use_cnn=False, use_obia=False,
                                  n_estimators_rf=10)
        ens.fit(small_feature_array, small_labels)

        H, W, N = 4, 4, 50
        rng = np.random.default_rng(5)
        fmap = rng.random((H, W, N), dtype=np.float32)
        r1 = ens.predict(fmap, year=2000)

        ens.save(tmp_path / "models")

        ens2 = EnsembleClassifier(use_quantum=False, use_cnn=False, use_obia=False,
                                   n_estimators_rf=10)
        ens2.load(tmp_path / "models")
        r2 = ens2.predict(fmap, year=2000)

        np.testing.assert_array_equal(r1.class_map, r2.class_map)
