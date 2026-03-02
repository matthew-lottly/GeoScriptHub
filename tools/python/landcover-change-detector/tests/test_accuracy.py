"""Tests for accuracy assessment.

Covers:
- NLCD reclassification
- Confusion matrix computation
- Cohen's Kappa
- Producer's / User's accuracy
- F1 scores
- Temporal consistency
- CSV export
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from landcover_change.constants import NUM_CLASSES, CLASS_NAMES, NLCD_TO_CLASS
from landcover_change.accuracy import (
    reclassify_nlcd,
    compute_confusion_matrix,
    compute_temporal_consistency,
    AccuracyAssessor,
    AccuracyResult,
    ConfusionMetrics,
    export_accuracy_csv,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(77)


# ── NLCD Reclassification ────────────────────────────────────────

class TestNLCDReclassification:
    def test_known_codes(self) -> None:
        # Water (11) → 0, Deciduous Forest (41) → 2, Dev High (24) → 6
        nlcd = np.array([11, 41, 24, 31, 82], dtype="int32")
        result = reclassify_nlcd(nlcd)
        assert result[0] == NLCD_TO_CLASS[11]
        assert result[1] == NLCD_TO_CLASS[41]
        assert result[2] == NLCD_TO_CLASS[24]
        assert result[3] == NLCD_TO_CLASS[31]  # barren
        assert result[4] == NLCD_TO_CLASS[82]  # agriculture

    def test_unknown_code_defaults_barren(self) -> None:
        nlcd = np.array([255, 0, 99], dtype="int32")
        result = reclassify_nlcd(nlcd)
        assert np.all(result == 7)  # default barren

    def test_preserves_shape(self) -> None:
        nlcd = np.full((5, 5), 11, dtype="int32")
        result = reclassify_nlcd(nlcd)
        assert result.shape == (5, 5)


# ── Confusion Matrix ─────────────────────────────────────────────

class TestConfusionMatrix:
    def test_perfect_classification(self) -> None:
        labels = np.arange(NUM_CLASSES).repeat(10)
        cm = compute_confusion_matrix(labels, labels, 2020)
        assert cm.overall_accuracy == 1.0
        assert cm.kappa == 1.0
        np.testing.assert_allclose(cm.producers_accuracy, 1.0)
        np.testing.assert_allclose(cm.users_accuracy, 1.0)
        np.testing.assert_allclose(cm.f1, 1.0)

    def test_random_classification(self, rng: np.random.Generator) -> None:
        pred = rng.integers(0, NUM_CLASSES, size=1000).astype("int32")
        ref = rng.integers(0, NUM_CLASSES, size=1000).astype("int32")
        cm = compute_confusion_matrix(pred, ref, 2020)
        # Random should have ~1/NUM_CLASSES accuracy
        assert 0.0 < cm.overall_accuracy < 0.4
        assert -0.2 < cm.kappa < 0.2
        assert cm.matrix.sum() == 1000

    def test_matrix_shape(self, rng: np.random.Generator) -> None:
        pred = rng.integers(0, NUM_CLASSES, size=500).astype("int32")
        ref = rng.integers(0, NUM_CLASSES, size=500).astype("int32")
        cm = compute_confusion_matrix(pred, ref, 2010)
        assert cm.matrix.shape == (NUM_CLASSES, NUM_CLASSES)
        assert cm.year == 2010

    def test_known_matrix(self) -> None:
        # 3 correct water, 2 correct forest, 1 water→forest error
        pred = np.array([0, 0, 0, 2, 2, 0], dtype="int32")
        ref = np.array([0, 0, 0, 2, 2, 2], dtype="int32")
        cm = compute_confusion_matrix(pred, ref, 2020)
        assert cm.matrix[0, 0] == 3  # correct water
        assert cm.matrix[2, 2] == 2  # correct forest
        assert cm.matrix[0, 2] == 1  # water predicted but was forest
        assert cm.overall_accuracy == 5.0 / 6.0

    def test_kappa_bounds(self, rng: np.random.Generator) -> None:
        pred = rng.integers(0, NUM_CLASSES, size=500).astype("int32")
        ref = rng.integers(0, NUM_CLASSES, size=500).astype("int32")
        cm = compute_confusion_matrix(pred, ref, 2020)
        assert -1.0 <= cm.kappa <= 1.0


# ── Temporal Consistency ─────────────────────────────────────────

class TestTemporalConsistency:
    def test_stable_sequence(self) -> None:
        cm = np.full((10, 10), 3, dtype="int32")
        maps = [cm.copy() for _ in range(10)]
        years = list(range(2010, 2020))
        tc = compute_temporal_consistency(maps, years)
        assert tc.mean_annual_change_rate == 0.0
        assert tc.flip_flop_fraction == 0.0

    def test_flip_flop_detection(self) -> None:
        shape = (10, 10)
        cm0 = np.zeros(shape, dtype="int32")
        cm1 = np.ones(shape, dtype="int32")  # all change
        cm2 = np.zeros(shape, dtype="int32")  # all revert
        tc = compute_temporal_consistency([cm0, cm1, cm2], [2020, 2021, 2022])
        assert tc.flip_flop_fraction == 1.0  # 100% flip-flop
        assert tc.mean_annual_change_rate == 1.0

    def test_single_year(self) -> None:
        cm = np.zeros((5, 5), dtype="int32")
        tc = compute_temporal_consistency([cm], [2020])
        assert tc.mean_annual_change_rate == 0.0


# ── CSV Export ────────────────────────────────────────────────────

class TestCSVExport:
    def test_export_creates_files(self, rng: np.random.Generator) -> None:
        from landcover_change.accuracy import TemporalConsistency

        pred = rng.integers(0, NUM_CLASSES, size=200).astype("int32")
        ref = rng.integers(0, NUM_CLASSES, size=200).astype("int32")
        m = compute_confusion_matrix(pred, ref, 2019)

        result = AccuracyResult(
            metrics_per_epoch=[m],
            overall_metrics=m,
            temporal_consistency=TemporalConsistency(0.05, 0.1, 0.02),
            nlcd_years_used=[2019],
            sample_count=200,
        )

        with tempfile.TemporaryDirectory() as tmp:
            paths = export_accuracy_csv(result, Path(tmp))
            assert len(paths) >= 3
            for p in paths:
                assert p.exists()
                content = p.read_text(encoding="utf-8")
                assert len(content) > 10


# ── AccuracyAssessor Integration ─────────────────────────────────

class TestAccuracyAssessor:
    def test_assess_without_nlcd(self, rng: np.random.Generator) -> None:
        """When NLCD fetch fails, should still produce self-consistency."""
        from landcover_change.quantum_classifier import ClassificationResult

        years = [2019, 2020, 2021]
        classifications = []
        for y in years:
            cm = rng.integers(0, NUM_CLASSES, size=(10, 10)).astype("int32")
            probs = np.zeros((10, 10, NUM_CLASSES), dtype="float32")
            for c in range(NUM_CLASSES):
                probs[:, :, c] = (cm == c).astype("float32") * 0.8 + 0.02
            probs /= probs.sum(axis=-1, keepdims=True)
            classifications.append(ClassificationResult(
                year=y, class_map=cm,
                class_probabilities=probs,
                confidence=probs.max(axis=-1),
                quantum_entropy=np.zeros((10, 10), dtype="float32"),
                valid_mask=np.ones((10, 10), dtype=bool),
                shape=(10, 10),
            ))

        bbox = (-95.15, 29.70, -95.02, 29.80)
        assessor = AccuracyAssessor(bbox=bbox)
        # This will try to fetch NLCD but fail (no network in tests typically)
        result = assessor.assess(classifications, years)
        assert isinstance(result, AccuracyResult)
        assert result.overall_metrics is not None
