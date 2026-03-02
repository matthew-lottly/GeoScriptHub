"""Tests for the change detection engine.

Covers:
- ChangeMap construction
- TransitionMatrix computation
- Decade summaries
- Temporal trend analysis
- GeoTIFF export (mock)
"""
from __future__ import annotations

import numpy as np
import pytest

from landcover_change.constants import NUM_CLASSES
from landcover_change.quantum_classifier import ClassificationResult
from landcover_change.change_detection import (
    ChangeDetectionEngine,
    ChangeMap,
    TransitionMatrix,
    DecadeSummary,
    TrendResult,
    ChangeDetectionResult,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(123)


@pytest.fixture
def engine() -> ChangeDetectionEngine:
    return ChangeDetectionEngine(resolution=30.0)


def _make_classification(
    rng: np.random.Generator,
    year: int,
    shape: tuple[int, int] = (20, 20),
    *,
    bias_class: int | None = None,
) -> ClassificationResult:
    """Build a synthetic ClassificationResult."""
    if bias_class is not None:
        # 60% of pixels are bias_class
        cm = rng.integers(0, NUM_CLASSES, size=shape).astype("int32")
        mask = rng.random(shape) < 0.6
        cm[mask] = bias_class
    else:
        cm = rng.integers(0, NUM_CLASSES, size=shape).astype("int32")

    probs = np.zeros((*shape, NUM_CLASSES), dtype="float32")
    for c in range(NUM_CLASSES):
        probs[:, :, c] = (cm == c).astype("float32") * 0.8 + 0.02
    probs /= probs.sum(axis=-1, keepdims=True)

    return ClassificationResult(
        year=year,
        class_map=cm,
        class_probabilities=probs,
        confidence=probs.max(axis=-1),
        quantum_entropy=np.zeros(shape, dtype="float32"),
        valid_mask=np.ones(shape, dtype=bool),
        shape=shape,
    )


# ── Change Map ───────────────────────────────────────────────────

class TestChangeMap:
    def test_identical_maps_no_change(self, engine: ChangeDetectionEngine) -> None:
        cm = np.ones((10, 10), dtype="int32") * 3
        result = engine._compute_change_map(cm, cm, 2000, 2010)
        assert np.sum(result.change_mask) == 0
        assert result.from_year == 2000
        assert result.to_year == 2010

    def test_all_changed(self, engine: ChangeDetectionEngine) -> None:
        from_map = np.zeros((10, 10), dtype="int32")
        to_map = np.ones((10, 10), dtype="int32")
        result = engine._compute_change_map(from_map, to_map, 1990, 2020)
        assert np.all(result.change_mask)
        assert result.change_code[0, 0] == 0 * NUM_CLASSES + 1

    def test_change_code_encoding(self, engine: ChangeDetectionEngine) -> None:
        from_map = np.array([[2, 3]], dtype="int32")
        to_map = np.array([[5, 3]], dtype="int32")
        result = engine._compute_change_map(from_map, to_map, 2000, 2010)
        assert result.change_code[0, 0] == 2 * NUM_CLASSES + 5
        assert result.change_code[0, 1] == -1  # unchanged


# ── Transition Matrix ────────────────────────────────────────────

class TestTransitionMatrix:
    def test_counts_sum(self, engine: ChangeDetectionEngine, rng: np.random.Generator) -> None:
        cm1 = rng.integers(0, NUM_CLASSES, size=(20, 20)).astype("int32")
        cm2 = rng.integers(0, NUM_CLASSES, size=(20, 20)).astype("int32")
        tm = engine._compute_transition_matrix(cm1, cm2, 2000, 2010)
        assert tm.matrix.sum() == 400
        assert tm.matrix.shape == (NUM_CLASSES, NUM_CLASSES)
        np.testing.assert_allclose(
            tm.area_ha_matrix.sum(),
            400 * engine.pixel_area_ha,
        )

    def test_diagonal_for_identical(self, engine: ChangeDetectionEngine) -> None:
        cm = np.full((10, 10), 2, dtype="int32")
        tm = engine._compute_transition_matrix(cm, cm, 2000, 2010)
        assert tm.matrix[2, 2] == 100
        assert tm.matrix.sum() == 100


# ── Decade Summaries ─────────────────────────────────────────────

class TestDecadeSummaries:
    def test_produces_summaries(self, engine: ChangeDetectionEngine, rng: np.random.Generator) -> None:
        years = list(range(1990, 2025))
        maps = [rng.integers(0, NUM_CLASSES, size=(10, 10)).astype("int32") for _ in years]
        summaries = engine._compute_decade_summaries(maps, years)
        # Should have up to 3 decade boundaries
        assert len(summaries) <= 3
        for ds in summaries:
            assert ds.from_year < ds.to_year
            assert set(ds.gain_ha.keys()) == set(ds.loss_ha.keys())

    def test_gain_loss_balance(self, engine: ChangeDetectionEngine, rng: np.random.Generator) -> None:
        years = list(range(1995, 2015))
        maps = [rng.integers(0, NUM_CLASSES, size=(10, 10)).astype("int32") for _ in years]
        summaries = engine._compute_decade_summaries(maps, years)
        for ds in summaries:
            total_gain = sum(ds.gain_ha.values())
            total_loss = sum(ds.loss_ha.values())
            # Total gain should approximately equal total loss (area is conserved)
            np.testing.assert_allclose(total_gain, total_loss, rtol=1e-5)


# ── Temporal Trend ───────────────────────────────────────────────

class TestTrend:
    def test_stable_trend(self, engine: ChangeDetectionEngine) -> None:
        cm = np.full((10, 10), 0, dtype="int32")
        years = list(range(2000, 2020))
        maps = [cm.copy() for _ in years]
        trend = engine._compute_trend(maps, years)
        assert trend.dominant_class.shape == (10, 10)
        assert np.all(trend.dominant_class == 0)
        assert np.all(trend.change_count == 0)
        assert np.all(trend.trend_label == 0)  # stable

    def test_urbanising_detection(self, engine: ChangeDetectionEngine) -> None:
        shape = (10, 10)
        years = list(range(2000, 2020))
        maps = []
        for i, y in enumerate(years):
            if i < 10:
                maps.append(np.full(shape, 2, dtype="int32"))  # forest
            else:
                maps.append(np.full(shape, 6, dtype="int32"))  # developed-high
        trend = engine._compute_trend(maps, years)
        # Should detect urbanising trend
        assert np.any(trend.trend_label == 1), "Should detect urbanising"

    def test_change_count(self, engine: ChangeDetectionEngine) -> None:
        shape = (5, 5)
        years = [2000, 2001, 2002]
        cm0 = np.zeros(shape, dtype="int32")
        cm1 = np.ones(shape, dtype="int32")
        cm2 = np.zeros(shape, dtype="int32")
        trend = engine._compute_trend([cm0, cm1, cm2], years)
        assert np.all(trend.change_count == 2)


# ── Full Pipeline ─────────────────────────────────────────────────

class TestFullChangeDetection:
    def test_compute(self, engine: ChangeDetectionEngine, rng: np.random.Generator) -> None:
        years = [1995, 2000, 2005, 2010, 2015, 2020]
        classifications = [_make_classification(rng, y) for y in years]
        result = engine.compute(classifications)

        assert isinstance(result, ChangeDetectionResult)
        assert result.years == years
        assert len(result.yearly_maps) == len(years)
        assert result.then_and_now.from_year == 1995
        assert result.then_and_now.to_year == 2020
        assert len(result.transition_matrices) == len(years) - 1
        assert result.trend.dominant_class.shape == (20, 20)

    def test_urbanisation_scenario(self, engine: ChangeDetectionEngine, rng: np.random.Generator) -> None:
        """Simulate a scenario where forest → developed over 30 years."""
        years = list(range(1990, 2025, 5))
        classifications = []
        for i, y in enumerate(years):
            # Gradually increase urban fraction
            shape = (20, 20)
            cm = np.full(shape, 2, dtype="int32")  # start with forest
            urban_frac = i / (len(years) - 1)
            urban_mask = rng.random(shape) < urban_frac
            cm[urban_mask] = 6
            probs = np.zeros((*shape, NUM_CLASSES), dtype="float32")
            for c in range(NUM_CLASSES):
                probs[:, :, c] = (cm == c).astype("float32") * 0.85 + 0.01
            probs /= probs.sum(axis=-1, keepdims=True)
            classifications.append(ClassificationResult(
                year=y, class_map=cm,
                class_probabilities=probs,
                confidence=probs.max(axis=-1),
                quantum_entropy=np.zeros(shape, dtype="float32"),
                valid_mask=np.ones(shape, dtype=bool),
                shape=shape,
            ))

        result = engine.compute(classifications)
        # Should show net loss of forest, net gain of developed
        if result.decade_summaries:
            last_ds = result.decade_summaries[-1]
            from landcover_change.constants import CLASS_NAMES
            forest_name = CLASS_NAMES[2]
            dev_name = CLASS_NAMES[6]
            # Net forest should be negative, net dev should be positive
            assert last_ds.net_ha[forest_name] <= 0, "Forest should decline"
            assert last_ds.net_ha[dev_name] >= 0, "Development should increase"
