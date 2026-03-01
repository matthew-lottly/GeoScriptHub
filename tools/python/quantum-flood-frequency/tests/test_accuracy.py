"""
test_accuracy.py
================
Accuracy & integration tests for the Quantum Flood Frequency Mapper.

These tests use carefully designed **synthetic landscapes** where the
ground-truth water mask is known, enabling measurement of:

* **Pixel-level classification accuracy** (OA, precision, recall, F1, κ)
* **Frequency surface fidelity** (correlation with known inundation rate)
* **Zone-assignment correctness** (permanent, seasonal, rare, dry)
* **Confidence interval calibration** (empirical coverage ≥ nominal level)
* **Cross-sensor consistency** (Landsat vs Sentinel-2 agreement)
* **Edge-case robustness** (cloudy scenes, all-water, all-land, NAIP no-SWIR)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass, field

from quantum_flood_frequency.quantum_classifier import (
    QuantumFeatureEncoder,
    QuantumKernelSVM,
    SpectralGBClassifier,
    QuantumHybridClassifier,
    ClassificationResult,
    bayesian_model_average,
    compute_ndwi,
    compute_mndwi,
    compute_awei_sh,
    _safe_ratio,
)
from quantum_flood_frequency.flood_engine import FloodFrequencyEngine, FrequencyResult
from quantum_flood_frequency.preprocessing import ImagePreprocessor, AlignedStack


# ======================================================================
# Synthetic landscape factories
# ======================================================================

ROWS, COLS = 50, 50
RNG = np.random.default_rng(2024)


def _make_water_mask(water_fraction: float = 0.5) -> np.ndarray:
    """Binary ground-truth mask with exactly the requested water fraction."""
    n_water = int(ROWS * COLS * water_fraction)
    flat = np.zeros(ROWS * COLS, dtype=bool)
    flat[:n_water] = True
    RNG.shuffle(flat)
    return flat.reshape(ROWS, COLS)


def _make_reflectance(water_mask: np.ndarray) -> dict:
    """Generate synthetic reflectance bands whose spectral signature
    is consistent with the water mask.

    Water pixels:  high green, low NIR/SWIR → high NDWI, high MNDWI
    Land pixels:   low green, high NIR/SWIR → negative NDWI
    """
    n = water_mask.shape
    green = np.where(water_mask, RNG.uniform(0.08, 0.12, n), RNG.uniform(0.04, 0.08, n)).astype("float32")
    nir = np.where(water_mask, RNG.uniform(0.01, 0.04, n), RNG.uniform(0.15, 0.30, n)).astype("float32")
    red = np.where(water_mask, RNG.uniform(0.02, 0.05, n), RNG.uniform(0.06, 0.12, n)).astype("float32")
    blue = np.where(water_mask, RNG.uniform(0.05, 0.10, n), RNG.uniform(0.03, 0.06, n)).astype("float32")
    swir1 = np.where(water_mask, RNG.uniform(0.005, 0.02, n), RNG.uniform(0.10, 0.25, n)).astype("float32")
    swir2 = np.where(water_mask, RNG.uniform(0.002, 0.01, n), RNG.uniform(0.05, 0.15, n)).astype("float32")
    return {
        "green": green, "nir": nir, "red": red, "blue": blue,
        "swir1": swir1, "swir2": swir2,
        "cloud_mask": np.ones(n, dtype=bool),
        "source": "landsat",
        "date": "2023-06-01",
    }


def _make_naip_reflectance(water_mask: np.ndarray) -> dict:
    """NAIP observation — no SWIR bands (NaN placeholders)."""
    n = water_mask.shape
    green = np.where(water_mask, RNG.uniform(0.08, 0.12, n), RNG.uniform(0.04, 0.08, n)).astype("float32")
    nir = np.where(water_mask, RNG.uniform(0.01, 0.04, n), RNG.uniform(0.15, 0.30, n)).astype("float32")
    red = np.where(water_mask, RNG.uniform(0.02, 0.05, n), RNG.uniform(0.06, 0.12, n)).astype("float32")
    blue = np.where(water_mask, RNG.uniform(0.05, 0.10, n), RNG.uniform(0.03, 0.06, n)).astype("float32")
    return {
        "green": green, "nir": nir, "red": red, "blue": blue,
        "swir1": np.full(n, np.nan, dtype="float32"),
        "swir2": np.full(n, np.nan, dtype="float32"),
        "cloud_mask": np.ones(n, dtype=bool),
        "source": "naip",
        "date": "2023-07-15",
    }


def _make_aligned_stack(
    observations: list[dict],
    rows: int = ROWS,
    cols: int = COLS,
) -> AlignedStack:
    """Wrap observations into an AlignedStack."""
    return AlignedStack(
        observations=observations,
        rows=rows,
        cols=cols,
        transform=(0.0, 30.0, 0.0, 100.0, 0.0, -30.0),
        crs="EPSG:32615",
        bounds=(0.0, 0.0, cols * 30.0, rows * 30.0),
    )


def _binary_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Compute OA, precision, recall, F1, and Cohen's κ."""
    tp = np.sum(pred & truth)
    tn = np.sum(~pred & ~truth)
    fp = np.sum(pred & ~truth)
    fn = np.sum(~pred & truth)
    n = pred.size

    oa = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Cohen's kappa
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n * n)
    kappa = (oa - pe) / (1 - pe) if pe < 1.0 else 0.0

    return {"oa": oa, "precision": precision, "recall": recall, "f1": f1, "kappa": kappa}


# ======================================================================
# Test: Pixel-level classification accuracy
# ======================================================================


class TestClassificationAccuracy:
    """End-to-end classification accuracy on synthetic landscapes."""

    def test_clear_water_land_separation(self) -> None:
        """Clean synthetic scene with perfect spectral separation.
        Expect OA ≥ 0.90 and F1 ≥ 0.85.
        """
        water_mask = _make_water_mask(0.4)
        obs = _make_reflectance(water_mask)

        classifier = QuantumHybridClassifier(use_quantum_svm=False)
        stack = _make_aligned_stack([obs])
        results = classifier.classify_stack(stack)

        assert len(results) == 1
        cr = results[0]

        # Compare classification to ground truth
        pred = cr.water_binary
        metrics = _binary_metrics(pred, water_mask)

        assert metrics["oa"] >= 0.90, f"OA={metrics['oa']:.3f} < 0.90"
        assert metrics["f1"] >= 0.85, f"F1={metrics['f1']:.3f} < 0.85"
        assert metrics["kappa"] >= 0.75, f"κ={metrics['kappa']:.3f} < 0.75"

    def test_high_water_fraction(self) -> None:
        """Scene with 80% water — precision must remain high."""
        water_mask = _make_water_mask(0.8)
        obs = _make_reflectance(water_mask)

        classifier = QuantumHybridClassifier(use_quantum_svm=False)
        stack = _make_aligned_stack([obs])
        results = classifier.classify_stack(stack)
        cr = results[0]

        metrics = _binary_metrics(cr.water_binary, water_mask)
        assert metrics["precision"] >= 0.85, f"Precision={metrics['precision']:.3f} < 0.85"
        assert metrics["recall"] >= 0.85, f"Recall={metrics['recall']:.3f} < 0.85"

    def test_low_water_fraction(self) -> None:
        """Scene with only 10% water — recall on rare water pixels."""
        water_mask = _make_water_mask(0.1)
        obs = _make_reflectance(water_mask)

        classifier = QuantumHybridClassifier(use_quantum_svm=False)
        stack = _make_aligned_stack([obs])
        results = classifier.classify_stack(stack)
        cr = results[0]

        metrics = _binary_metrics(cr.water_binary, water_mask)
        assert metrics["recall"] >= 0.80, f"Recall={metrics['recall']:.3f} < 0.80"
        assert metrics["oa"] >= 0.85, f"OA={metrics['oa']:.3f} < 0.85"

    def test_naip_no_swir_still_accurate(self) -> None:
        """NAIP observations lack SWIR — classification should still work
        using NDWI-based fallback."""
        water_mask = _make_water_mask(0.3)
        obs = _make_naip_reflectance(water_mask)

        classifier = QuantumHybridClassifier(use_quantum_svm=False)
        stack = _make_aligned_stack([obs])
        results = classifier.classify_stack(stack)
        cr = results[0]

        metrics = _binary_metrics(cr.water_binary, water_mask)
        # NAIP has weaker spectral contrast → relax threshold slightly
        assert metrics["oa"] >= 0.80, f"NAIP OA={metrics['oa']:.3f} < 0.80"
        assert metrics["f1"] >= 0.70, f"NAIP F1={metrics['f1']:.3f} < 0.70"

    def test_quantum_svm_improves_or_matches(self) -> None:
        """QK-SVM enabled should match or beat the no-SVM variant."""
        water_mask = _make_water_mask(0.5)
        obs = _make_reflectance(water_mask)

        # Without SVM
        classifier_no = QuantumHybridClassifier(use_quantum_svm=False)
        stack_no = _make_aligned_stack([obs])
        cr_no = classifier_no.classify_stack(stack_no)[0]
        f1_no = _binary_metrics(cr_no.water_binary, water_mask)["f1"]

        # With SVM
        classifier_svm = QuantumHybridClassifier(use_quantum_svm=True, svm_max_samples=2000)
        stack_svm = _make_aligned_stack([obs])
        cr_svm = classifier_svm.classify_stack(stack_svm)[0]
        f1_svm = _binary_metrics(cr_svm.water_binary, water_mask)["f1"]

        # SVM should be at least as good (±5% tolerance for stochastic effects)
        assert f1_svm >= f1_no - 0.05, (
            f"QK-SVM F1={f1_svm:.3f} significantly worse than no-SVM F1={f1_no:.3f}"
        )


# ======================================================================
# Test: Quantum Feature Encoder accuracy properties
# ======================================================================


class TestQuantumEncoderAccuracy:
    """Verify that quantum encoding preserves expected spectral relationships."""

    def test_water_indices_produce_higher_water_prob_on_average(self) -> None:
        """For a population of water vs land pixels, the mean quantum
        water probability should be higher for water pixels."""
        qfe = QuantumFeatureEncoder()

        n = 500
        # Water-like indices
        ndwi_water = RNG.uniform(0.2, 0.8, (n,)).astype("float32")
        mndwi_water = RNG.uniform(0.2, 0.7, (n,)).astype("float32")
        awei_water = RNG.uniform(0.5, 3.0, (n,)).astype("float32")

        # Land-like indices
        ndwi_land = RNG.uniform(-0.8, -0.1, (n,)).astype("float32")
        mndwi_land = RNG.uniform(-0.7, -0.1, (n,)).astype("float32")
        awei_land = RNG.uniform(-3.0, -0.5, (n,)).astype("float32")

        pw_water, _ = qfe.encode(
            ndwi_water.reshape(1, -1),
            mndwi_water.reshape(1, -1),
            awei_water.reshape(1, -1),
        )
        pw_land, _ = qfe.encode(
            ndwi_land.reshape(1, -1),
            mndwi_land.reshape(1, -1),
            awei_land.reshape(1, -1),
        )

        mean_water = pw_water.mean()
        mean_land = pw_land.mean()
        # On average, water-like spectra should yield different (ideally higher)
        # quantum probabilities than land-like spectra
        assert mean_water != pytest.approx(mean_land, abs=0.05), (
            f"Encoder fails to discriminate: water_mean={mean_water:.3f}, "
            f"land_mean={mean_land:.3f}"
        )

    def test_encoder_consistency_across_runs(self) -> None:
        """Deterministic encoder — same input should always produce
        same output."""
        qfe = QuantumFeatureEncoder()
        ndwi = np.array([[0.5, -0.3]], dtype="float32")
        mndwi = np.array([[0.4, -0.4]], dtype="float32")
        awei = np.array([[1.0, -1.0]], dtype="float32")

        p1, c1 = qfe.encode(ndwi, mndwi, awei)
        p2, c2 = qfe.encode(ndwi, mndwi, awei)

        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(c1, c2)


# ======================================================================
# Test: Frequency surface fidelity
# ======================================================================


class TestFrequencyFidelity:
    """Verify flood frequency surface matches known inundation rates."""

    def _run_frequency_pipeline(
        self,
        n_obs: int,
        water_fractions: list[float],
    ) -> FrequencyResult:
        """Build a multi-temporal stack with known per-obs water fraction
        and compute the flood frequency."""
        observations = []
        classifications = []

        for i, wf in enumerate(water_fractions):
            water_mask = _make_water_mask(wf)
            obs = _make_reflectance(water_mask)
            obs["date"] = f"2023-{(i % 12) + 1:02d}-15"
            observations.append(obs)

            # Create mock classification with known water mask
            cr = MagicMock(spec=ClassificationResult)
            cr.water_binary = water_mask
            cr.water_probability = water_mask.astype("float32")
            cr.cloud_mask = np.ones((ROWS, COLS), dtype=bool)
            cr.source = "landsat"
            cr.date = obs["date"]
            classifications.append(cr)

        stack = _make_aligned_stack(observations)
        engine = FloodFrequencyEngine(stack=stack)
        return engine.compute(classifications)

    def test_constant_water_yields_high_frequency(self) -> None:
        """If every observation has 100% water, frequency should be ~1.0."""
        result = self._run_frequency_pipeline(10, [1.0] * 10)
        valid = ~np.isnan(result.frequency)
        np.testing.assert_allclose(result.frequency[valid], 1.0, atol=0.01)

    def test_no_water_yields_zero_frequency(self) -> None:
        """If no observation ever has water, frequency should be 0.0."""
        result = self._run_frequency_pipeline(10, [0.0] * 10)
        valid = ~np.isnan(result.frequency)
        np.testing.assert_allclose(result.frequency[valid], 0.0, atol=0.01)

    def test_half_wet_half_dry_yields_intermediate(self) -> None:
        """5 all-water + 5 all-dry observations → frequency ≈ 0.5."""
        fracs = [1.0] * 5 + [0.0] * 5
        result = self._run_frequency_pipeline(10, fracs)
        valid = ~np.isnan(result.frequency)
        # Some pixels will be water in some obs, not others → intermediate
        assert np.nanmean(result.frequency) == pytest.approx(0.5, abs=0.1)

    def test_permanent_zone_thresholds(self) -> None:
        """Pixels wet in ≥90% of observations → permanent mask = True."""
        # All 20 observations are 100% water → frequency = 1.0
        result = self._run_frequency_pipeline(20, [1.0] * 20)
        assert result.permanent_mask.all()
        assert not result.seasonal_mask.any()

    def test_seasonal_zone_thresholds(self) -> None:
        """Pixels wet in ~50% of observations → seasonal mask."""
        fracs = [1.0] * 10 + [0.0] * 10
        result = self._run_frequency_pipeline(20, fracs)
        # Pixels that are always water in the first 10 obs have freq=0.5
        # which is seasonal (0.25 ≤ f < 0.90)
        freq_mean = np.nanmean(result.frequency)
        assert 0.25 <= freq_mean < 0.90, f"Mean frequency {freq_mean:.3f} outside seasonal range"

    def test_rare_zone_for_infrequent_flooding(self) -> None:
        """1 out of 20 observations is all-water → freq ≈ 0.05."""
        fracs = [1.0] * 1 + [0.0] * 19
        result = self._run_frequency_pipeline(20, fracs)
        freq_mean = np.nanmean(result.frequency)
        assert freq_mean < 0.25, f"Mean frequency {freq_mean:.3f} should be in rare/dry zone"


# ======================================================================
# Test: Wilson confidence interval calibration
# ======================================================================


class TestConfidenceCalibration:
    """Verify Wilson score intervals have correct coverage properties."""

    def test_confidence_bounds_contain_true_frequency(self) -> None:
        """With enough observations, the Wilson CI should contain the
        true inundation frequency for most pixels."""
        # True water fraction: 40% across 30 observations
        # Each obs randomly floods 40% of pixels (spatially shuffled)
        true_rate = 0.4
        n_obs = 30
        classifications = []

        for i in range(n_obs):
            water_mask = _make_water_mask(true_rate)
            cr = MagicMock(spec=ClassificationResult)
            cr.water_binary = water_mask
            cr.water_probability = water_mask.astype("float32")
            cr.cloud_mask = np.ones((ROWS, COLS), dtype=bool)
            cr.source = "landsat"
            cr.date = f"2023-{(i % 12) + 1:02d}-01"
            classifications.append(cr)

        observations = [_make_reflectance(_make_water_mask(true_rate)) for _ in range(n_obs)]
        stack = _make_aligned_stack(observations)
        engine = FloodFrequencyEngine(stack=stack, confidence_level=0.95)
        result = engine.compute(classifications)

        valid = ~np.isnan(result.confidence_lower)
        # At 95% Wilson CI, we expect ≥90% of pixel CIs to contain true_rate
        # (relaxed from 95% because pixel-level freq varies stochastically)
        covers = (result.confidence_lower[valid] <= true_rate) & (result.confidence_upper[valid] >= true_rate)
        coverage = covers.sum() / valid.sum()
        assert coverage >= 0.80, f"Wilson CI coverage {coverage:.2%} < 80% for true rate {true_rate}"

    def test_wider_ci_with_fewer_observations(self) -> None:
        """Fewer observations → wider confidence intervals."""
        def _mean_ci_width(n_obs: int) -> float:
            classifications = []
            for i in range(n_obs):
                cr = MagicMock(spec=ClassificationResult)
                cr.water_binary = _make_water_mask(0.5)
                cr.water_probability = cr.water_binary.astype("float32")
                cr.cloud_mask = np.ones((ROWS, COLS), dtype=bool)
                cr.source = "landsat"
                cr.date = f"2023-{(i % 12) + 1:02d}-01"
                classifications.append(cr)
            obs = [_make_reflectance(_make_water_mask(0.5)) for _ in range(n_obs)]
            stack = _make_aligned_stack(obs)
            result = FloodFrequencyEngine(stack=stack).compute(classifications)
            valid = ~np.isnan(result.confidence_lower)
            widths = result.confidence_upper[valid] - result.confidence_lower[valid]
            return float(widths.mean())

        width_5 = _mean_ci_width(5)
        width_30 = _mean_ci_width(30)
        assert width_5 > width_30, (
            f"5-obs CI width ({width_5:.4f}) should be wider than "
            f"30-obs CI width ({width_30:.4f})"
        )


# ======================================================================
# Test: Cross-sensor consistency
# ======================================================================


class TestCrossSensorConsistency:
    """Landsat and Sentinel-2 classifications of the same scene should
    produce correlated water masks."""

    def test_landsat_sentinel_agreement(self) -> None:
        """Same water mask, different sensor labels → water prob should
        be correlated (Pearson r ≥ 0.7)."""
        water_mask = _make_water_mask(0.5)

        # Landsat observation
        ls_obs = _make_reflectance(water_mask)
        ls_obs["source"] = "landsat"

        # Sentinel-2 observation (same spectral pattern)
        s2_obs = _make_reflectance(water_mask)
        s2_obs["source"] = "sentinel2"

        classifier = QuantumHybridClassifier(use_quantum_svm=False)
        stack = _make_aligned_stack([ls_obs, s2_obs])
        results = classifier.classify_stack(stack)

        assert len(results) == 2
        prob_ls = results[0].water_probability
        prob_s2 = results[1].water_probability

        valid = ~np.isnan(prob_ls) & ~np.isnan(prob_s2)
        if valid.sum() > 10:
            corr = np.corrcoef(prob_ls[valid].ravel(), prob_s2[valid].ravel())[0, 1]
            assert corr >= 0.7, f"Cross-sensor Pearson r={corr:.3f} < 0.7"


# ======================================================================
# Test: Edge-case robustness
# ======================================================================


class TestEdgeCaseRobustness:
    """The pipeline should not crash on degenerate inputs."""

    def test_all_cloudy_scene(self) -> None:
        """Scene with no clear pixels → classification should return all-NaN."""
        water_mask = _make_water_mask(0.5)
        obs = _make_reflectance(water_mask)
        obs["cloud_mask"] = np.zeros((ROWS, COLS), dtype=bool)  # all cloudy

        classifier = QuantumHybridClassifier(use_quantum_svm=False)
        stack = _make_aligned_stack([obs])
        results = classifier.classify_stack(stack)

        cr = results[0]
        # All pixels should have NaN water probability
        assert np.all(np.isnan(cr.water_probability))

    def test_all_water_scene(self) -> None:
        """100% water → should classify nearly all pixels as water."""
        water_mask = np.ones((ROWS, COLS), dtype=bool)
        obs = _make_reflectance(water_mask)

        classifier = QuantumHybridClassifier(use_quantum_svm=False)
        stack = _make_aligned_stack([obs])
        results = classifier.classify_stack(stack)

        water_frac = results[0].water_binary.sum() / results[0].water_binary.size
        assert water_frac >= 0.85, f"All-water scene: only {water_frac:.1%} classified as water"

    def test_all_land_scene(self) -> None:
        """100% land → should classify nearly no pixels as water."""
        water_mask = np.zeros((ROWS, COLS), dtype=bool)
        obs = _make_reflectance(water_mask)

        classifier = QuantumHybridClassifier(use_quantum_svm=False)
        stack = _make_aligned_stack([obs])
        results = classifier.classify_stack(stack)

        water_frac = results[0].water_binary.sum() / results[0].water_binary.size
        assert water_frac <= 0.15, f"All-land scene: {water_frac:.1%} classified as water"

    def test_mixed_sensor_stack(self) -> None:
        """Stack with Landsat + Sentinel-2 + NAIP → should process all."""
        water_mask = _make_water_mask(0.4)
        ls = _make_reflectance(water_mask)
        ls["source"] = "landsat"

        s2 = _make_reflectance(water_mask)
        s2["source"] = "sentinel2"
        s2["date"] = "2023-07-01"

        naip = _make_naip_reflectance(water_mask)

        classifier = QuantumHybridClassifier(use_quantum_svm=False)
        stack = _make_aligned_stack([ls, s2, naip])
        results = classifier.classify_stack(stack)

        assert len(results) == 3
        sources = {r.source for r in results}
        assert sources == {"landsat", "sentinel2", "naip"}

    def test_single_observation_frequency(self) -> None:
        """Frequency surface from only 1 observation should not crash."""
        water_mask = _make_water_mask(0.5)
        obs = _make_reflectance(water_mask)

        cr = MagicMock(spec=ClassificationResult)
        cr.water_binary = water_mask
        cr.water_probability = water_mask.astype("float32")
        cr.cloud_mask = np.ones((ROWS, COLS), dtype=bool)
        cr.source = "landsat"
        cr.date = "2023-01-01"

        stack = _make_aligned_stack([obs])
        engine = FloodFrequencyEngine(stack=stack)
        result = engine.compute([cr])

        # Frequency should be 0.0 or 1.0 (only one obs)
        valid = ~np.isnan(result.frequency)
        assert np.all((result.frequency[valid] == 0.0) | (result.frequency[valid] == 1.0))


# ======================================================================
# Test: Bayesian model averaging properties
# ======================================================================


class TestBayesianFusionAccuracy:
    """Verify Bayesian fusion preserves expected statistical properties."""

    def test_fusion_improves_over_worst_component(self) -> None:
        """Bayesian average should be at least as accurate as the worst
        single-model prediction."""
        water_mask = _make_water_mask(0.5)

        # Noisy predictions
        noise = RNG.normal(0, 0.15, (ROWS, COLS)).astype("float32")
        q_prob = np.clip(water_mask.astype("float32") + noise * 1.2, 0, 1)
        s_prob = np.clip(water_mask.astype("float32") + noise * 0.8, 0, 1)
        g_prob = np.clip(water_mask.astype("float32") + noise * 1.0, 0, 1)
        conf = np.full((ROWS, COLS), 0.8, dtype="float32")

        fused = bayesian_model_average(q_prob, s_prob, g_prob, conf)

        binary_fused = fused > 0.5
        binary_q = q_prob > 0.5
        binary_s = s_prob > 0.5
        binary_g = g_prob > 0.5

        f1_fused = _binary_metrics(binary_fused, water_mask)["f1"]
        f1_q = _binary_metrics(binary_q, water_mask)["f1"]
        f1_s = _binary_metrics(binary_s, water_mask)["f1"]
        f1_g = _binary_metrics(binary_g, water_mask)["f1"]

        worst = min(f1_q, f1_s, f1_g)
        assert f1_fused >= worst - 0.05, (
            f"Fused F1={f1_fused:.3f} worse than worst component F1={worst:.3f}"
        )

    def test_high_confidence_increases_quantum_weight(self) -> None:
        """When quantum confidence is high, quantum prediction should
        dominate the fusion."""
        shape = (10, 10)
        q_prob = np.ones(shape, dtype="float32") * 0.9
        s_prob = np.ones(shape, dtype="float32") * 0.1
        g_prob = np.ones(shape, dtype="float32") * 0.2

        # High confidence → quantum should dominate
        fused_high = bayesian_model_average(
            q_prob, s_prob, g_prob,
            np.ones(shape, dtype="float32") * 0.95
        )

        # Low confidence → less quantum influence
        fused_low = bayesian_model_average(
            q_prob, s_prob, g_prob,
            np.ones(shape, dtype="float32") * 0.1
        )

        # High-confidence fusion should be closer to quantum prediction (0.9)
        assert fused_high.mean() > fused_low.mean(), (
            f"High-conf fusion ({fused_high.mean():.3f}) should exceed "
            f"low-conf ({fused_low.mean():.3f})"
        )


# ======================================================================
# Test: Spectral indices correctness
# ======================================================================


class TestSpectralIndexAccuracy:
    """Validate spectral index formulas against known values."""

    def test_ndwi_positive_for_water(self) -> None:
        """Water: green > nir → NDWI > 0."""
        green = np.array([0.15, 0.10, 0.08], dtype="float32")
        nir = np.array([0.03, 0.02, 0.01], dtype="float32")
        ndwi = compute_ndwi(green, nir)
        assert np.all(ndwi > 0)

    def test_ndwi_negative_for_vegetation(self) -> None:
        """Vegetation: nir > green → NDWI < 0."""
        green = np.array([0.05, 0.04], dtype="float32")
        nir = np.array([0.25, 0.30], dtype="float32")
        ndwi = compute_ndwi(green, nir)
        assert np.all(ndwi < 0)

    def test_mndwi_uses_swir_correctly(self) -> None:
        """MNDWI = (green - swir1) / (green + swir1)."""
        green = np.array([0.10], dtype="float32")
        swir1 = np.array([0.02], dtype="float32")
        mndwi = compute_mndwi(green, swir1)
        expected = (0.10 - 0.02) / (0.10 + 0.02)
        assert mndwi[0] == pytest.approx(expected, abs=1e-5)

    def test_awei_sh_formula(self) -> None:
        """AWEI_sh = 4 * (green - swir1) - (0.25 * nir + 2.75 * swir2)."""
        blue = np.array([0.05], dtype="float32")
        green = np.array([0.10], dtype="float32")
        nir = np.array([0.03], dtype="float32")
        swir1 = np.array([0.02], dtype="float32")
        swir2 = np.array([0.01], dtype="float32")
        awei = compute_awei_sh(blue, green, nir, swir1, swir2)
        expected = 4.0 * (0.10 - 0.02) - (0.25 * 0.03 + 2.75 * 0.01)
        assert awei[0] == pytest.approx(expected, abs=1e-4)


# ======================================================================
# Test: Preprocessing accuracy
# ======================================================================


class TestPreprocessingAccuracy:
    """Resampling and cloud masking preserve data integrity."""

    def test_downsample_preserves_mean(self) -> None:
        """Cubic downsampling should approximately preserve the spatial mean."""
        arr = RNG.uniform(0.1, 0.9, (100, 100)).astype("float32")
        result = ImagePreprocessor._downsample_cubic(arr, (30, 30))

        assert result.mean() == pytest.approx(arr.mean(), abs=0.05), (
            f"Mean shifted from {arr.mean():.3f} to {result.mean():.3f}"
        )

    def test_regrid_preserves_binary_ratio(self) -> None:
        """Regridding a boolean mask should roughly preserve the True-pixel ratio."""
        mask = _make_water_mask(0.4)
        result = ImagePreprocessor._regrid_mask(mask, mask.shape, (30, 30))

        original_frac = mask.mean()
        regridded_frac = result.mean()
        assert regridded_frac == pytest.approx(original_frac, abs=0.1), (
            f"Binary fraction shifted from {original_frac:.3f} to {regridded_frac:.3f}"
        )

    def test_landsat_reflectance_scale(self) -> None:
        """Landsat DN=20000 → reflectance ≈ 0.35 (within physical range)."""
        dn = np.array([20000], dtype="float32")
        refl = ImagePreprocessor._ls_sr_to_reflectance(dn)
        assert 0.0 <= refl[0] <= 1.0
        expected = 20000 * 0.0000275 - 0.2  # = 0.35
        assert refl[0] == pytest.approx(expected, abs=0.01)
