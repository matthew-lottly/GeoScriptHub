"""test_change_detection.py — Tests for ChangeDetector."""
from __future__ import annotations

import numpy as np
import pytest

from deep_fusion_landcover.change_detection import ChangeDetector, TransitionMatrix
from deep_fusion_landcover.constants import NUM_CLASSES


@pytest.fixture(scope="module")
def detector() -> ChangeDetector:
    return ChangeDetector()


@pytest.fixture(scope="module")
def toy_change_result(detector, small_class_maps):
    years = [1990, 2000, 2010, 2020]
    return detector.analyze(small_class_maps, years)


class TestTransitionMatrix:
    def test_matrix_shape(self, small_class_maps):
        tm = ChangeDetector._build_transition_matrix(
            small_class_maps[0], small_class_maps[1], 1990, 2000
        )
        assert tm.matrix.shape == (NUM_CLASSES, NUM_CLASSES)

    def test_matrix_non_negative(self, small_class_maps):
        tm = ChangeDetector._build_transition_matrix(
            small_class_maps[0], small_class_maps[1], 1990, 2000
        )
        assert (tm.matrix >= 0).all()

    def test_to_dataframe_has_labels(self, small_class_maps):
        tm = ChangeDetector._build_transition_matrix(
            small_class_maps[0], small_class_maps[1], 1990, 2000
        )
        df = tm.to_dataframe()
        assert df.shape == (NUM_CLASSES, NUM_CLASSES)
        assert df.index[0].startswith("from_")

    def test_net_change_sums_zero(self, small_class_maps):
        """Total net change across all classes = 0 (what is gained is lost elsewhere)."""
        tm = ChangeDetector._build_transition_matrix(
            small_class_maps[0], small_class_maps[1], 1990, 2000
        )
        net = tm.net_change()
        assert abs(net.sum()) < 1e-6


class TestChangeDetector:
    def test_n_transitions(self, toy_change_result, small_class_maps):
        assert len(toy_change_result.transitions) == len(small_class_maps) - 1

    def test_trajectory_map_shape(self, toy_change_result, small_class_maps):
        H, W = small_class_maps[0].shape
        assert toy_change_result.trajectory_map.shape == (H, W)

    def test_hotspot_map_dtype(self, toy_change_result, small_class_maps):
        H, W = small_class_maps[0].shape
        assert toy_change_result.hotspot_map.shape == (H, W)
        assert toy_change_result.hotspot_map.dtype == bool

    def test_annual_fractions_shape(self, toy_change_result, small_class_maps):
        n_years = len(small_class_maps)
        df = toy_change_result.annual_fractions
        assert df.shape[0] == n_years
        assert df.shape[1] == NUM_CLASSES

    def test_annual_fractions_sum_to_one(self, toy_change_result):
        row_sums = toy_change_result.annual_fractions.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-5)

    def test_intensity_df_columns(self, toy_change_result):
        cols = set(toy_change_result.intensity_df.columns)
        for expected in ("from_year", "to_year", "total_change_ha", "change_rate_pct"):
            assert expected in cols

    def test_save_transitions_creates_files(self, detector, toy_change_result, tmp_path):
        detector.save_transitions(toy_change_result, tmp_path)
        csvs = list((tmp_path / "transitions").glob("*.csv"))
        assert len(csvs) == len(toy_change_result.transitions)

    def test_save_summary_creates_files(self, detector, toy_change_result, tmp_path):
        detector.save_summary(toy_change_result, tmp_path)
        assert (tmp_path / "annual_class_fractions.csv").exists()
        assert (tmp_path / "annual_intensity.csv").exists()
        assert (tmp_path / "urban_logistic.json").exists()
