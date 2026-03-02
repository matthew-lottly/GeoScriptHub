"""test_quantum_classifier.py — Tests for QuantumVQCClassifier."""
from __future__ import annotations

import numpy as np
import pytest

from deep_fusion_landcover.quantum_classifier import QuantumVQCClassifier, VQCircuit
from deep_fusion_landcover.constants import NUM_CLASSES


class TestVQCircuit:
    def test_forward_shape(self):
        """VQCircuit forward pass must return (n_qubits,) measurement vector."""
        n_q = 4
        circuit = VQCircuit(n_qubits=n_q, n_layers=2)
        x = np.zeros(n_q)
        result = circuit.forward(x)
        assert result.shape == (n_q,)

    def test_forward_values_bounded(self):
        """Born-rule measurements must be in [-1, 1]."""
        circuit = VQCircuit(n_qubits=4, n_layers=2)
        rng = np.random.default_rng(7)
        for _ in range(10):
            x = rng.uniform(0, np.pi, size=4)
            result = circuit.forward(x)
            assert np.all(result >= -1.0) and np.all(result <= 1.0)

    def test_different_params_different_output(self):
        """Two circuits with different θ should (very likely) give different output."""
        rng = np.random.default_rng(99)
        c1 = VQCircuit(n_qubits=4, n_layers=2)
        c2 = VQCircuit(n_qubits=4, n_layers=2)
        c2.theta = rng.uniform(0, 2 * np.pi, c2.theta.shape)
        x = rng.uniform(0, np.pi, size=4)
        assert not np.allclose(c1.forward(x), c2.forward(x), atol=1e-6)


class TestQuantumVQCClassifier:
    def test_fit_predict_shape(self, small_feature_array, small_labels):
        clf = QuantumVQCClassifier(use_qk_svm=False)
        clf.fit(small_feature_array, small_labels)

        feature_stack = small_feature_array.reshape(20, 10, -1)
        result = clf.predict(feature_stack)

        assert result.class_map.shape == (20, 10)
        assert result.class_map.min() >= 0
        assert result.class_map.max() < NUM_CLASSES

    def test_predict_proba_sums_to_one(self, small_feature_array, small_labels):
        clf = QuantumVQCClassifier(use_qk_svm=False)
        clf.fit(small_feature_array, small_labels)

        feature_stack = small_feature_array.reshape(20, 10, -1)
        result = clf.predict(feature_stack)
        proba = result.class_probs.reshape(-1, NUM_CLASSES)

        assert proba.shape[1] == NUM_CLASSES
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_save_load_roundtrip(self, tmp_path, small_feature_array, small_labels):
        clf = QuantumVQCClassifier(use_qk_svm=False)
        clf.fit(small_feature_array, small_labels)

        feature_stack = small_feature_array.reshape(20, 10, -1)
        proba_before = clf.predict(feature_stack).class_probs.reshape(-1, NUM_CLASSES)

        clf.save(tmp_path / "quantum_params.npz")

        clf2 = QuantumVQCClassifier(use_qk_svm=False)
        clf2.fit(small_feature_array, small_labels)  # required to init scaler
        clf2.load(tmp_path / "quantum_params.npz")
        proba_after = clf2.predict(feature_stack).class_probs.reshape(-1, NUM_CLASSES)

        np.testing.assert_allclose(proba_before, proba_after, atol=1e-4)
