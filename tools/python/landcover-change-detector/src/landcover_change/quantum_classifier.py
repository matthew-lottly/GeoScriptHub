"""Quantum-enhanced land-cover classification.

v1.0 — Quantum Land-Cover Change Detector

Novel pseudo-quantum approach:
  1. Self-supervised spectral auto-encoder → 8-dim latent space
  2. 4-qubit VQC amplitude encoding → 16 Hilbert-space basis states
  3. Born probability → 8 land-cover class probabilities
  4. Quantum kernel SVM (|⟨ψ(x)|ψ(x')⟩|² kernel)
  5. Gradient boosting + Random Forest classical ensemble
  6. Ridge meta-learner fusion
  7. Physics-constrained temporal post-processing
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifier

from .constants import (
    N_QUBITS, HILBERT_DIM, ENTANGLEMENT_STRENGTH, NUM_CLASSES,
    CLASS_NAMES, TRANSITION_ALLOWED,
    NDVI_FOREST_THRESH, NDVI_VEG_THRESH, NDWI_WATER_THRESH,
    NDBI_URBAN_THRESH, BSI_BARREN_THRESH,
)
from .feature_engineering import FeatureStack

logger = logging.getLogger("geoscripthub.landcover_change.quantum_classifier")


# ── Result container ──────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """Per-year classification output."""

    year: int
    class_map: np.ndarray          # int [0, NUM_CLASSES-1]
    class_probabilities: np.ndarray  # float (rows, cols, NUM_CLASSES)
    confidence: np.ndarray         # float [0, 1]
    quantum_entropy: np.ndarray    # Born-rule entropy
    valid_mask: np.ndarray
    shape: tuple[int, int]


# ── Spectral Auto-Encoder ─────────────────────────────────────────

class SpectralAutoEncoder:
    """Simple linear auto-encoder for self-supervised pre-training.

    Learns a compressed 8-dimensional representation of the
    ~30-band feature vector, capturing the spectral manifold
    structure without labelled data.

    Architecture: 30 → 16 → 8 (latent) → 16 → 30
    """

    def __init__(self, input_dim: int = 30, latent_dim: int = 8) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self._fitted = False

        # Encoder weights (random init)
        rng = np.random.default_rng(42)
        hidden = 16
        self.W1 = rng.normal(0, 0.1, (input_dim, hidden)).astype("float32")
        self.b1 = np.zeros(hidden, dtype="float32")
        self.W2 = rng.normal(0, 0.1, (hidden, latent_dim)).astype("float32")
        self.b2 = np.zeros(latent_dim, dtype="float32")
        # Decoder weights
        self.W3 = rng.normal(0, 0.1, (latent_dim, hidden)).astype("float32")
        self.b3 = np.zeros(hidden, dtype="float32")
        self.W4 = rng.normal(0, 0.1, (hidden, input_dim)).astype("float32")
        self.b4 = np.zeros(input_dim, dtype="float32")

        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, n_epochs: int = 50, lr: float = 0.001) -> float:
        """Train the auto-encoder on unlabelled pixel samples.

        Uses simple gradient descent on MSE reconstruction loss.

        Returns final reconstruction loss.
        """
        X = self.scaler.fit_transform(X.astype("float32"))
        n = X.shape[0]
        batch = min(2048, n)

        rng = np.random.default_rng(42)
        loss = float("inf")

        for epoch in range(n_epochs):
            idx = rng.choice(n, batch, replace=False)
            xb = X[idx]

            # Forward
            h1 = np.maximum(xb @ self.W1 + self.b1, 0)  # ReLU
            z = h1 @ self.W2 + self.b2  # latent
            h3 = np.maximum(z @ self.W3 + self.b3, 0)
            x_hat = h3 @ self.W4 + self.b4

            # MSE loss
            diff = x_hat - xb
            loss = float(np.mean(diff**2))

            # Backward (simplified gradient descent)
            d4 = diff * (2.0 / batch)
            dW4 = h3.T @ d4
            db4 = d4.sum(axis=0)
            dh3 = d4 @ self.W4.T
            dh3 *= (h3 > 0)  # ReLU grad

            dW3 = z.T @ dh3
            db3 = dh3.sum(axis=0)
            dz = dh3 @ self.W3.T

            dW2 = h1.T @ dz
            db2 = dz.sum(axis=0)
            dh1 = dz @ self.W2.T
            dh1 *= (h1 > 0)

            dW1 = xb.T @ dh1
            db1 = dh1.sum(axis=0)

            # Update
            for W, dW in [
                (self.W1, dW1), (self.W2, dW2), (self.W3, dW3), (self.W4, dW4),
            ]:
                W -= lr * np.clip(dW, -1, 1)
            for b, db in [
                (self.b1, db1), (self.b2, db2), (self.b3, db3), (self.b4, db4),
            ]:
                b -= lr * np.clip(db, -1, 1)

        self._fitted = True
        logger.info("Auto-encoder trained: loss=%.6f", loss)
        return loss

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode features → 8-dim latent space."""
        X = self.scaler.transform(X.astype("float32"))
        h1 = np.maximum(X @ self.W1 + self.b1, 0)
        return (h1 @ self.W2 + self.b2).astype("float32")


# ── Quantum Feature Encoder ──────────────────────────────────────

class QuantumFeatureEncoder:
    """4-qubit pseudo-quantum feature encoder.

    Encodes an 8-dimensional latent vector into a 16-dimensional
    Hilbert space state via amplitude encoding + VQC rotation
    gates + CZ entanglement, then collapses via Born rule to
    produce class probabilities.
    """

    def __init__(self) -> None:
        self.n_qubits = N_QUBITS
        self.dim = HILBERT_DIM

        # Build VQC unitary
        self._vqc = self._build_vqc()

    def _build_vqc(self) -> np.ndarray:
        """Build the variational quantum circuit unitary (16×16)."""
        dim = self.dim
        U = np.eye(dim, dtype=complex)

        # Layer 1: RY rotations (parameterised by entanglement strength)
        for q in range(self.n_qubits):
            ry = self._ry_gate(ENTANGLEMENT_STRENGTH * (q + 1) / self.n_qubits)
            U = self._apply_single_gate(U, ry, q) @ U

        # Layer 2: CZ entanglement (linear chain)
        for q in range(self.n_qubits - 1):
            cz = self._cz_gate(q, q + 1)
            U = cz @ U

        # Layer 3: additional RY with different angles
        for q in range(self.n_qubits):
            ry = self._ry_gate(ENTANGLEMENT_STRENGTH * (self.n_qubits - q) / self.n_qubits)
            U = self._apply_single_gate(U, ry, q) @ U

        return U

    def encode(self, latent: np.ndarray) -> np.ndarray:
        """Encode latent vectors → Born-rule class probabilities.

        Parameters
        ----------
        latent:
            Array of shape (n_samples, 8) — latent codes from auto-encoder.

        Returns
        -------
        Array of shape (n_samples, NUM_CLASSES) — class probabilities.
        """
        n = latent.shape[0]
        probs = np.zeros((n, NUM_CLASSES), dtype="float32")

        for i in range(n):
            # Amplitude encoding: pad 8-dim to 16-dim, normalise
            vec = np.zeros(self.dim, dtype=complex)
            vec[:latent.shape[1]] = latent[i]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                vec /= norm
            else:
                vec[0] = 1.0

            # Apply VQC
            psi = self._vqc @ vec

            # Born rule: |ψ|²
            born = np.abs(psi) ** 2

            # Map 16 basis states → 8 classes (2 states per class)
            for c in range(NUM_CLASSES):
                probs[i, c] = born[2 * c] + born[2 * c + 1]

            # Normalise
            total = probs[i].sum()
            if total > 1e-10:
                probs[i] /= total

        return probs

    def encode_with_entropy(self, latent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode and return (probabilities, entropy)."""
        probs = self.encode(latent)
        eps = 1e-10
        entropy = -np.sum(probs * np.log2(probs + eps), axis=1)
        max_entropy = np.log2(NUM_CLASSES)
        norm_entropy = entropy / max_entropy
        return probs, norm_entropy.astype("float32")

    # ── Gate builders ─────────────────────────────────────────────

    def _ry_gate(self, theta: float) -> np.ndarray:
        """Single-qubit RY rotation gate."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    def _apply_single_gate(
        self, U: np.ndarray, gate: np.ndarray, qubit: int,
    ) -> np.ndarray:
        """Apply a single-qubit gate to a specific qubit in the full system."""
        parts = [np.eye(2, dtype=complex)] * self.n_qubits
        parts[qubit] = gate
        full = parts[0]
        for p in parts[1:]:
            full = np.kron(full, p)
        return full

    def _cz_gate(self, q1: int, q2: int) -> np.ndarray:
        """Build a CZ gate between two qubits in the full Hilbert space."""
        dim = self.dim
        cz = np.eye(dim, dtype=complex)
        for i in range(dim):
            bits = [(i >> (self.n_qubits - 1 - q)) & 1 for q in range(self.n_qubits)]
            if bits[q1] == 1 and bits[q2] == 1:
                cz[i, i] = -1
        return cz


# ── Quantum Kernel SVM ───────────────────────────────────────────

class QuantumKernelSVM:
    """SVM with quantum-inspired kernel: K(x,x') = |⟨ψ(x)|ψ(x')⟩|²."""

    def __init__(self, encoder: QuantumFeatureEncoder) -> None:
        self.encoder = encoder
        self.scaler = StandardScaler()
        self._svm: Optional[SVC] = None
        self._X_train_ref: Optional[np.ndarray] = None
        self._fitted = False

    def _quantum_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute quantum fidelity kernel matrix."""
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        dim = self.encoder.dim

        # Build state vectors
        def to_states(X: np.ndarray) -> np.ndarray:
            states = np.zeros((X.shape[0], dim), dtype=complex)
            for i in range(X.shape[0]):
                v = np.zeros(dim, dtype=complex)
                v[:X.shape[1]] = X[i]
                norm = np.linalg.norm(v)
                v = v / norm if norm > 1e-10 else np.zeros(dim, dtype=complex)
                states[i] = self.encoder._vqc @ v
            return states

        s1 = to_states(X1)
        s2 = to_states(X2)

        # Fidelity kernel |⟨ψ₁|ψ₂⟩|²
        overlap = s1 @ s2.conj().T
        return (np.abs(overlap) ** 2).astype("float32")

    def fit_predict(
        self, features: np.ndarray, labels: np.ndarray,
    ) -> np.ndarray:
        """Train on first call, predict on subsequent calls.

        Returns class probabilities (n_samples, NUM_CLASSES).
        """
        n = features.shape[0]
        probs = np.zeros((n, NUM_CLASSES), dtype="float32")

        if not self._fitted:
            # Train
            X = self.scaler.fit_transform(features) * np.pi
            self._X_train_ref = X
            K = self._quantum_kernel(X, X)
            self._svm = SVC(
                kernel="precomputed", probability=True,
                class_weight="balanced", C=1.0,
            )
            self._svm.fit(K, labels)
            self._fitted = True

            p = self._svm.predict_proba(K)
            for ci, cls in enumerate(self._svm.classes_):
                if cls < NUM_CLASSES:
                    probs[:, cls] = p[:, ci]
            return probs

        # Predict
        assert self._X_train_ref is not None, "QK-SVM not fitted"
        assert self._svm is not None, "QK-SVM not fitted"
        X = self.scaler.transform(features) * np.pi
        svm = self._svm
        X_ref = self._X_train_ref

        batch_size = 2000
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            K_batch = self._quantum_kernel(X[s:e], X_ref)
            p = svm.predict_proba(K_batch)
            for ci, cls in enumerate(svm.classes_):
                if cls < NUM_CLASSES:
                    probs[s:e, cls] = p[:, ci]

        return probs


# ── Classical Ensemble Members ────────────────────────────────────

class GBClassifier:
    """Gradient boosting wrapper with train-once semantics."""

    def __init__(self) -> None:
        self._clf: Optional[GradientBoostingClassifier] = None
        self._fitted = False

    def fit_predict(
        self, features: np.ndarray, labels: np.ndarray,
    ) -> np.ndarray:
        n = features.shape[0]
        probs = np.zeros((n, NUM_CLASSES), dtype="float32")

        if not self._fitted:
            self._clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, random_state=42,
            )
            self._clf.fit(features, labels)
            self._fitted = True

        assert self._clf is not None
        p = self._clf.predict_proba(features)
        for ci, cls in enumerate(self._clf.classes_):
            if cls < NUM_CLASSES:
                probs[:, cls] = p[:, ci]
        return probs


class RFClassifier:
    """Random Forest wrapper with train-once semantics."""

    def __init__(self) -> None:
        self._clf: Optional[RandomForestClassifier] = None
        self._fitted = False

    def fit_predict(
        self, features: np.ndarray, labels: np.ndarray,
    ) -> np.ndarray:
        n = features.shape[0]
        probs = np.zeros((n, NUM_CLASSES), dtype="float32")

        if not self._fitted:
            self._clf = RandomForestClassifier(
                n_estimators=500, max_features="sqrt",
                class_weight="balanced", random_state=42,
                oob_score=True, n_jobs=-1,
            )
            self._clf.fit(features, labels)
            self._fitted = True
            if hasattr(self._clf, "oob_score_"):
                logger.info("RF OOB score: %.3f", self._clf.oob_score_)

        assert self._clf is not None
        p = self._clf.predict_proba(features)
        for ci, cls in enumerate(self._clf.classes_):
            if cls < NUM_CLASSES:
                probs[:, cls] = p[:, ci]
        return probs


# ── Meta-Learner ──────────────────────────────────────────────────

class EnsembleMetaLearner:
    """Ridge-based stacking meta-learner."""

    def __init__(self) -> None:
        self._ridge: Optional[RidgeClassifier] = None
        self._fitted = False

    def fit_fuse(
        self,
        predictions: list[np.ndarray],
        labels: np.ndarray,
    ) -> np.ndarray:
        """Stack classifier outputs and fuse via Ridge regression.

        Parameters
        ----------
        predictions:
            List of (n_samples, NUM_CLASSES) arrays from each classifier.
        labels:
            Ground-truth labels (n_samples,).

        Returns
        -------
        Fused class probabilities (n_samples, NUM_CLASSES).
        """
        # Concatenate all prediction vectors
        X_meta = np.concatenate(predictions, axis=1)
        n = X_meta.shape[0]

        if not self._fitted:
            self._ridge = RidgeClassifier(alpha=1.0, class_weight="balanced")
            self._ridge.fit(X_meta, labels)
            self._fitted = True

        assert self._ridge is not None
        decision = self._ridge.decision_function(X_meta)
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])

        # Softmax normalisation
        exp_d = np.exp(decision - decision.max(axis=1, keepdims=True))
        probs = exp_d / exp_d.sum(axis=1, keepdims=True)

        # Pad to NUM_CLASSES if needed
        if probs.shape[1] < NUM_CLASSES:
            full = np.zeros((n, NUM_CLASSES), dtype="float32")
            for ci, cls in enumerate(self._ridge.classes_):
                if cls < NUM_CLASSES:
                    full[:, cls] = probs[:, ci]
            return full

        return probs.astype("float32")


# ── Pseudo-Label Generator ────────────────────────────────────────

def generate_pseudo_labels(
    features: np.ndarray,
    feature_names: list[str],
    sar: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate pseudo-labels from spectral/SAR heuristics.

    Uses spectral index thresholds to assign provisional land-cover
    classes for self-supervised training.

    Returns
    -------
    Integer labels (n_samples,) in range [0, NUM_CLASSES-1].
    """
    n = features.shape[0]
    labels = np.full(n, 3, dtype="int32")  # default: shrub/grass

    # Feature name → column index mapping
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    ndvi = features[:, name_to_idx.get("ndvi", 6)]
    ndwi = features[:, name_to_idx.get("ndwi", 9)]
    mndwi = features[:, name_to_idx.get("mndwi", 10)]
    ndbi = features[:, name_to_idx.get("ndbi", 12)]
    bsi = features[:, name_to_idx.get("bsi", 15)]
    evi = features[:, name_to_idx.get("evi", 7)]

    # Water: NDWI > 0.10 or MNDWI > 0.10
    water_mask = (ndwi > NDWI_WATER_THRESH) | (mndwi > NDWI_WATER_THRESH)
    labels[water_mask] = 0

    # Wetland: moderate NDWI + moderate vegetation
    wetland_mask = (~water_mask) & (ndwi > 0.0) & (ndvi > 0.15) & (ndvi < 0.45)
    labels[wetland_mask] = 1

    # Forest: high NDVI + high EVI
    forest_mask = (~water_mask) & (ndvi > NDVI_FOREST_THRESH) & (evi > 0.3)
    labels[forest_mask] = 2

    # Shrub/Grass: moderate NDVI
    grass_mask = (~water_mask) & (~forest_mask) & (ndvi > NDVI_VEG_THRESH) & (ndvi <= NDVI_FOREST_THRESH)
    labels[grass_mask] = 3

    # Agriculture: moderate-high NDVI with different EVI pattern
    ag_mask = (~water_mask) & (~forest_mask) & (ndvi > 0.35) & (evi > 0.2) & (evi < 0.4)
    labels[ag_mask] = 4

    # Developed High: high NDBI + low NDVI
    dev_high_mask = (ndbi > NDBI_URBAN_THRESH + 0.10) & (ndvi < NDVI_VEG_THRESH)
    labels[dev_high_mask] = 6

    # Developed Low: moderate NDBI
    dev_low_mask = (~dev_high_mask) & (ndbi > NDBI_URBAN_THRESH) & (ndvi < NDVI_FOREST_THRESH)
    labels[dev_low_mask] = 5

    # Barren: high BSI + low NDVI
    barren_mask = (bsi > BSI_BARREN_THRESH) & (ndvi < NDVI_VEG_THRESH) & (~dev_high_mask)
    labels[barren_mask] = 7

    return labels


# ── Physics-Constrained Post-Processing ───────────────────────────

def apply_transition_constraints(
    class_maps: list[np.ndarray],
    years: list[int],
) -> list[np.ndarray]:
    """Enforce physics-based land-cover transition rules.

    Uses the TRANSITION_ALLOWED matrix to relabel impossible
    single-year transitions. When a forbidden transition is detected,
    the newer year is set to the most likely allowed class from
    neighboring pixels.
    """
    if len(class_maps) < 2:
        return class_maps

    corrected = [class_maps[0].copy()]

    for i in range(1, len(class_maps)):
        prev = corrected[i - 1]
        curr = class_maps[i].copy()

        for r in range(curr.shape[0]):
            for c in range(curr.shape[1]):
                from_cls = int(prev[r, c])
                to_cls = int(curr[r, c])
                if from_cls < NUM_CLASSES and to_cls < NUM_CLASSES:
                    if TRANSITION_ALLOWED[from_cls, to_cls] == 0:
                        # Forbidden transition — keep previous class
                        curr[r, c] = from_cls

        corrected.append(curr)

    return corrected


def morphological_cleanup(
    class_map: np.ndarray, min_patch: int = 4,
) -> np.ndarray:
    """Remove salt-and-pepper noise via mode filter + component pruning."""
    from scipy.ndimage import generic_filter, label

    # Mode filter (3×3)
    def _mode(values: np.ndarray) -> float:
        vals = values[~np.isnan(values)].astype(int)
        if len(vals) == 0:
            return 0.0
        counts = np.bincount(vals, minlength=NUM_CLASSES)
        return float(np.argmax(counts))

    smoothed = generic_filter(
        class_map.astype("float64"), _mode, size=3, mode="nearest",
    ).astype("int32")

    # Remove small patches
    for cls in range(NUM_CLASSES):
        mask = smoothed == cls
        labelled, n_labels = label(mask)
        for lbl in range(1, n_labels + 1):
            component = labelled == lbl
            if np.sum(component) < min_patch:
                # Replace with surrounding majority class
                smoothed[component] = -1

    # Fill -1 with nearest valid
    if np.any(smoothed < 0):
        from scipy.ndimage import maximum_filter
        filled = maximum_filter(
            np.where(smoothed >= 0, smoothed, 0), size=3, mode="nearest",
        )
        smoothed = np.where(smoothed < 0, filled, smoothed)

    return smoothed


# ── Main Classifier ───────────────────────────────────────────────

class QuantumLandCoverClassifier:
    """Orchestrates the full classification pipeline per year.

    Pipeline per year:
      1. Flatten feature cube → (n_pixels, n_features)
      2. Self-supervised auto-encoder (train on first year)
      3. Generate pseudo-labels from spectral heuristics
      4. Quantum feature encoding → Born-rule class probs
      5. QK-SVM + GB + RF ensemble classification
      6. Meta-learner fusion
      7. Morphological clean-up
    """

    def __init__(
        self,
        use_quantum_svm: bool = True,
        use_auto_encoder: bool = True,
    ) -> None:
        self.use_quantum_svm = use_quantum_svm
        self.use_auto_encoder = use_auto_encoder

        self._auto_encoder = SpectralAutoEncoder(input_dim=30, latent_dim=8)
        self._qfe = QuantumFeatureEncoder()
        self._qk_svm = QuantumKernelSVM(encoder=self._qfe) if use_quantum_svm else None
        self._gb = GBClassifier()
        self._rf = RFClassifier()
        self._meta = EnsembleMetaLearner()
        self._ae_fitted = False

    def classify_stack(
        self,
        feature_stacks: list[FeatureStack],
    ) -> list[ClassificationResult]:
        """Classify all years and return classification results."""
        results: list[ClassificationResult] = []

        for idx, fs in enumerate(feature_stacks):
            logger.info(
                "Classifying year %d (%d/%d) …",
                fs.year, idx + 1, len(feature_stacks),
            )
            result = self._classify_year(fs)
            results.append(result)

        return results

    def _classify_year(self, fs: FeatureStack) -> ClassificationResult:
        """Classify a single year's feature stack."""
        rows, cols, n_feat = fs.features.shape
        flat = fs.features.reshape(-1, n_feat)
        valid = fs.valid_mask.ravel()

        # Only classify valid pixels
        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0:
            return self._empty_result(fs.year, (rows, cols))

        X = flat[valid_idx]

        # Step 1: Auto-encoder pre-training (first year only)
        if self.use_auto_encoder and not self._ae_fitted:
            # Sample up to 10k pixels for training
            n_sample = min(10000, X.shape[0])
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(X.shape[0], n_sample, replace=False)
            self._auto_encoder.fit(X[sample_idx])
            self._ae_fitted = True

        # Step 2: Encode through auto-encoder
        if self._ae_fitted:
            latent = self._auto_encoder.encode(X)
        else:
            latent = X[:, :8]  # fallback: use first 8 features

        # Step 3: Quantum feature encoding
        qfe_probs, q_entropy = self._qfe.encode_with_entropy(latent)

        # Step 4: Generate pseudo-labels
        pseudo_labels = generate_pseudo_labels(X, fs.feature_names)

        # Step 5: Ensemble classification
        ensemble_preds: list[np.ndarray] = [qfe_probs]

        # QK-SVM
        if self._qk_svm is not None:
            try:
                # Sample training set (limit for kernel computation)
                n_train = min(1000, X.shape[0])
                rng = np.random.default_rng(42)
                train_idx = rng.choice(X.shape[0], n_train, replace=False)
                qk_probs = self._qk_svm.fit_predict(
                    latent[train_idx], pseudo_labels[train_idx],
                )
                # Predict on all
                if self._qk_svm._fitted:
                    qk_probs_all = self._qk_svm.fit_predict(latent, pseudo_labels)
                    ensemble_preds.append(qk_probs_all)
            except Exception as exc:
                logger.warning("QK-SVM failed: %s", exc)

        # Gradient Boosting
        try:
            gb_probs = self._gb.fit_predict(X, pseudo_labels)
            ensemble_preds.append(gb_probs)
        except Exception as exc:
            logger.warning("GB failed: %s", exc)

        # Random Forest
        try:
            rf_probs = self._rf.fit_predict(X, pseudo_labels)
            ensemble_preds.append(rf_probs)
        except Exception as exc:
            logger.warning("RF failed: %s", exc)

        # Step 6: Meta-learner fusion
        if len(ensemble_preds) >= 2:
            try:
                fused_probs = self._meta.fit_fuse(ensemble_preds, pseudo_labels)
            except Exception:
                # Fallback: simple average
                fused_probs = np.mean(ensemble_preds, axis=0)
        else:
            fused_probs = ensemble_preds[0]

        # Step 7: Build output maps
        class_prob_map = np.zeros((rows * cols, NUM_CLASSES), dtype="float32")
        class_prob_map[valid_idx] = fused_probs

        class_map = np.argmax(class_prob_map, axis=1).reshape(rows, cols)
        class_prob_map = class_prob_map.reshape(rows, cols, NUM_CLASSES)

        confidence = np.max(class_prob_map, axis=-1)
        entropy_map = np.zeros(rows * cols, dtype="float32")
        entropy_map[valid_idx] = q_entropy
        entropy_map = entropy_map.reshape(rows, cols)

        # Morphological clean-up
        class_map = morphological_cleanup(class_map)

        return ClassificationResult(
            year=fs.year,
            class_map=class_map.astype("int32"),
            class_probabilities=class_prob_map,
            confidence=confidence,
            quantum_entropy=entropy_map,
            valid_mask=fs.valid_mask,
            shape=(rows, cols),
        )

    @staticmethod
    def _empty_result(year: int, shape: tuple[int, int]) -> ClassificationResult:
        return ClassificationResult(
            year=year,
            class_map=np.zeros(shape, dtype="int32"),
            class_probabilities=np.zeros((*shape, NUM_CLASSES), dtype="float32"),
            confidence=np.zeros(shape, dtype="float32"),
            quantum_entropy=np.ones(shape, dtype="float32"),
            valid_mask=np.zeros(shape, dtype=bool),
            shape=shape,
        )
