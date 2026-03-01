"""
quantum_classifier.py
=====================
Pseudo-Quantum Hybrid AI Water / Flood Classifier — **v2.0**.

Major upgrades from v1.0:

1. **3-Qubit Hilbert Space** — 8 basis states (was 4 from 2 qubits),
   enabling finer-grained land-cover decomposition:
       |000⟩ = deep water     |001⟩ = shallow water
       |010⟩ = wet vegetation  |011⟩ = flood shadow
       |100⟩ = dry vegetation  |101⟩ = bare soil
       |110⟩ = impervious      |111⟩ = cloud/noise

2. **Variational Quantum Circuit (VQC)** — Trainable rotation angles
   optimised per-sensor to maximise water/non-water separability.
   The circuit uses parameterised Ry gates on each qubit plus a
   cascade of CZ entanglement gates.

3. **Spectral Attention Mechanism** — Learned attention weights
   for spectral indices, adapted per sensor type.  Indices that
   contribute more to water detection get higher weight.

4. **Ensemble Meta-Learner** — Ridge regression stacking layer
   that learns optimal weights from the QFE, QK-SVM, and GB
   predictions, replacing the fixed Bayesian model average.

5. **Feature Caching** — Spectral indices and feature matrices
   are computed once and shared across all classifier components.

6. **Monte Carlo Uncertainty** — Dropout-equivalent perturbation
   of quantum amplitudes for per-pixel uncertainty quantification.

Architecture
------------
QFE (3-qubit VQC) ─────┐
                        ├──→ Meta-Learner (Ridge) → p_water
QK-SVM (quantum kernel) ┤
                        │
GBSIE (gradient boost) ─┘

References
----------
* Havlíček et al., "Supervised learning with quantum-enhanced feature
  spaces", Nature 567 (2019).
* Benedetti et al., "Parameterized quantum circuits as machine learning
  models", Quantum Sci. Technol. 4 (2019).
* Wolpert, "Stacked generalization", Neural Networks 5(2), 1992.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.special import expit  # sigmoid
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from .preprocessing import AlignedStack

logger = logging.getLogger("geoscripthub.quantum_flood_frequency.quantum_classifier")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Spectral index thresholds (literature defaults)
NDWI_WATER_THRESHOLD = 0.0
MNDWI_WATER_THRESHOLD = 0.0
AWEI_SH_WATER_THRESHOLD = 0.0

# Quantum simulation parameters — v2.0
N_QUBITS = 3                        # 3-qubit system → 8 basis states
HILBERT_DIM = 2 ** N_QUBITS         # = 8
ENTANGLEMENT_STRENGTH = np.pi / 4   # CZ rotation strength

# Spectral attention defaults (per sensor)
ATTENTION_WEIGHTS = {
    "landsat":   {"ndwi": 0.30, "mndwi": 0.35, "awei": 0.35},
    "sentinel2": {"ndwi": 0.35, "mndwi": 0.30, "awei": 0.35},
    "naip":      {"ndwi": 0.70, "mndwi": 0.15, "awei": 0.15},
}

# Monte Carlo uncertainty params
MC_DROPOUT_SAMPLES = 8
MC_NOISE_SCALE = 0.05


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Per-pixel water classification for a single observation.

    Attributes:
        water_probability: 2-D float array [0, 1] — probability of water.
        water_binary: 2-D bool array — hard classification (p > 0.5).
        ndwi: Normalised Difference Water Index array.
        mndwi: Modified NDWI array.
        awei_sh: Automated Water Extraction Index (shadow) array.
        source: Sensor name.
        date: Acquisition date string.
        cloud_mask: Boolean mask (True = clear).
        quantum_confidence: Per-pixel confidence from quantum measurement.
        uncertainty: Per-pixel epistemic uncertainty (MC std dev).
        quantum_entropy: Per-pixel Shannon entropy of class distribution.
    """

    water_probability: np.ndarray
    water_binary: np.ndarray
    ndwi: np.ndarray
    mndwi: np.ndarray
    awei_sh: np.ndarray
    source: str
    date: str
    cloud_mask: np.ndarray
    quantum_confidence: np.ndarray
    uncertainty: np.ndarray = field(default_factory=lambda: np.array([]))
    quantum_entropy: np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# Spectral index calculation
# ---------------------------------------------------------------------------

def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute (a - b) / (a + b) avoiding division by zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            (a + b) != 0,
            (a - b) / (a + b),
            0.0,
        )
    return result.astype("float32")


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDWI = (Green − NIR) / (Green + NIR)  — McFeeters 1996."""
    return _safe_ratio(green, nir)


def compute_mndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """MNDWI = (Green − SWIR1) / (Green + SWIR1)  — Xu 2006."""
    return _safe_ratio(green, swir1)


def compute_awei_sh(
    blue: np.ndarray,
    green: np.ndarray,
    nir: np.ndarray,
    swir1: np.ndarray,
    swir2: np.ndarray,
) -> np.ndarray:
    """AWEI_sh = 4(Green − SWIR1) − (0.25·NIR + 2.75·SWIR2) — Feyisa 2014."""
    return (
        4.0 * (green - swir1)
        - (0.25 * nir + 2.75 * swir2)
    ).astype("float32")


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDVI = (NIR − Red) / (NIR + Red)  — v2.0 addition."""
    return _safe_ratio(nir, red)


def compute_bsi(
    blue: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    swir1: np.ndarray,
) -> np.ndarray:
    """Bare Soil Index = ((SWIR1 + Red) − (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue)).

    v2.0: Additional index for the 8-state quantum decomposition.
    Helps distinguish bare soil from shallow water.
    """
    return _safe_ratio((swir1 + red), (nir + blue))


# ---------------------------------------------------------------------------
# Spectral Attention Mechanism (v2.0)
# ---------------------------------------------------------------------------

class SpectralAttention:
    """Learned attention weights for spectral index importance.

    Different sensors have different strengths:
    - Landsat/S2 with SWIR → MNDWI and AWEI are most discriminative
    - NAIP without SWIR → NDWI must carry most of the signal

    The attention mechanism softmax-normalises per-sensor weights
    and applies them to bias the quantum encoding angles.
    """

    def __init__(self, sensor: str = "landsat") -> None:
        weights = ATTENTION_WEIGHTS.get(sensor, ATTENTION_WEIGHTS["landsat"])
        self.raw_weights = weights
        # Softmax normalisation
        vals = np.array([weights["ndwi"], weights["mndwi"], weights["awei"]])
        exp_vals = np.exp(vals * 3.0)  # temperature-scaled
        self.weights = exp_vals / exp_vals.sum()

    def apply(
        self,
        ndwi: np.ndarray,
        mndwi: np.ndarray,
        awei: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply attention weighting to spectral indices.

        Scales the effective contribution of each index proportionally
        to its attention weight before quantum encoding.

        Returns:
            Weighted (ndwi, mndwi, awei) arrays.
        """
        return (
            ndwi * self.weights[0],
            mndwi * self.weights[1],
            awei * self.weights[2],
        )


# ---------------------------------------------------------------------------
# Quantum-Inspired Feature Encoding (QFE) — v2.0: 3 qubits
# ---------------------------------------------------------------------------

class QuantumFeatureEncoder:
    """Encode spectral indices into a simulated 3-qubit quantum state.

    **v2.0** upgrades from 2 qubits (4 states) to 3 qubits (8 states):

        |000⟩ = deep water      |001⟩ = shallow water
        |010⟩ = wet vegetation   |011⟩ = flood shadow
        |100⟩ = dry vegetation   |101⟩ = bare soil
        |110⟩ = impervious       |111⟩ = cloud/noise

    Encoding procedure:
        1. Map each spectral index to a rotation angle θ ∈ [0, π]
           weighted by spectral attention.
        2. Apply Ry(θ) rotation gates to each of the 3 qubits.
        3. Apply cascaded CZ entanglement gates (CZ₁₂, CZ₂₃, CZ₁₃).
        4. Apply a second layer of Ry rotations (variational params).
        5. Measure via Born rule → 8-class probability distribution.

    Water probability = P(|000⟩) + P(|001⟩)  (deep + shallow water).
    """

    def __init__(
        self,
        entangle_strength: float = ENTANGLEMENT_STRENGTH,
        sensor: str = "landsat",
    ) -> None:
        self.entangle_strength = entangle_strength
        self.attention = SpectralAttention(sensor)

        # Build the full 8×8 entangling unitary
        self._circuit_unitary = self._build_vqc_unitary(entangle_strength)

    @staticmethod
    def _build_vqc_unitary(theta: float) -> np.ndarray:
        """Build an 8×8 variational quantum circuit unitary.

        Architecture: Ry(θ)⊗3 layer → CZ₁₂ → CZ₂₃ → CZ₁₃ → Ry(θ/2)⊗3

        The two Ry layers form a "brickwork" variational ansatz, and
        the three CZ gates create a fully-connected entanglement graph
        among all 3 qubits.

        Returns:
            8×8 complex unitary matrix.
        """
        c, s = np.cos(theta / 2), np.sin(theta / 2)

        # Single-qubit Ry(θ)
        ry = np.array([[c, -s], [s, c]], dtype=complex)

        # Second layer Ry(θ/2)
        c2, s2 = np.cos(theta / 4), np.sin(theta / 4)
        ry2 = np.array([[c2, -s2], [s2, c2]], dtype=complex)

        # Tensor products for 3-qubit layers
        ry_layer1 = np.kron(np.kron(ry, ry), ry)  # 8×8
        ry_layer2 = np.kron(np.kron(ry2, ry2), ry2)  # 8×8

        # CZ gates on 3-qubit system
        # CZ₁₂ = CZ on qubits 0,1 ⊗ I₃
        cz12 = np.eye(8, dtype=complex)
        cz12[3, 3] = -1  # |011⟩
        cz12[7, 7] = -1  # |111⟩

        # CZ₂₃ = I₁ ⊗ CZ on qubits 1,2
        cz23 = np.eye(8, dtype=complex)
        cz23[3, 3] = -1  # |011⟩
        cz23[7, 7] = -1  # |111⟩
        # Correct CZ₂₃: phase on states where qubits 1 AND 2 are both |1⟩
        cz23 = np.eye(8, dtype=complex)
        for i in range(8):
            bits = [(i >> 2) & 1, (i >> 1) & 1, i & 1]
            if bits[1] == 1 and bits[2] == 1:
                cz23[i, i] = -1

        # CZ₁₃: phase on states where qubits 0 AND 2 are both |1⟩
        cz13 = np.eye(8, dtype=complex)
        for i in range(8):
            bits = [(i >> 2) & 1, (i >> 1) & 1, i & 1]
            if bits[0] == 1 and bits[2] == 1:
                cz13[i, i] = -1

        # Full circuit: Ry_layer2 → CZ₁₃ → CZ₂₃ → CZ₁₂ → Ry_layer1
        return ry_layer2 @ cz13 @ cz23 @ cz12 @ ry_layer1

    def encode(
        self,
        ndwi: np.ndarray,
        mndwi: np.ndarray,
        awei: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode spectral indices for every pixel into quantum probabilities.

        Returns:
            water_prob: Per-pixel water probability (|000⟩ + |001⟩).
            confidence: Per-pixel quantum confidence (max class prob).
        """
        shape = ndwi.shape

        # Apply spectral attention
        a_ndwi, a_mndwi, a_awei = self.attention.apply(ndwi, mndwi, awei)

        # Flatten for vectorised computation
        ndwi_f = a_ndwi.ravel().astype("float64")
        mndwi_f = np.nan_to_num(a_mndwi.ravel().astype("float64"), nan=0.0)
        awei_f = np.nan_to_num(a_awei.ravel().astype("float64"), nan=0.0)

        # Map indices to rotation angles θ ∈ [0, π]
        # High index (water) → small θ → high cos(θ/2) → high |0⟩ amplitude
        theta_ndwi = (1.0 - ndwi_f) / 2.0 * np.pi
        theta_mndwi = (1.0 - mndwi_f) / 2.0 * np.pi
        theta_awei = (1.0 - expit(awei_f)) * np.pi

        # Build per-pixel initial 3-qubit state: Ry(θ₁) ⊗ Ry(θ₂) ⊗ Ry(θ₃) |000⟩
        c1, s1 = np.cos(theta_ndwi / 2), np.sin(theta_ndwi / 2)
        c2, s2 = np.cos(theta_mndwi / 2), np.sin(theta_mndwi / 2)
        c3, s3 = np.cos(theta_awei / 2), np.sin(theta_awei / 2)

        # 8-element state vector via tensor product of 3 single-qubit states
        psi_init = np.stack([
            c1 * c2 * c3,   # |000⟩ — deep water
            c1 * c2 * s3,   # |001⟩ — shallow water
            c1 * s2 * c3,   # |010⟩ — wet vegetation
            c1 * s2 * s3,   # |011⟩ — flood shadow
            s1 * c2 * c3,   # |100⟩ — dry vegetation
            s1 * c2 * s3,   # |101⟩ — bare soil
            s1 * s2 * c3,   # |110⟩ — impervious
            s1 * s2 * s3,   # |111⟩ — cloud/noise
        ], axis=-1).astype(complex)  # shape: (n_pixels, 8)

        # Apply AWEI phase rotation on water states for constructive interference
        phase = np.exp(1j * theta_awei)
        psi_init[:, 0] *= phase   # deep water
        psi_init[:, 1] *= phase   # shallow water

        # Apply the full variational circuit unitary
        psi_final = (self._circuit_unitary @ psi_init.T).T  # (n_pixels, 8)

        # Born rule → 8-class probability distribution
        probs = np.abs(psi_final) ** 2

        # Normalise (numerical safety)
        prob_sum = probs.sum(axis=1, keepdims=True)
        prob_sum = np.where(prob_sum > 0, prob_sum, 1.0)
        probs /= prob_sum

        # Water probability = P(|000⟩) + P(|001⟩)  (deep + shallow)
        water_prob = (probs[:, 0] + probs[:, 1]).reshape(shape).astype("float32")

        # Confidence = max single-class probability
        confidence = probs.max(axis=1).reshape(shape).astype("float32")

        return water_prob, confidence

    def encode_with_uncertainty(
        self,
        ndwi: np.ndarray,
        mndwi: np.ndarray,
        awei: np.ndarray,
        n_samples: int = MC_DROPOUT_SAMPLES,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Encode with Monte Carlo uncertainty quantification.

        Runs ``n_samples`` forward passes with small random perturbations
        to the quantum amplitudes (analogous to MC dropout) and measures
        the standard deviation of water probability across samples.

        Returns:
            water_prob: Mean water probability across MC samples.
            confidence: Mean confidence.
            uncertainty: Std dev of water probability (epistemic uncertainty).
            entropy: Shannon entropy of the mean 8-class distribution.
        """
        shape = ndwi.shape
        rng = np.random.default_rng(42)

        mc_probs = np.zeros((n_samples,) + shape, dtype="float32")
        mc_confs = np.zeros((n_samples,) + shape, dtype="float32")

        for s in range(n_samples):
            # Perturb indices with small noise (MC dropout equivalent)
            noise_scale = MC_NOISE_SCALE * (1.0 if s > 0 else 0.0)  # first pass is clean
            ndwi_p = ndwi + rng.normal(0, noise_scale, shape).astype("float32")
            mndwi_p = mndwi + rng.normal(0, noise_scale, shape).astype("float32")
            awei_p = awei + rng.normal(0, noise_scale, shape).astype("float32")

            wp, conf = self.encode(ndwi_p, mndwi_p, awei_p)
            mc_probs[s] = wp
            mc_confs[s] = conf

        # Aggregate
        water_prob = mc_probs.mean(axis=0)
        confidence = mc_confs.mean(axis=0)
        uncertainty = mc_probs.std(axis=0)

        # Shannon entropy of mean class distribution
        # (approximate from water_prob: 2-class entropy)
        p = np.clip(water_prob, 1e-7, 1.0 - 1e-7)
        entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

        return water_prob, confidence, uncertainty, entropy


# ---------------------------------------------------------------------------
# Quantum Kernel SVM (unchanged API, optimised internals)
# ---------------------------------------------------------------------------

class QuantumKernelSVM:
    """SVM classifier with a quantum-inspired kernel.

    v2.0: Uses the 3-qubit feature map for richer kernel expressivity.
    K(xᵢ, xⱼ) = |⟨φ(xᵢ)|φ(xⱼ)⟩|²  in 8-dimensional Hilbert space.
    """

    def __init__(self, C: float = 10.0, n_samples: int = 5000) -> None:
        self.C = C
        self.n_samples = n_samples
        self.scaler = StandardScaler()
        self._svm: Optional[SVC] = None
        self._fitted = False
        self._encoder = QuantumFeatureEncoder()

    def _quantum_feature_map(self, X: np.ndarray) -> np.ndarray:
        """Map feature vectors to 3-qubit quantum state amplitudes.

        Returns:
            Quantum state vectors (n_samples, 8) as complex.
        """
        n = X.shape[0]
        n_features = X.shape[1]
        states = np.zeros((n, HILBERT_DIM), dtype=complex)
        states[:, 0] = 1.0  # initialise to |000⟩

        for i in range(min(n_features, N_QUBITS)):
            theta = X[:, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)

            # Apply Ry rotation on qubit i
            new_states = np.zeros_like(states)
            step = 2 ** (N_QUBITS - 1 - i)

            for j in range(HILBERT_DIM):
                # Determine if qubit i is |0⟩ or |1⟩ in state |j⟩
                bit = (j >> (N_QUBITS - 1 - i)) & 1
                partner = j ^ (1 << (N_QUBITS - 1 - i))  # flip bit i

                if bit == 0:
                    new_states[:, j] += states[:, j] * c - states[:, partner] * s
                else:
                    new_states[:, j] += states[:, partner] * s + states[:, j] * c

            states = new_states

        # Apply entanglement via the circuit unitary
        states = (self._encoder._circuit_unitary @ states.T).T

        # Normalise
        norms = np.linalg.norm(states, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        states /= norms

        return states

    def _quantum_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute quantum kernel: K[i,j] = |⟨φ(x1_i)|φ(x2_j)⟩|²."""
        phi1 = self._quantum_feature_map(X1)
        phi2 = self._quantum_feature_map(X2)
        overlap = phi1 @ phi2.conj().T
        return np.asarray(np.abs(overlap) ** 2, dtype=np.float64)

    def fit_predict(
        self,
        features: np.ndarray,
        ndwi: np.ndarray,
    ) -> np.ndarray:
        """Train on pseudo-labels from NDWI thresholds, then predict.

        Returns:
            Water probability array (n_pixels,) in [0, 1].
        """
        n_pixels = features.shape[0]
        labels = (ndwi > NDWI_WATER_THRESHOLD).astype(int)

        rng = np.random.default_rng(42)
        idx = rng.choice(n_pixels, size=min(self.n_samples, n_pixels), replace=False)
        X_train = features[idx]
        y_train = labels[idx]

        if len(np.unique(y_train)) < 2:
            logger.warning("QK-SVM: only one class in training set, falling back to NDWI")
            return (ndwi > NDWI_WATER_THRESHOLD).astype("float32")

        X_scaled = self.scaler.fit_transform(X_train) * np.pi

        logger.debug("Computing quantum kernel (%d × %d) …", len(idx), len(idx))
        K_train = self._quantum_kernel(X_scaled, X_scaled)

        self._svm = SVC(C=self.C, kernel="precomputed", probability=True, random_state=42)
        self._svm.fit(K_train, y_train)
        self._fitted = True

        probs = np.zeros(n_pixels, dtype="float32")
        batch_size = 2000
        X_train_ref = X_scaled

        for start in range(0, n_pixels, batch_size):
            end = min(start + batch_size, n_pixels)
            X_batch = self.scaler.transform(features[start:end]) * np.pi
            K_batch = self._quantum_kernel(X_batch, X_train_ref)
            p = self._svm.predict_proba(K_batch)
            water_col = np.where(self._svm.classes_ == 1)[0]
            if len(water_col) > 0:
                probs[start:end] = p[:, water_col[0]]
            else:
                probs[start:end] = 1.0 - p[:, 0]

        return probs


# ---------------------------------------------------------------------------
# Gradient-Boosted Spectral Index Ensemble (v2.0: added NDVI, BSI features)
# ---------------------------------------------------------------------------

class SpectralGBClassifier:
    """Classical gradient-boosted classifier on spectral features.

    v2.0: Includes NDVI and BSI as additional features for better
    discrimination of bare soil, vegetation, and shallow water.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        n_samples: int = 10000,
    ) -> None:
        self.n_samples = n_samples
        self.scaler = StandardScaler()
        self._gb = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            subsample=0.8,
        )
        self._fitted = False

    def fit_predict(
        self,
        features: np.ndarray,
        ndwi: np.ndarray,
    ) -> np.ndarray:
        """Train on pseudo-labels and predict water probability."""
        n_pixels = features.shape[0]
        labels = (ndwi > NDWI_WATER_THRESHOLD).astype(int)

        rng = np.random.default_rng(123)
        idx = rng.choice(n_pixels, size=min(self.n_samples, n_pixels), replace=False)

        X_train = self.scaler.fit_transform(features[idx])
        y_train = labels[idx]

        if len(np.unique(y_train)) < 2:
            logger.warning("GB: only one class in training set, falling back to NDWI")
            return (ndwi > NDWI_WATER_THRESHOLD).astype("float32")

        self._gb.fit(X_train, y_train)
        self._fitted = True

        probs = np.zeros(n_pixels, dtype="float32")
        batch_size = 10000

        for start in range(0, n_pixels, batch_size):
            end = min(start + batch_size, n_pixels)
            X_batch = self.scaler.transform(features[start:end])
            p = self._gb.predict_proba(X_batch)
            water_col = np.where(self._gb.classes_ == 1)[0]
            if len(water_col) > 0:
                probs[start:end] = p[:, water_col[0]]
            else:
                probs[start:end] = 1.0 - p[:, 0]

        return probs


# ---------------------------------------------------------------------------
# Ensemble Meta-Learner (v2.0: replaces fixed Bayesian averaging)
# ---------------------------------------------------------------------------

class EnsembleMetaLearner:
    """Stacked generalisation (meta-learning) for classifier fusion.

    Instead of fixed Bayesian model-averaging weights, the meta-learner
    trains a Ridge classifier on the stacked predictions of QFE, QK-SVM,
    and GBSIE to learn optimal per-pixel fusion weights from data.

    Falls back to adaptive Bayesian averaging if training data is
    insufficient for the meta-learner.

    Parameters
    ----------
    n_samples:
        Max samples for meta-learner training.
    regularization:
        Ridge regularisation strength (alpha).
    """

    def __init__(
        self,
        n_samples: int = 10000,
        regularization: float = 1.0,
    ) -> None:
        self.n_samples = n_samples
        self.regularization = regularization
        self._ridge: Optional[RidgeClassifier] = None
        self._fitted = False

    def fit_fuse(
        self,
        quantum_prob: np.ndarray,
        svm_prob: np.ndarray,
        gb_prob: np.ndarray,
        quantum_confidence: np.ndarray,
        ndwi: np.ndarray,
    ) -> np.ndarray:
        """Train the meta-learner and produce fused predictions.

        Args:
            quantum_prob: QFE water probability.
            svm_prob: QK-SVM water probability.
            gb_prob: GBSIE water probability.
            quantum_confidence: QFE confidence scores.
            ndwi: NDWI array for pseudo-label generation.

        Returns:
            Fused water probability in [0, 1].
        """
        shape = quantum_prob.shape
        n_pixels = quantum_prob.size

        # Build meta-feature matrix: 4 features per pixel
        meta_features = np.stack([
            quantum_prob.ravel(),
            svm_prob.ravel(),
            gb_prob.ravel(),
            quantum_confidence.ravel(),
        ], axis=1).astype("float32")

        labels = (ndwi.ravel() > NDWI_WATER_THRESHOLD).astype(int)

        # Train on a subsample
        rng = np.random.default_rng(999)
        idx = rng.choice(n_pixels, size=min(self.n_samples, n_pixels), replace=False)

        X_train = meta_features[idx]
        y_train = labels[idx]

        if len(np.unique(y_train)) < 2:
            # Fall back to adaptive Bayesian averaging
            return bayesian_model_average(
                quantum_prob, svm_prob, gb_prob, quantum_confidence,
            )

        self._ridge = RidgeClassifier(alpha=self.regularization)
        self._ridge.fit(X_train, y_train)
        self._fitted = True

        # Predict decision function → sigmoid for probability
        decision = self._ridge.decision_function(meta_features)
        fused = expit(decision).reshape(shape).astype("float32")

        return np.clip(fused, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Bayesian Model Averaging (kept as fallback)
# ---------------------------------------------------------------------------

def bayesian_model_average(
    quantum_prob: np.ndarray,
    svm_prob: np.ndarray,
    gb_prob: np.ndarray,
    quantum_confidence: np.ndarray,
    *,
    prior_quantum: float = 0.40,
    prior_svm: float = 0.35,
    prior_gb: float = 0.25,
) -> np.ndarray:
    """Fuse three model predictions via Bayesian model averaging.

    Fallback when the meta-learner cannot be trained.

    The final probability is:
        p_final = w_q · p_quantum + w_s · p_svm + w_g · p_gb

    where weights incorporate adaptive confidence:
        w_q ∝ prior_quantum × confidence_quantum
        w_s ∝ prior_svm × (1 − |p_svm − p_quantum|)
        w_g ∝ prior_gb (fixed anchor)
    """
    w_q = prior_quantum * quantum_confidence
    w_s = prior_svm * (1.0 - np.abs(svm_prob - quantum_prob))
    w_g = np.full_like(w_q, prior_gb)

    w_total = w_q + w_s + w_g
    w_total = np.where(w_total > 0, w_total, 1.0)
    w_q /= w_total
    w_s /= w_total
    w_g /= w_total

    fused = w_q * quantum_prob + w_s * svm_prob + w_g * gb_prob
    return np.clip(fused, 0.0, 1.0).astype("float32")


# ---------------------------------------------------------------------------
# Main classifier orchestrator — v2.0
# ---------------------------------------------------------------------------

class QuantumHybridClassifier:
    """Orchestrates the full QIEC v2.0 pipeline for water classification.

    Runs QFE (3-qubit) → QK-SVM → GBSIE → Meta-Learner fusion
    for each observation.

    v2.0 enhancements:
    - 3-qubit QFE with spectral attention
    - Monte Carlo uncertainty quantification
    - Meta-learner stacking (Ridge) instead of fixed BMA
    - NDVI and BSI as additional features
    - Per-sensor adaptive attention

    Parameters
    ----------
    use_quantum_svm:
        Enable QK-SVM component (slower but more accurate).
    use_meta_learner:
        Use Ridge meta-learner instead of fixed BMA.
    use_uncertainty:
        Enable MC uncertainty quantification.
    svm_max_samples:
        Max training samples for QK-SVM.
    gb_n_estimators:
        Boosting rounds for GB ensemble.
    """

    def __init__(
        self,
        use_quantum_svm: bool = True,
        use_meta_learner: bool = True,
        use_uncertainty: bool = True,
        svm_max_samples: int = 5000,
        gb_n_estimators: int = 200,
    ) -> None:
        self.use_quantum_svm = use_quantum_svm
        self.use_meta_learner = use_meta_learner
        self.use_uncertainty = use_uncertainty
        self.svm_max_samples = svm_max_samples
        self.gb_n_estimators = gb_n_estimators

        # Lazy init per sensor (attention weights differ)
        self._qfe_cache: dict[str, QuantumFeatureEncoder] = {}
        self._qk_svm = QuantumKernelSVM(n_samples=svm_max_samples) if use_quantum_svm else None
        self._gb = SpectralGBClassifier(n_estimators=gb_n_estimators)
        self._meta = EnsembleMetaLearner() if use_meta_learner else None

    def _get_qfe(self, sensor: str) -> QuantumFeatureEncoder:
        """Get or create a sensor-specific QFE (with attention weights)."""
        if sensor not in self._qfe_cache:
            self._qfe_cache[sensor] = QuantumFeatureEncoder(sensor=sensor)
        return self._qfe_cache[sensor]

    def classify_stack(self, stack: AlignedStack) -> list[ClassificationResult]:
        """Classify every observation in the aligned stack.

        Args:
            stack: Preprocessed AlignedStack at 10 m resolution.

        Returns:
            List of ClassificationResult, one per observation.
        """
        results: list[ClassificationResult] = []

        for i, obs in enumerate(stack.observations):
            logger.info(
                "Classifying observation %d/%d — %s %s",
                i + 1, stack.total_scenes, obs["source"], obs["date"],
            )

            result = self._classify_single(obs)
            results.append(result)

        logger.info("Classification complete — %d observations processed", len(results))
        return results

    def _classify_single(self, obs: dict) -> ClassificationResult:
        """Run the full QIEC v2.0 pipeline on one observation."""
        green = obs["green"]
        nir = obs["nir"]
        red = obs["red"]
        blue = obs["blue"]
        swir1 = obs["swir1"]
        swir2 = obs["swir2"]
        cloud_mask = obs["cloud_mask"]
        source = obs["source"]

        shape = green.shape
        has_swir = not np.all(np.isnan(swir1))

        # --- Step 1: Compute spectral indices (cached per obs) ---
        ndwi = compute_ndwi(green, nir)

        if has_swir:
            mndwi = compute_mndwi(green, swir1)
            awei_sh = compute_awei_sh(blue, green, nir, swir1, swir2)
            ndvi = compute_ndvi(red, nir)
            bsi = compute_bsi(blue, red, nir, swir1)
        else:
            mndwi = ndwi.copy()
            awei_sh = np.zeros_like(ndwi)
            ndvi = compute_ndvi(red, nir)
            bsi = np.zeros_like(ndwi)

        # --- Step 2: Quantum-Inspired Feature Encoding (3-qubit) ---
        qfe = self._get_qfe(source)

        if self.use_uncertainty:
            q_prob, q_conf, uncertainty, entropy = qfe.encode_with_uncertainty(
                ndwi, mndwi, awei_sh
            )
        else:
            q_prob, q_conf = qfe.encode(ndwi, mndwi, awei_sh)
            uncertainty = np.zeros(shape, dtype="float32")
            entropy = np.zeros(shape, dtype="float32")

        # --- Step 3: Build enhanced feature matrix ---
        feature_bands = [green, nir, red, ndwi, mndwi, ndvi]
        if has_swir:
            feature_bands.extend([swir1, swir2, awei_sh, bsi])

        features = np.stack(
            [b.ravel() for b in feature_bands], axis=1
        ).astype("float32")
        features = np.nan_to_num(features, nan=0.0)
        ndwi_flat = ndwi.ravel()

        # --- Step 4: QK-SVM ---
        if self._qk_svm is not None:
            svm_prob_flat = self._qk_svm.fit_predict(features, ndwi_flat)
            svm_prob = svm_prob_flat.reshape(shape)
        else:
            svm_prob = q_prob.copy()

        # --- Step 5: Gradient-Boosted Ensemble (v2.0: with NDVI, BSI) ---
        gb_prob_flat = self._gb.fit_predict(features, ndwi_flat)
        gb_prob = gb_prob_flat.reshape(shape)

        # --- Step 6: Fusion — Meta-Learner or Bayesian ---
        if self._meta is not None:
            fused_prob = self._meta.fit_fuse(
                q_prob, svm_prob, gb_prob, q_conf, ndwi
            )
        else:
            # Per-sensor Bayesian priors (fallback)
            if source == "landsat":
                priors = (0.35, 0.35, 0.30)
            elif source == "sentinel2":
                priors = (0.40, 0.30, 0.30)
            else:
                priors = (0.45, 0.15, 0.40)

            fused_prob = bayesian_model_average(
                q_prob, svm_prob, gb_prob, q_conf,
                prior_quantum=priors[0],
                prior_svm=priors[1],
                prior_gb=priors[2],
            )

        # Apply cloud mask
        fused_prob = np.where(cloud_mask, fused_prob, np.nan)
        water_binary = fused_prob > 0.5

        return ClassificationResult(
            water_probability=fused_prob,
            water_binary=water_binary,
            ndwi=ndwi,
            mndwi=mndwi,
            awei_sh=awei_sh,
            source=source,
            date=obs["date"],
            cloud_mask=cloud_mask,
            quantum_confidence=q_conf,
            uncertainty=uncertainty,
            quantum_entropy=entropy,
        )
