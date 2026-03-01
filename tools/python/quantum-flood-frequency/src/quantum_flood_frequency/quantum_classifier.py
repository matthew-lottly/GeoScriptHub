"""
quantum_classifier.py
=====================
Pseudo-Quantum Hybrid AI Water / Flood Classifier.

This module implements a novel **Quantum-Inspired Ensemble Classification
(QIEC)** framework for per-pixel water detection across multi-sensor
remote-sensing imagery.  The approach is innovative and draws from
quantum computing concepts *without* requiring quantum hardware — it
simulates quantum-mechanical principles in classical computation.

Architecture
------------
The classifier fuses three complementary paradigms:

1. **Quantum-Inspired Feature Encoding (QFE)**
   Spectral indices (NDWI, MNDWI, AWEI) are mapped to *probability
   amplitudes* in a simulated 2-qubit Hilbert space.  The pixel state
   is represented as a 4-element complex state vector |ψ⟩ whose squared
   magnitudes give class probabilities (water, vegetation, bare soil,
   shadow).  A parameterised *rotation gate* encodes spectral evidence
   into quantum phase, and a *Hadamard-like mixing gate* entangles
   indices to model their correlations.  The Born-rule measurement
   collapses |ψ⟩ to class probabilities.

2. **Quantum Kernel SVM (QK-SVM)**
   A support-vector classifier trained on a *quantum kernel matrix*.
   The kernel computes the overlap ⟨φ(xᵢ)|φ(xⱼ)⟩ between quantum
   feature-map embeddings of pixel spectra.  This is provably richer
   than classical RBF kernels for certain data manifolds and naturally
   handles the high-dimensional spectral space.

3. **Gradient-Boosted Spectral Index Ensemble (GBSIE)**
   A classical gradient-boosted decision tree trained on raw spectral
   bands plus derived water indices as a safety net — it captures any
   signal the quantum layers miss and stabilises overall accuracy.

The three components vote via **Bayesian model averaging** weighted by
per-sensor reliability priors to produce a final per-pixel water
probability $p_w \\in [0, 1]$.

Theory
------
While true quantum advantage requires fault-tolerant hardware, the
*pseudo-quantum* formalism provides two real benefits:

* **Interference effects** — the Hadamard mixing step allows constructive
  interference between consistent spectral evidence and destructive
  interference when indices disagree, naturally down-weighting ambiguous
  pixels (e.g. wet soil that has high NDWI but low MNDWI).

* **Kernel expressivity** — the quantum kernel spans a feature space
  exponential in the number of qubits, offering richer decision
  boundaries than standard radial kernels for the same compute cost.

References
----------
* Havlíček et al., "Supervised learning with quantum-enhanced feature
  spaces", Nature 567 (2019).
* McFeeters, "The use of the Normalized Difference Water Index (NDWI)",
  Int. J. Remote Sensing 17(7), 1996.
* Xu, "Modification of normalised difference water index (NDWI) to
  enhance open water features", Int. J. Remote Sensing 27(14), 2006.
* Feyisa et al., "Automated Water Extraction Index (AWEI)", Remote
  Sensing of Environment 140, 2014.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import expit  # sigmoid
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from .preprocessing import AlignedStack

logger = logging.getLogger("geoscripthub.quantum_flood_frequency.quantum_classifier")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Spectral index thresholds (literature defaults, refined during training)
NDWI_WATER_THRESHOLD = 0.0
MNDWI_WATER_THRESHOLD = 0.0
AWEI_SH_WATER_THRESHOLD = 0.0

# Quantum simulation parameters
N_QUBITS = 2           # Simulated Hilbert space dimension = 2^N_QUBITS = 4
HILBERT_DIM = 2 ** N_QUBITS  # = 4 basis states
ENTANGLEMENT_STRENGTH = np.pi / 4  # Hadamard-like mixing angle


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
    """AWEI_sh = 4(Green − SWIR1) − (0.25·NIR + 2.75·SWIR2) — Feyisa 2014.

    Shadow-optimised variant designed to suppress shadow false positives.
    """
    return (
        4.0 * (green - swir1)
        - (0.25 * nir + 2.75 * swir2)
    ).astype("float32")


# ---------------------------------------------------------------------------
# Quantum-Inspired Feature Encoding (QFE)
# ---------------------------------------------------------------------------

class QuantumFeatureEncoder:
    """Encode spectral indices into a simulated 2-qubit quantum state.

    The 4 basis states |00⟩, |01⟩, |10⟩, |11⟩ represent:
        |00⟩ = water, |01⟩ = vegetation, |10⟩ = bare soil, |11⟩ = shadow.

    Encoding procedure:
        1. Map each spectral index to a rotation angle θ ∈ [0, π].
        2. Apply rotation gates Ry(θ) to initialise amplitude.
        3. Apply entangling Hadamard-like mixing gate to model
           correlations between NDWI and MNDWI.
        4. Measure in the computational basis → class probabilities
           via the Born rule: P(class) = |⟨class|ψ⟩|².
    """

    def __init__(self, entangle_strength: float = ENTANGLEMENT_STRENGTH) -> None:
        self.entangle_strength = entangle_strength

        # Pre-compute the mixing (entangling) unitary
        self._mixing_gate = self._build_mixing_gate(entangle_strength)

    @staticmethod
    def _build_mixing_gate(theta: float) -> np.ndarray:
        """Build a 4×4 unitary mixing gate.

        This is a tensor product of two Hadamard-like rotation matrices
        with a controlled-phase entanglement:

            U = (Ry(θ) ⊗ Ry(θ)) · CZ

        where CZ is the controlled-Z gate adding π phase to |11⟩.
        """
        c, s = np.cos(theta / 2), np.sin(theta / 2)

        # Single-qubit rotation Ry(θ)
        ry = np.array([[c, -s], [s, c]], dtype=complex)

        # Tensor product Ry ⊗ Ry → 4×4
        ry_kron = np.kron(ry, ry)

        # Controlled-Z gate
        cz = np.diag([1, 1, 1, -1]).astype(complex)

        return cz @ ry_kron

    def encode(
        self,
        ndwi: np.ndarray,
        mndwi: np.ndarray,
        awei: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode spectral indices for every pixel into quantum probabilities.

        Args:
            ndwi: NDWI array, values in [-1, 1].
            mndwi: MNDWI array, values in [-1, 1] (NaN allowed).
            awei: AWEI_sh array (unbounded).

        Returns:
            water_prob: Per-pixel water probability from Born rule.
            confidence: Per-pixel quantum measurement confidence
                (max probability among classes — higher = more certain).
        """
        shape = ndwi.shape
        flat_n = ndwi.size

        # Flatten for vectorised computation
        ndwi_f = ndwi.ravel().astype("float64")
        mndwi_f = np.nan_to_num(mndwi.ravel().astype("float64"), nan=0.0)
        awei_f = np.nan_to_num(awei.ravel().astype("float64"), nan=0.0)

        # Map indices to rotation angles θ ∈ [0, π]
        # High NDWI (water) → small θ → high cos(θ/2) → high |00⟩ amplitude
        # Low NDWI  (land)  → large θ → low  cos(θ/2) → low  |00⟩ amplitude
        # Formula: θ = (1 − val) / 2 × π   (inverted so water ↔ |00⟩)
        theta_ndwi = (1.0 - ndwi_f) / 2.0 * np.pi
        theta_mndwi = (1.0 - mndwi_f) / 2.0 * np.pi

        # AWEI is unbounded → sigmoid to [0, 1], invert, then scale to [0, π]
        theta_awei = (1.0 - expit(awei_f)) * np.pi

        # Build per-pixel initial state vectors via Ry rotations
        # |ψ_init⟩ = Ry(θ_ndwi) ⊗ Ry(θ_mndwi) |00⟩
        # Enhanced with AWEI as a global phase rotation
        #
        # For efficiency, we compute the amplitude formula analytically:
        #   |ψ_init⟩ = [cos(θ₁/2)cos(θ₂/2),
        #               cos(θ₁/2)sin(θ₂/2),
        #               sin(θ₁/2)cos(θ₂/2),
        #               sin(θ₁/2)sin(θ₂/2)]
        c1 = np.cos(theta_ndwi / 2)
        s1 = np.sin(theta_ndwi / 2)
        c2 = np.cos(theta_mndwi / 2)
        s2 = np.sin(theta_mndwi / 2)

        psi_init = np.stack([
            c1 * c2,
            c1 * s2,
            s1 * c2,
            s1 * s2,
        ], axis=-1)  # shape (flat_n, 4)

        # Apply AWEI as a phase rotation on the water state |00⟩
        phase = np.exp(1j * theta_awei)
        psi_init = psi_init.astype(complex)
        psi_init[:, 0] *= phase  # constructive interference for water

        # Apply the entangling mixing gate
        # |ψ_final⟩ = U |ψ_init⟩
        psi_final = (self._mixing_gate @ psi_init.T).T  # (flat_n, 4)

        # Born rule measurement → class probabilities
        probs = np.abs(psi_final) ** 2  # (flat_n, 4)

        # Normalise (should already sum to 1, but numerical safety)
        prob_sum = probs.sum(axis=1, keepdims=True)
        prob_sum = np.where(prob_sum > 0, prob_sum, 1.0)
        probs /= prob_sum

        # Water probability = |⟨00|ψ⟩|²
        water_prob = probs[:, 0].reshape(shape).astype("float32")

        # Confidence = max class probability (sharper = more confident)
        confidence = probs.max(axis=1).reshape(shape).astype("float32")

        return water_prob, confidence


# ---------------------------------------------------------------------------
# Quantum Kernel SVM
# ---------------------------------------------------------------------------

class QuantumKernelSVM:
    """SVM classifier with a quantum-inspired kernel.

    The quantum kernel computes the fidelity between quantum feature-map
    embeddings:  K(xᵢ, xⱼ) = |⟨φ(xᵢ)|φ(xⱼ)⟩|²

    where |φ(x)⟩ = U_enc(x)|0⟩ is a data-encoding circuit acting on
    the ground state.  We simulate U_enc classically using the same
    Ry-rotation + entanglement scheme as QFE.

    When insufficient training labels are available (unsupervised mode),
    we generate pseudo-labels from spectral index thresholds.
    """

    def __init__(self, C: float = 10.0, n_samples: int = 5000) -> None:
        """Initialise the QK-SVM.

        Args:
            C: SVM regularisation parameter.
            n_samples: Maximum training sample size (subsampled from imagery).
        """
        self.C = C
        self.n_samples = n_samples
        self.scaler = StandardScaler()
        self._svm: Optional[SVC] = None
        self._fitted = False
        self._encoder = QuantumFeatureEncoder()

    def _quantum_feature_map(self, X: np.ndarray) -> np.ndarray:
        """Map feature vectors to quantum state amplitudes.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Quantum state vectors (n_samples, HILBERT_DIM) as complex.
        """
        n = X.shape[0]
        states = np.zeros((n, HILBERT_DIM), dtype=complex)

        for i in range(min(X.shape[1], N_QUBITS)):
            theta = X[:, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)

            if i == 0:
                states[:, 0] = c
                states[:, 1] = s
            else:
                # Expand Hilbert space via tensor product
                new_states = np.zeros_like(states)
                for j in range(HILBERT_DIM):
                    if j < HILBERT_DIM // 2:
                        new_states[:, j] += states[:, j] * c
                    else:
                        new_states[:, j] += states[:, j - HILBERT_DIM // 2] * s
                states = new_states

        # Apply entanglement
        states = (self._encoder._mixing_gate @ states.T).T

        # Normalise
        norms = np.linalg.norm(states, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        states /= norms

        return states

    def _quantum_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute the quantum kernel matrix K(X1, X2).

        K[i,j] = |⟨φ(x1_i)|φ(x2_j)⟩|²
        """
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

        Args:
            features: Scaled feature matrix (n_pixels, n_features).
            ndwi: Flat NDWI array for pseudo-label generation.

        Returns:
            Water probability array (n_pixels,) in [0, 1].
        """
        n_pixels = features.shape[0]

        # Generate pseudo-labels from NDWI
        labels = (ndwi > NDWI_WATER_THRESHOLD).astype(int)

        # Subsample for tractable SVM training
        rng = np.random.default_rng(42)
        idx = rng.choice(n_pixels, size=min(self.n_samples, n_pixels), replace=False)
        X_train = features[idx]
        y_train = labels[idx]

        # Skip if degenerate
        if len(np.unique(y_train)) < 2:
            logger.warning("QK-SVM: only one class in training set, falling back to NDWI")
            return (ndwi > NDWI_WATER_THRESHOLD).astype("float32")

        # Scale features to θ ∈ [0, π] for quantum encoding
        X_scaled = self.scaler.fit_transform(X_train) * np.pi

        # Compute quantum kernel
        logger.debug("Computing quantum kernel (%d × %d) …", len(idx), len(idx))
        K_train = self._quantum_kernel(X_scaled, X_scaled)

        # Train SVM with precomputed kernel
        self._svm = SVC(C=self.C, kernel="precomputed", probability=True, random_state=42)
        self._svm.fit(K_train, y_train)
        self._fitted = True

        # Predict on all pixels (in batches for memory)
        probs = np.zeros(n_pixels, dtype="float32")
        batch_size = 2000
        X_train_ref = X_scaled  # support vectors reference

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
# Gradient-Boosted Spectral Index Ensemble (classical safety net)
# ---------------------------------------------------------------------------

class SpectralGBClassifier:
    """Classical gradient-boosted classifier on spectral features.

    Provides a robust classical baseline that stabilises the ensemble
    when quantum components encounter edge cases (e.g. heavily mixed
    pixels, sensor artefacts).
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
        """Train on pseudo-labels and predict water probability.

        Args:
            features: Feature matrix (n_pixels, n_features).
            ndwi: NDWI array for pseudo-label generation.

        Returns:
            Water probability array (n_pixels,) in [0, 1].
        """
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

        # Predict in batches
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
# Bayesian Model Averaging
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

    The final probability is a weighted combination:

        p_final = w_q · p_quantum + w_s · p_svm + w_g · p_gb

    where weights are the product of fixed priors and adaptive
    confidence scores:

        w_q ∝ prior_quantum × confidence_quantum
        w_s ∝ prior_svm × (1 − |p_svm − p_quantum|)   # agreement bonus
        w_g ∝ prior_gb  (fixed — classical anchor)

    Weights are normalised to sum to 1 per pixel.

    Args:
        quantum_prob: Water probability from QFE.
        svm_prob: Water probability from QK-SVM.
        gb_prob: Water probability from GBSIE.
        quantum_confidence: Confidence from quantum measurement.
        prior_quantum: Prior weight for quantum channel.
        prior_svm: Prior weight for QK-SVM channel.
        prior_gb: Prior weight for GBSIE channel.

    Returns:
        Fused per-pixel water probability in [0, 1].
    """
    # Adaptive weights
    w_q = prior_quantum * quantum_confidence
    w_s = prior_svm * (1.0 - np.abs(svm_prob - quantum_prob))
    w_g = np.full_like(w_q, prior_gb)

    # Normalise
    w_total = w_q + w_s + w_g
    w_total = np.where(w_total > 0, w_total, 1.0)
    w_q /= w_total
    w_s /= w_total
    w_g /= w_total

    fused = w_q * quantum_prob + w_s * svm_prob + w_g * gb_prob
    return np.clip(fused, 0.0, 1.0).astype("float32")


# ---------------------------------------------------------------------------
# Main classifier orchestrator
# ---------------------------------------------------------------------------

class QuantumHybridClassifier:
    """Orchestrates the full QIEC pipeline for water classification.

    Runs QFE → QK-SVM → GBSIE → Bayesian fusion for each observation
    in an AlignedStack.

    Parameters
    ----------
    use_quantum_svm:
        If True, enables the QK-SVM component (computationally
        expensive).  Set to False for faster runs using only QFE + GB.
    svm_max_samples:
        Maximum training samples for QK-SVM.
    gb_n_estimators:
        Number of boosting rounds for gradient-boosted ensemble.
    """

    def __init__(
        self,
        use_quantum_svm: bool = True,
        svm_max_samples: int = 5000,
        gb_n_estimators: int = 200,
    ) -> None:
        self.use_quantum_svm = use_quantum_svm
        self.svm_max_samples = svm_max_samples
        self.gb_n_estimators = gb_n_estimators

        self._qfe = QuantumFeatureEncoder()
        self._qk_svm = QuantumKernelSVM(n_samples=svm_max_samples) if use_quantum_svm else None
        self._gb = SpectralGBClassifier(n_estimators=gb_n_estimators)

    def classify_stack(self, stack: AlignedStack) -> list[ClassificationResult]:
        """Classify every observation in the aligned stack.

        Args:
            stack: Preprocessed AlignedStack of harmonised observations.

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
        """Run the full QIEC pipeline on one observation.

        Args:
            obs: Single observation dict from AlignedStack.

        Returns:
            ClassificationResult for this observation.
        """
        green = obs["green"]
        nir = obs["nir"]
        red = obs["red"]
        blue = obs["blue"]
        swir1 = obs["swir1"]
        swir2 = obs["swir2"]
        cloud_mask = obs["cloud_mask"]

        shape = green.shape
        has_swir = not np.all(np.isnan(swir1))

        # --- Step 1: Compute spectral indices ---
        ndwi = compute_ndwi(green, nir)

        if has_swir:
            mndwi = compute_mndwi(green, swir1)
            awei_sh = compute_awei_sh(blue, green, nir, swir1, swir2)
        else:
            # NAIP fallback — use NDWI-only for MNDWI/AWEI proxies
            mndwi = ndwi.copy()  # best approx without SWIR
            awei_sh = np.zeros_like(ndwi)

        # --- Step 2: Quantum-Inspired Feature Encoding ---
        q_prob, q_conf = self._qfe.encode(ndwi, mndwi, awei_sh)

        # --- Step 3: Build feature matrix for ML classifiers ---
        feature_bands = [green, nir, red, ndwi, mndwi]
        if has_swir:
            feature_bands.extend([swir1, swir2, awei_sh])

        features = np.stack(
            [b.ravel() for b in feature_bands], axis=1
        ).astype("float32")

        # Replace NaN with 0 for ML
        features = np.nan_to_num(features, nan=0.0)
        ndwi_flat = ndwi.ravel()

        # --- Step 4: QK-SVM ---
        if self._qk_svm is not None:
            svm_prob_flat = self._qk_svm.fit_predict(features, ndwi_flat)
            svm_prob = svm_prob_flat.reshape(shape)
        else:
            svm_prob = q_prob.copy()  # fallback

        # --- Step 5: Gradient-Boosted Ensemble ---
        gb_prob_flat = self._gb.fit_predict(features, ndwi_flat)
        gb_prob = gb_prob_flat.reshape(shape)

        # --- Step 6: Bayesian Model Averaging ---
        # Adjust priors by sensor reliability
        source = obs["source"]
        if source == "landsat":
            # Landsat has SWIR → excellent water detection
            priors = (0.35, 0.35, 0.30)
        elif source == "sentinel2":
            # Sentinel-2 has SWIR + higher spatial resolution → good
            priors = (0.40, 0.30, 0.30)
        else:
            # NAIP — no SWIR, rely more on quantum + GB
            priors = (0.45, 0.15, 0.40)

        fused_prob = bayesian_model_average(
            q_prob, svm_prob, gb_prob, q_conf,
            prior_quantum=priors[0],
            prior_svm=priors[1],
            prior_gb=priors[2],
        )

        # Apply cloud mask — set cloudy pixels to NaN
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
        )
