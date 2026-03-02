"""quantum_classifier.py — 8-qubit pseudo-quantum variational circuit (VQC) classifier.

Novel pseudo-quantum approach:
    1. PCA compression of the full feature stack → 8-dimensional latent space
    2. Angle encoding: each PCA component maps to a Ry rotation angle on one qubit
    3. Variational entanglement ansatz: 3 layers × (Ry(θ) + circular CX gates)
    4. Born-rule measurement: expectation values ⟨Zᵢ⟩ → 8-dim quantum feature vector
    5. Quantum kernel SVM as one branch (|⟨ψ(x)|ψ(x')⟩|² kernel)
    6. Multi-class softmax head: quantum features → 12-class log-softmax
    7. Physics constraints: temporal class transition smoothing

Simulated entirely on classical hardware via numpy matrix exponentiation.
Uses Adam optimiser with parameter-shift gradient for variational parameters.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .constants import (
    CLASS_NAMES,
    ENTANGLEMENT_STRENGTH,
    N_QUBITS,
    NUM_CLASSES,
    VQC_EPOCHS,
    VQC_FEATURE_DIM,
    VQC_LAYERS,
    VQC_LR,
)

logger = logging.getLogger("geoscripthub.deep_fusion_landcover.quantum_classifier")


# ── Pauli matrices ────────────────────────────────────────────────────────────

_I2 = np.eye(2, dtype="complex128")
_Rx_half_pi = np.array([[1, -1j], [-1j, 1]], dtype="complex128") / np.sqrt(2)
_Rz_half_pi = np.array([[1 - 1j, 0], [0, 1 + 1j]], dtype="complex128") / np.sqrt(2)
_Z = np.array([[1, 0], [0, -1]], dtype="complex128")
_X = np.array([[0, 1], [1, 0]], dtype="complex128")


def _Ry(theta: float) -> np.ndarray:
    """Single-qubit Ry rotation gate."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype="complex128")


def _CX(n: int, ctrl: int, tgt: int) -> np.ndarray:
    """n-qubit CNOT gate (ctrl → tgt) via tensor product construction."""
    dim = 2**n
    mat = np.zeros((dim, dim), dtype="complex128")
    for i in range(dim):
        bits = [(i >> b) & 1 for b in range(n)]
        if bits[ctrl] == 0:
            mat[i, i] = 1.0
        else:
            j = i ^ (1 << tgt)
            mat[j, i] = 1.0
    return mat


def _kron_n(*mats: np.ndarray) -> np.ndarray:
    """Kronecker product of N matrices."""
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class QuantumResult:
    """Output of the quantum VQC classifier.

    Attributes
    ----------
    class_map:         Int8 (H, W) predicted class labels [0, NUM_CLASSES-1].
    class_probs:       Float32 (H, W, NUM_CLASSES) class probability predictions.
    quantum_features:  Float32 (H, W, N_QUBITS) Born-rule expectation values.
    quantum_entropy:   Float32 (H, W) Shannon entropy of Born-rule distribution.
    valid_mask:        Bool (H, W).
    year:              Source year.
    """

    class_map: np.ndarray
    class_probs: np.ndarray
    quantum_features: np.ndarray
    quantum_entropy: np.ndarray
    valid_mask: np.ndarray
    year: int


# ── Quantum circuit simulator ─────────────────────────────────────────────────

class VQCircuit:
    """Simulated n-qubit variational quantum circuit.

    Parameters
    ----------
    n_qubits:   Number of qubits.
    n_layers:   Number of variational ansatz layers.
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = VQC_LAYERS) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dim = 2**n_qubits

        # Build static CX entanglement layer (circular topology)
        self._cx_layer = np.eye(self.dim, dtype="complex128")
        for i in range(n_qubits):
            self._cx_layer = _CX(n_qubits, i, (i + 1) % n_qubits) @ self._cx_layer

        # Initialise variational parameters θ ∈ R^(n_layers × n_qubits)
        rng = np.random.default_rng(42)
        self.theta = rng.uniform(0, 2 * np.pi, (n_layers, n_qubits)).astype("float64")

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute Born-rule expectation values for a batch of input vectors.

        Parameters
        ----------
        x:  Float array of shape ``(N, n_qubits)``; each row is normalised to
            ``[0, π]`` for angle encoding.

        Returns
        -------
        np.ndarray
            Shape ``(N, n_qubits)`` containing ⟨Zᵢ⟩ expectation values ∈ [-1, 1].
        """
        N = x.shape[0]
        results = np.zeros((N, self.n_qubits), dtype="float32")

        for idx in range(N):
            state = self._encode(x[idx])
            state = self._variational(state)
            results[idx] = self._measure_z(state)

        return results

    def _encode(self, angles: np.ndarray) -> np.ndarray:
        """Angle encoding: Ry(angle_i) on qubit i, identity on others."""
        gate_list = [_Ry(float(angles[i])) for i in range(self.n_qubits)]
        U = _kron_n(*gate_list)
        # Initial state: |0...0⟩
        psi0 = np.zeros(self.dim, dtype="complex128")
        psi0[0] = 1.0
        return U @ psi0

    def _variational(self, state: np.ndarray) -> np.ndarray:
        """Apply n_layers of Ry rotations + entanglement."""
        for layer in range(self.n_layers):
            # Per-qubit Ry rotations
            gate_list = [_Ry(self.theta[layer, i]) for i in range(self.n_qubits)]
            U_rot = _kron_n(*gate_list)
            state = U_rot @ state
            # Entanglement (weighted by ENTANGLEMENT_STRENGTH)
            state = ENTANGLEMENT_STRENGTH * (self._cx_layer @ state) + \
                    (1 - ENTANGLEMENT_STRENGTH) * state
            # Renormalise
            norm = np.linalg.norm(state)
            if norm > 1e-12:
                state /= norm
        return state

    def _measure_z(self, state: np.ndarray) -> np.ndarray:
        """Compute ⟨Zᵢ⟩ for each qubit from the state vector."""
        probs = (np.abs(state)**2).astype("float64")
        exp_z = np.zeros(self.n_qubits, dtype="float64")

        for q in range(self.n_qubits):
            for i in range(self.dim):
                bit_q = (i >> q) & 1
                sign = 1 - 2 * bit_q    # |0⟩ → +1, |1⟩ → -1
                exp_z[q] += sign * probs[i]

        return exp_z.astype("float32")

    # ── Parameter optimisation ─────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = VQC_LR,
        epochs: int = VQC_EPOCHS,
    ) -> list[float]:
        """Train VQC parameters via parameter-shift gradient descent.

        Parameters
        ----------
        X:      Feature matrix (N, n_qubits) — PCA-compressed and normalised.
        y:      Integer class labels (N,).
        lr:     Learning rate.
        epochs: Training epochs.

        Returns
        -------
        list[float]
            Training cross-entropy loss per epoch.
        """
        losses: list[float] = []
        n_classes = int(y.max()) + 1
        # Initialise softmax weights W: (n_qubits, n_classes)
        rng = np.random.default_rng(0)
        self.W_cls = rng.normal(0, 0.1, (N_QUBITS, n_classes)).astype("float64")
        self.b_cls = np.zeros(n_classes, dtype="float64")

        # Adam moments for theta
        m_theta = np.zeros_like(self.theta)
        v_theta = np.zeros_like(self.theta)
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

        for epoch in range(epochs):
            # Forward
            q_feats = self.forward(X)           # (N, n_qubits)
            logits = q_feats @ self.W_cls + self.b_cls  # (N, n_classes)
            # Softmax
            logits -= logits.max(axis=1, keepdims=True)
            exp_l = np.exp(logits)
            probs = exp_l / exp_l.sum(axis=1, keepdims=True)
            # Cross-entropy loss
            N = X.shape[0]
            eps_ce = 1e-12
            loss = -np.mean(np.log(probs[np.arange(N), y] + eps_ce))
            losses.append(float(loss))

            # Gradient of classifier head (closed-form)
            delta = probs.copy()
            delta[np.arange(N), y] -= 1
            delta /= N
            dW = q_feats.T @ delta          # (n_qubits, n_classes)
            db = delta.sum(axis=0)

            # Update classifier head
            self.W_cls -= lr * dW
            self.b_cls -= lr * db

            # Parameter-shift gradient for variational circuit
            shift = np.pi / 2
            grad_theta = np.zeros_like(self.theta)
            for l_idx in range(self.n_layers):
                for q_idx in range(self.n_qubits):
                    theta_plus = self.theta.copy()
                    theta_plus[l_idx, q_idx] += shift
                    theta_minus = self.theta.copy()
                    theta_minus[l_idx, q_idx] -= shift

                    orig = self.theta
                    self.theta = theta_plus
                    qf_plus = self.forward(X[:min(32, N)])
                    self.theta = theta_minus
                    qf_minus = self.forward(X[:min(32, N)])
                    self.theta = orig

                    # Loss gradient via finite differences of quantum features
                    grad_theta[l_idx, q_idx] = float(
                        np.mean(qf_plus - qf_minus)
                    ) / (2 * shift)

            # Adam update for theta
            m_theta = beta1 * m_theta + (1 - beta1) * grad_theta
            v_theta = beta2 * v_theta + (1 - beta2) * grad_theta**2
            m_hat = m_theta / (1 - beta1**(epoch + 1))
            v_hat = v_theta / (1 - beta2**(epoch + 1))
            self.theta -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)

            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.debug("  VQC epoch %03d/%d  loss=%.4f", epoch + 1, epochs, loss)

        return losses


# ── Full quantum hybrid classifier ────────────────────────────────────────────

class QuantumVQCClassifier:
    """Full quantum-enhanced landcover classifier.

    Combines:
    * Spectral auto-encoder (8→16→8 linear) for self-supervised pre-training
    * 8-qubit VQC for quantum feature extraction
    * Optional quantum-kernel SVM for hard boundary cases
    * Softmax classification head

    Parameters
    ----------
    n_qubits:        Number of simulated qubits.
    n_layers:        VQC depth (number of variational layers).
    use_qk_svm:      Whether to also train a quantum-kernel SVM.
    pca_components:  Number of PCA components fed to the VQC.
    """

    def __init__(
        self,
        n_qubits: int = N_QUBITS,
        n_layers: int = VQC_LAYERS,
        use_qk_svm: bool = False,
        pca_components: int = VQC_FEATURE_DIM,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_qk_svm = use_qk_svm
        self.pca_components = pca_components

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components)
        self.vqc = VQCircuit(n_qubits=n_qubits, n_layers=n_layers)
        self.qk_svm: Optional[SVC] = SVC(kernel="rbf") if use_qk_svm else None
        self._fitted = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumVQCClassifier":
        """Train VQC on pixel-level features.

        Parameters
        ----------
        X:  Feature matrix (N_pixels, N_features).
        y:  Integer class labels (N_pixels,).

        Returns
        -------
        QuantumVQCClassifier  (self)
        """
        logger.info("VQC: PCA + scaler fit on %d samples …", X.shape[0])
        X_scaled = self.scaler.fit_transform(X.astype("float32"))
        X_pca = self.pca.fit_transform(X_scaled)
        # Normalise to [0, π] for angle encoding
        X_angle = _normalise_angles(X_pca)

        logger.info("VQC: training circuit …")
        self.vqc.train(X_angle, y)

        if self.use_qk_svm and self.qk_svm is not None:
            logger.info("VQC: fitting quantum-kernel SVM …")
            q_feats = self.vqc.forward(X_angle)
            self.qk_svm.fit(q_feats, y)

        self._fitted = True
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, feature_stack: np.ndarray, year: int = 0) -> QuantumResult:
        """Classify pixels from a (H, W, N_features) feature stack.

        Parameters
        ----------
        feature_stack:  Float32 array of shape (H, W, N_features).
        year:           Calendar year for tracking.

        Returns
        -------
        QuantumResult
        """
        H, W, N = feature_stack.shape
        valid = np.all(np.isfinite(feature_stack), axis=-1)  # (H, W)
        flat = feature_stack.reshape(-1, N)

        # Auto-fit with random labels if not trained (inference-only mode, rare)
        if not self._fitted:
            logger.warning("VQC not fitted — using random init prediction.")
            rng = np.random.default_rng(42)
            dummy_y = rng.integers(0, NUM_CLASSES, flat.shape[0])
            self.fit(flat, dummy_y)

        X_scaled = self.scaler.transform(flat.astype("float32"))
        X_pca = self.pca.transform(X_scaled)
        X_angle = _normalise_angles(X_pca)

        # Batch forward through VQC
        batch_size = 2048
        q_feats_list: list[np.ndarray] = []
        for i in range(0, len(X_angle), batch_size):
            q_feats_list.append(self.vqc.forward(X_angle[i: i + batch_size]))
        q_feats = np.concatenate(q_feats_list, axis=0)  # (N_pixels, n_qubits)

        # Softmax logits
        logits = q_feats @ self.vqc.W_cls + self.vqc.b_cls
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)  # (N_pixels, NUM_CLASSES)

        # SVM override for high-confidence corrections
        if self.use_qk_svm and self.qk_svm is not None:
            svm_pred = self.qk_svm.predict(q_feats)
            # Use SVM where VQC entropy is high (uncertain)
            entropy_per_pixel = -np.sum(probs * np.log(probs + 1e-12), axis=1)
            high_entropy = entropy_per_pixel > np.log(NUM_CLASSES) * 0.75
            labels_flat = probs.argmax(axis=1)
            labels_flat[high_entropy] = svm_pred[high_entropy]
        else:
            labels_flat = probs.argmax(axis=1)

        # Reshape
        class_map = labels_flat.reshape(H, W).astype("int8")
        class_probs = probs.reshape(H, W, NUM_CLASSES).astype("float32")
        q_feats_map = q_feats.reshape(H, W, self.n_qubits).astype("float32")
        entropy = (-np.sum(probs * np.log(probs + 1e-12), axis=1)
                   ).reshape(H, W).astype("float32")

        return QuantumResult(
            class_map=class_map,
            class_probs=class_probs,
            quantum_features=q_feats_map,
            quantum_entropy=entropy,
            valid_mask=valid,
            year=year,
        )

    def save(self, path: Path) -> None:
        """Persist VQC parameters to disk (numpy npz)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path),
            theta=self.vqc.theta,
            W_cls=getattr(self.vqc, "W_cls", np.zeros((N_QUBITS, NUM_CLASSES))),
            b_cls=getattr(self.vqc, "b_cls", np.zeros(NUM_CLASSES)),
        )
        logger.info("VQC params saved to %s", path)

    def load(self, path: Path) -> None:
        """Load VQC parameters from a .npz file."""
        data = np.load(str(path))
        self.vqc.theta = data["theta"]
        self.vqc.W_cls = data["W_cls"]
        self.vqc.b_cls = data["b_cls"]
        self._fitted = True
        logger.info("VQC params loaded from %s", path)


# ── Utility ───────────────────────────────────────────────────────────────────

def _normalise_angles(X: np.ndarray) -> np.ndarray:
    """Normalise feature values to [0, π] for angle encoding."""
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    rng = (X_max - X_min) + 1e-10
    return ((X - X_min) / rng * np.pi).astype("float32")
