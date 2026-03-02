"""ensemble.py — 5-branch stacked meta-classifier for landcover mapping.

Architecture
------------
Branch 1: RandomForest(n_estimators=500)
          — trained on raw 150-dim feature stack

Branch 2: GradientBoostingClassifier
          — trained on PCA-50 reduced features

Branch 3: QuantumVQCClassifier
          — 8-qubit VQC + softmax head (12-class probabilities)

Branch 4: CNNEncoder → LinearSVC head
          — 128-dim ONNX embedding → LinearSVC trained per class

Branch 5: SAM2 OBIA spatial context
          — segment-mean RF predictions (object homogeneity smoothing)

Stacker:  LightGBM(n_estimators=200, num_leaves=63)
          — fit on out-of-fold class probabilities from all 5 branches

The pipeline trains each branch independently, then trains the LightGBM
meta-learner on stacked out-of-fold predictions to correct systematic
biases and combine complementary signal sources.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .constants import CLASS_NAMES, NUM_CLASSES, VALIDATION_SAMPLE_N

logger = logging.getLogger("geoscripthub.deep_fusion_landcover.ensemble")

try:
    import lightgbm as lgb  # type: ignore[import-untyped]
    LGBM_AVAILABLE = True
except ImportError:
    lgb = None  # type: ignore[assignment]
    logger.debug("lightgbm not installed — using LogisticRegression meta-learner.")
    LGBM_AVAILABLE = False


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """Final ensemble classification output for one year.

    Attributes
    ----------
    class_map:         Int8 (H, W) predicted class labels [1, NUM_CLASSES].
    class_probs:       Float32 (H, W, NUM_CLASSES) stacked probabilities.
    confidence:        Float32 (H, W) max-class probability ∈ [0, 1].
    branch_maps:       Dict mapping branch name → (H, W) int8 prediction.
    valid_mask:        Bool (H, W).
    year:              Calendar year.
    """

    class_map: np.ndarray
    class_probs: np.ndarray
    confidence: np.ndarray
    branch_maps: dict[str, np.ndarray]
    valid_mask: np.ndarray
    year: int


# ── Ensemble classifier ───────────────────────────────────────────────────────

class EnsembleClassifier:
    """5-branch stacked ensemble for 12-class Austin landcover mapping.

    Parameters
    ----------
    use_quantum:    Include the QuantumVQC branch.
    use_cnn:        Include the CNN embedding branch.
    use_obia:       Include the SAM2 OBIA spatial context branch.
    n_estimators_rf:  Number of trees in the Random Forest branches.
    random_state:   RNG seed for reproducibility.
    """

    def __init__(
        self,
        use_quantum: bool = True,
        use_cnn: bool = True,
        use_obia: bool = True,
        n_estimators_rf: int = 500,
        random_state: int = 42,
    ) -> None:
        self.use_quantum = use_quantum
        self.use_cnn = use_cnn
        self.use_obia = use_obia
        self.random_state = random_state

        # Branch models
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators_rf,
            max_features="sqrt",
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced",
        )
        self.pca_gb = PCA(n_components=50)
        self.gb = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=random_state,
        )
        self.scaler_gb = StandardScaler()

        self.cnn_svc = LinearSVC(C=1.0, max_iter=2000, random_state=random_state)
        self.scaler_cnn = StandardScaler()

        # Quantum branch: imported lazily to avoid circular
        self.quantum_clf: Any = None

        # Meta-learner
        if LGBM_AVAILABLE:
            self.meta: Any = lgb.LGBMClassifier(  # type: ignore[union-attr]
                n_estimators=200,
                num_leaves=63,
                learning_rate=0.05,
                n_jobs=-1,
                random_state=random_state,
            )
        else:
            self.meta = LogisticRegression(
                C=1.0, max_iter=1000, multi_class="multinomial", n_jobs=-1
            )

        self._fitted = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cnn_embeddings: Optional[np.ndarray] = None,
    ) -> "EnsembleClassifier":
        """Train all branches and the meta-learner.

        Parameters
        ----------
        X:               Feature matrix (N_pixels, N_features).
        y:               Integer class labels (N_pixels,) in [0, NUM_CLASSES-1].
        cnn_embeddings:  Optional (N_pixels, 128) CNN embedding matrix.

        Returns
        -------
        EnsembleClassifier  (self)
        """
        logger.info("Fitting Branch 1: RandomForest …")
        self.rf.fit(X, y)

        logger.info("Fitting Branch 2: GradientBoosting(PCA-50) …")
        X_s = self.scaler_gb.fit_transform(X.astype("float32"))
        X_pca = self.pca_gb.fit_transform(X_s)
        self.gb.fit(X_pca, y)

        if self.use_quantum:
            logger.info("Fitting Branch 3: QuantumVQC …")
            from .quantum_classifier import QuantumVQCClassifier
            qclf = QuantumVQCClassifier(use_qk_svm=False)
            qclf.fit(X, y)
            self.quantum_clf = qclf

        if self.use_cnn and cnn_embeddings is not None:
            logger.info("Fitting Branch 4: CNN-LinearSVC …")
            emb_s = self.scaler_cnn.fit_transform(cnn_embeddings.astype("float32"))
            self.cnn_svc.fit(emb_s, y)  # type: ignore[union-attr]
        else:
            self.cnn_svc = None  # type: ignore[assignment]

        logger.info("Fitting meta-learner (LightGBM) on out-of-fold stacks …")
        meta_X = self._stack_probs(X, y=y, cnn_embeddings=cnn_embeddings, oof=True)
        self.meta.fit(meta_X, y)

        self._fitted = True
        logger.info("Ensemble training complete.")
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        feature_stack: np.ndarray,
        cnn_emb_map: Optional[np.ndarray] = None,
        segment_array: Optional[np.ndarray] = None,
        year: int = 0,
    ) -> ClassificationResult:
        """Classify a full (H, W, N_features) annual composite.

        Parameters
        ----------
        feature_stack:  Float32 (H, W, N_features) feature array.
        cnn_emb_map:    Optional (H, W, 128) CNN embedding map.
        segment_array:  Optional (H, W) integer segment labels for OBIA smoothing.
        year:           Calendar year.

        Returns
        -------
        ClassificationResult
        """
        H, W, N = feature_stack.shape
        valid = np.all(np.isfinite(feature_stack), axis=-1)
        flat = feature_stack.reshape(-1, N)

        if not self._fitted:
            logger.warning("Ensemble not fitted — auto-fitting with dummy labels.")
            rng = np.random.default_rng(42)
            dummy_y = rng.integers(0, NUM_CLASSES, flat.shape[0])
            self.fit(flat, dummy_y)

        cnn_flat = (
            cnn_emb_map.reshape(-1, cnn_emb_map.shape[-1])
            if cnn_emb_map is not None else None
        )

        # Build stacked probability matrix
        meta_X = self._stack_probs(flat, cnn_embeddings=cnn_flat)
        probs = self._meta_proba(meta_X)  # (N_pixels, NUM_CLASSES)

        # OBIA smoothing: replace per-pixel label with segment-modal label
        labels_flat = probs.argmax(axis=1)
        if segment_array is not None and self.use_obia:
            labels_flat = self._obia_smooth(
                labels_flat, probs, segment_array.reshape(-1), H * W
            )

        class_map = (labels_flat + 1).reshape(H, W).astype("int8")  # 1-indexed
        class_probs = probs.reshape(H, W, NUM_CLASSES).astype("float32")
        confidence = probs.max(axis=1).reshape(H, W).astype("float32")

        # Per-branch maps
        rf_labels = self.rf.predict(flat).reshape(H, W).astype("int8")
        branch_maps = {"random_forest": rf_labels}

        return ClassificationResult(
            class_map=class_map,
            class_probs=class_probs,
            confidence=confidence,
            branch_maps=branch_maps,
            valid_mask=valid,
            year=year,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _stack_probs(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        cnn_embeddings: Optional[np.ndarray] = None,
        oof: bool = False,
    ) -> np.ndarray:
        """Build the stacked probability matrix from all active branches.

        For OOF training mode (``oof=True``) the function uses 5-fold OOF
        predictions to prevent data leakage in meta-learner training.
        Each branch contributes ``NUM_CLASSES`` columns → total width =
        (n_active_branches × NUM_CLASSES).
        """
        N = X.shape[0]
        parts: list[np.ndarray] = []

        # Branch 1: Random Forest
        if oof and y is not None:
            parts.append(self._oof_proba(self.rf, X, y))
        else:
            parts.append(self.rf.predict_proba(X))

        # Branch 2: Gradient Boosting on PCA
        X_s = self.scaler_gb.transform(X.astype("float32"))
        X_pca = self.pca_gb.transform(X_s)
        if oof and y is not None:
            parts.append(self._oof_proba(self.gb, X_pca, y))
        else:
            parts.append(self.gb.predict_proba(X_pca))

        # Branch 3: Quantum VQC
        if self.use_quantum and self.quantum_clf is not None:
            qr = self.quantum_clf.predict(X, year=0)  # type: ignore[union-attr]
            parts.append(
                qr.class_probs.reshape(N, NUM_CLASSES)  # type: ignore[union-attr]
            )

        # Branch 4: CNN linear SVC
        if self.use_cnn and self.cnn_svc is not None and cnn_embeddings is not None:
            emb_s = self.scaler_cnn.transform(cnn_embeddings.astype("float32"))
            # LinearSVC has no predict_proba; use decision function + softmax
            dec = self.cnn_svc.decision_function(emb_s)
            dec -= dec.max(axis=1, keepdims=True)
            exp_d = np.exp(dec)
            cnn_proba = exp_d / exp_d.sum(axis=1, keepdims=True)
            parts.append(cnn_proba.astype("float32"))

        if not parts:
            return np.ones((N, NUM_CLASSES), dtype="float32") / NUM_CLASSES

        return np.concatenate(parts, axis=1).astype("float32")

    def _meta_proba(self, meta_X: np.ndarray) -> np.ndarray:
        """Get probability predictions from the meta-learner."""
        return self.meta.predict_proba(meta_X).astype("float32")  # type: ignore[union-attr]

    @staticmethod
    def _oof_proba(
        model: RandomForestClassifier | GradientBoostingClassifier,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
    ) -> np.ndarray:
        """Generate out-of-fold probability predictions for meta-training."""
        from sklearn.model_selection import StratifiedKFold
        from copy import deepcopy

        oof_probs = np.zeros((len(X), NUM_CLASSES), dtype="float32")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        for train_idx, val_idx in skf.split(X, y):
            m = deepcopy(model)
            m.fit(X[train_idx], y[train_idx])
            proba = m.predict_proba(X[val_idx])
            # Align columns to full class set
            aligned = np.zeros((len(val_idx), NUM_CLASSES), dtype="float32")
            for col_i, cls in enumerate(m.classes_):
                if 0 <= cls < NUM_CLASSES:
                    aligned[:, int(cls)] = proba[:, col_i]
            oof_probs[val_idx] = aligned

        return oof_probs

    @staticmethod
    def _obia_smooth(
        labels: np.ndarray,
        probs: np.ndarray,
        segments: np.ndarray,
        n_pixels: int,
    ) -> np.ndarray:
        """Replace per-pixel label with the modal label within each segment."""
        from scipy.stats import mode
        unique_segs = np.unique(segments)
        out = labels.copy()
        for seg_id in unique_segs:
            if seg_id == 0:
                continue
            idx = segments == seg_id
            if idx.sum() == 0:
                continue
            seg_labels = labels[idx]
            modal_label = int(mode(seg_labels, keepdims=True).mode[0])
            out[idx] = modal_label
        return out

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, model_dir: Path) -> None:
        """Pickle all branch models and meta-learner to ``model_dir``."""
        import pickle

        model_dir.mkdir(parents=True, exist_ok=True)
        for name, obj in [
            ("rf.pkl", self.rf),
            ("gb.pkl", self.gb),
            ("pca_gb.pkl", self.pca_gb),
            ("scaler_gb.pkl", self.scaler_gb),
            ("scaler_cnn.pkl", self.scaler_cnn),
            ("meta.pkl", self.meta),
        ]:
            with open(model_dir / name, "wb") as f:
                pickle.dump(obj, f)
        if self.quantum_clf is not None:
            self.quantum_clf.save(model_dir / "quantum_params.npz")  # type: ignore[union-attr]
        logger.info("Ensemble models saved to %s", model_dir)

    def load(self, model_dir: Path) -> None:
        """Load all branch models from ``model_dir``."""
        import pickle

        for attr, name in [
            ("rf", "rf.pkl"), ("gb", "gb.pkl"),
            ("pca_gb", "pca_gb.pkl"), ("scaler_gb", "scaler_gb.pkl"),
            ("scaler_cnn", "scaler_cnn.pkl"), ("meta", "meta.pkl"),
        ]:
            p = model_dir / name
            if p.exists():
                with open(p, "rb") as f:
                    setattr(self, attr, pickle.load(f))

        qp = model_dir / "quantum_params.npz"
        if qp.exists() and self.quantum_clf is not None:
            self.quantum_clf.load(qp)  # type: ignore[union-attr]
        self._fitted = True
        logger.info("Ensemble models loaded from %s", model_dir)
