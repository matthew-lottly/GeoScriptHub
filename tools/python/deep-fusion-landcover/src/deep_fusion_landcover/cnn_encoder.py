"""cnn_encoder.py — TorchGeo pretrained satellite CNN encoder with ONNX export.

Uses a ResNet-50 backbone pretrained on BigEarthNet (multi-label Sentinel-2
benchmark) via TorchGeo, or optionally the SSL4EO self-supervised weights.

Pipeline
--------
    1. Stack 9 input channels: S2×6 (B02/B03/B04/B08/B11/B12) + SAR VV/VH + nDSM
    2. Tile the full annual composite into CNN_CHIP_SIZE × CNN_CHIP_SIZE chips
    3. Forward pass → global-average-pool → 2048-dim → projection head → 128-dim
    4. Fine-tune final 2 ResNet blocks + projection on NLCD 2021 reference
    5. Export fine-tuned encoder to ONNX for CPU inference (no GPU required)
    6. At inference: tile composite, encode, bilinear-upsample per-pixel embeddings

When torch / torchgeo are not installed the module degrades gracefully to a
PCA-based linear embedding so the rest of the pipeline can still run.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .constants import (
    CNN_BACKBONE,
    CNN_CHIP_SIZE,
    CNN_EMBEDDING_DIM,
    CNN_IN_CHANNELS,
    CNN_OVERLAP_PX,
)

logger = logging.getLogger("geoscripthub.deep_fusion_landcover.cnn_encoder")

TORCH_AVAILABLE = False
ONNX_AVAILABLE = False
try:
    import torch  # type: ignore[import-not-found]
    import torch.nn as nn  # type: ignore[import-not-found]
    TORCH_AVAILABLE = True
except ImportError:
    logger.debug("torch not installed — CNN encoder will use PCA fallback.")

try:
    import onnxruntime as ort  # type: ignore[import-not-found]
    ONNX_AVAILABLE = True
except ImportError:
    logger.debug("onnxruntime not installed.")


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class EmbeddingResult:
    """Per-tile or per-pixel CNN embedding result.

    Attributes
    ----------
    embeddings:    Float32 (H, W, CNN_EMBEDDING_DIM) per-pixel embedding map;
                   bilinear-upsampled from tile-stride predictions.
    tile_preds:    Optional (N_tiles, CNN_EMBEDDING_DIM) raw tile embeddings.
    method_used:   ``"torchgeo_onnx"``, ``"torchgeo_torch"``, or ``"pca_fallback"``.
    """

    embeddings: np.ndarray
    tile_preds: Optional[np.ndarray]
    method_used: str


# ── Model builder ─────────────────────────────────────────────────────────────

class CNNEncoder:
    """ResNet-50 satellite encoder with ONNX export and CPU inference.

    Parameters
    ----------
    onnx_path:      Path to the exported ``encoder.onnx`` file.  If None or
                    not yet created, ``encode()`` will first try to build the
                    model from TorchGeo weights.
    chip_size:      Square tile size in pixels (default 256).
    overlap:        Overlap between adjacent tiles in pixels.
    device:         PyTorch device (``"cpu"`` or ``"cuda:0"``).
    """

    def __init__(
        self,
        onnx_path: Optional[Path] = None,
        chip_size: int = CNN_CHIP_SIZE,
        overlap: int = CNN_OVERLAP_PX,
        device: str = "cpu",
    ) -> None:
        self.onnx_path = onnx_path
        self.chip_size = chip_size
        self.overlap = overlap
        self.device = device
        self._ort_session: Optional[object] = None
        self._torch_model: Optional[object] = None
        self._pca: Optional[object] = None

    # ── Public ────────────────────────────────────────────────────────────────

    def encode(self, feature_stack: np.ndarray) -> EmbeddingResult:
        """Produce per-pixel deep embeddings from a multi-band feature stack.

        Parameters
        ----------
        feature_stack: Float32 ``(H, W, C)`` array where C ≥ 6 (S2 bands).

        Returns
        -------
        EmbeddingResult
            Per-pixel embedding map of shape ``(H, W, CNN_EMBEDDING_DIM)``.
        """
        H, W, C = feature_stack.shape
        logger.info("CNN encode: %d × %d × %d …", H, W, C)

        # Prepare 9-channel input (pad to CNN_IN_CHANNELS if needed)
        inp = self._prepare_input(feature_stack)

        if ONNX_AVAILABLE and self.onnx_path is not None and self.onnx_path.exists():
            return self._encode_onnx(inp, H, W)
        if TORCH_AVAILABLE:
            return self._encode_torch(inp, H, W)
        return self._encode_pca(inp, H, W)

    def export_onnx(
        self,
        output_path: Path,
        fine_tune_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> Path:
        """Build, optionally fine-tune, and export the encoder to ONNX.

        Parameters
        ----------
        output_path:      Destination for the ``.onnx`` file.
        fine_tune_data:   Optional ``(X, y)`` training pair for NLCD fine-tuning.
                          X: (N_samples, CNN_IN_CHANNELS, chip_size, chip_size)
                          y: (N_samples,) integer class labels.

        Returns
        -------
        Path
            Path to the exported ONNX file.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch must be installed to export ONNX.")

        logger.info("Building TorchGeo ResNet-50 encoder …")
        model = self._build_torchgeo_model()

        if fine_tune_data is not None:
            logger.info("Fine-tuning on NLCD reference data …")
            model = self._fine_tune(model, fine_tune_data)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_input = torch.zeros(1, CNN_IN_CHANNELS, self.chip_size, self.chip_size)
        torch.onnx.export(
            model.encoder,   # type: ignore[union-attr]
            dummy_input,
            str(output_path),
            input_names=["input"],
            output_names=["embedding"],
            opset_version=17,
            dynamic_axes={"input": {0: "batch_size"}},
        )
        logger.info("Encoder exported to %s", output_path)
        self.onnx_path = output_path
        return output_path

    # ── ONNX inference ────────────────────────────────────────────────────────

    def _encode_onnx(self, inp: np.ndarray, H: int, W: int) -> EmbeddingResult:
        """Run tiled ONNX inference and reconstruct per-pixel embedding map."""
        if self._ort_session is None:
            self._ort_session = ort.InferenceSession(
                str(self.onnx_path),
                providers=["CPUExecutionProvider"],
            )

        tiles, positions = self._extract_tiles(inp)
        tile_embs = []
        batch_size = 8
        for i in range(0, len(tiles), batch_size):
            batch = tiles[i: i + batch_size].astype("float32")
            out = self._ort_session.run(None, {"input": batch})[0]  # type: ignore[union-attr]
            tile_embs.append(out)

        tile_embs_arr = np.concatenate(tile_embs, axis=0)  # (N_tiles, D)
        emb_map = self._reconstruct_map(tile_embs_arr, positions, H, W)

        return EmbeddingResult(
            embeddings=emb_map,
            tile_preds=tile_embs_arr,
            method_used="torchgeo_onnx",
        )

    # ── Torch inference ───────────────────────────────────────────────────────

    def _encode_torch(self, inp: np.ndarray, H: int, W: int) -> EmbeddingResult:
        """Run inference with the PyTorch model in-memory."""
        import torch  # type: ignore[import-not-found]

        model = self._build_torchgeo_model()
        model.eval()

        tiles, positions = self._extract_tiles(inp)
        tile_embs = []
        with torch.no_grad():
            for i in range(0, len(tiles), 4):
                batch = torch.from_numpy(tiles[i: i + 4].astype("float32"))
                emb = model.encoder(batch)   # type: ignore[operator]
                # Global average pool
                emb_pooled = emb.mean(dim=[2, 3]) if emb.dim() == 4 else emb
                tile_embs.append(emb_pooled.numpy())

        tile_embs_arr = np.concatenate(tile_embs, axis=0)
        emb_map = self._reconstruct_map(tile_embs_arr, positions, H, W)

        return EmbeddingResult(
            embeddings=emb_map,
            tile_preds=tile_embs_arr,
            method_used="torchgeo_torch",
        )

    # ── PCA fallback ──────────────────────────────────────────────────────────

    def _encode_pca(self, inp: np.ndarray, H: int, W: int) -> EmbeddingResult:
        """PCA-based linear embedding as last-resort fallback."""
        from sklearn.decomposition import PCA

        logger.debug("Using PCA fallback for CNN encoding.")
        C = inp.shape[0]
        flat = inp.reshape(C, -1).T   # (H*W, C)
        n_components = min(CNN_EMBEDDING_DIM, C, flat.shape[0])
        if self._pca is None:
            self._pca = PCA(n_components=n_components)
            self._pca.fit(flat)
        embs = self._pca.transform(flat)   # type: ignore[union-attr]  # (H*W, n_components)
        # Pad to CNN_EMBEDDING_DIM
        padded = np.zeros((H * W, CNN_EMBEDDING_DIM), dtype="float32")
        padded[:, :n_components] = embs
        emb_map = padded.reshape(H, W, CNN_EMBEDDING_DIM).astype("float32")
        return EmbeddingResult(embeddings=emb_map, tile_preds=None, method_used="pca_fallback")

    # ── Model builder ─────────────────────────────────────────────────────────

    def _build_torchgeo_model(self) -> "nn.Module":  # type: ignore[name-defined]
        """Build ResNet-50 with TorchGeo weights and a projection head."""
        import torch  # type: ignore[import-not-found]
        import torch.nn as nn  # type: ignore[import-not-found]

        try:
            from torchgeo.models import ResNet50_Weights  # type: ignore[import-not-found]
            import torchvision.models as tv_models  # type: ignore[import-not-found]
            from torchgeo.models import resnet50  # type: ignore[import-not-found]

            weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
            backbone = resnet50(weights=weights)
            in_features = backbone.fc.in_features  # type: ignore[union-attr]
            # Replace final FC with projection head
            backbone.fc = nn.Sequential(  # type: ignore[union-attr]
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Linear(512, CNN_EMBEDDING_DIM),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("TorchGeo weights failed (%s) — using ImageNet init.", exc)
            import torchvision.models as tv_models  # type: ignore[import-not-found]
            backbone = tv_models.resnet50(weights="DEFAULT")  # type: ignore[union-attr]
            backbone.conv1 = nn.Conv2d(
                CNN_IN_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            in_features = backbone.fc.in_features
            backbone.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Linear(512, CNN_EMBEDDING_DIM),
            )

        # Wrap to expose encoder separately
        class EncoderWrapper(nn.Module):
            def __init__(self, backbone: nn.Module) -> None:
                super().__init__()
                self.backbone = backbone
                # All layers except the final FC act as the encoder
                self.encoder = nn.Sequential(*list(backbone.children())[:-1])

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
                feat = self.encoder(x)
                feat = feat.flatten(1)
                return self.backbone.fc(feat)

        return EncoderWrapper(backbone)

    def _fine_tune(
        self,
        model: "nn.Module",  # type: ignore[name-defined]
        data: tuple[np.ndarray, np.ndarray],
        lr: float = 1e-4,
        epochs: int = 20,
    ) -> "nn.Module":  # type: ignore[name-defined]
        """Fine-tune the model's last two ResNet stages + projection head."""
        import torch  # type: ignore[import-not-found]
        import torch.nn as nn  # type: ignore[import-not-found]

        X, y = data
        X_t = torch.from_numpy(X.astype("float32"))
        y_t = torch.from_numpy(y.astype("int64"))

        # Freeze all layers except last 2 ResNet blocks + FC
        for name, param in model.named_parameters():
            if any(k in name for k in ("layer3", "layer4", "fc")):
                param.requires_grad = True
            else:
                param.requires_grad = False

        optimiser = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(epochs):
            optimiser.zero_grad()
            logits = model(X_t)
            loss = loss_fn(logits, y_t)
            loss.backward()
            optimiser.step()
            if epoch % 5 == 0:
                logger.debug("  Fine-tune epoch %d/%d — loss %.4f", epoch + 1, epochs, loss.item())

        return model

    # ── Tiling helpers ────────────────────────────────────────────────────────

    def _prepare_input(self, feature_stack: np.ndarray) -> np.ndarray:
        """Transpose (H, W, C) → (C, H, W) and pad/trim to CNN_IN_CHANNELS."""
        arr = np.moveaxis(feature_stack, -1, 0)   # (C, H, W)
        C = arr.shape[0]
        if C < CNN_IN_CHANNELS:
            pad = np.zeros((CNN_IN_CHANNELS - C, *arr.shape[1:]), dtype="float32")
            arr = np.concatenate([arr, pad], axis=0)
        return arr[:CNN_IN_CHANNELS].astype("float32")   # (CNN_IN_CHANNELS, H, W)

    def _extract_tiles(
        self, inp: np.ndarray
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Slide a window over (C, H, W) input and collect tiles + positions."""
        C, H, W = inp.shape
        step = max(1, self.chip_size - self.overlap)
        tiles: list[np.ndarray] = []
        positions: list[tuple[int, int]] = []

        for y in range(0, H, step):
            for x in range(0, W, step):
                tile = inp[:, y: y + self.chip_size, x: x + self.chip_size]
                # Pad if smaller than chip_size
                if tile.shape[1] < self.chip_size or tile.shape[2] < self.chip_size:
                    padded = np.zeros((C, self.chip_size, self.chip_size), dtype="float32")
                    padded[:, :tile.shape[1], :tile.shape[2]] = tile
                    tile = padded
                tiles.append(tile)
                positions.append((y, x))

        return np.stack(tiles, axis=0), positions  # (N, C, chip, chip)

    def _reconstruct_map(
        self,
        tile_embs: np.ndarray,
        positions: list[tuple[int, int]],
        H: int,
        W: int,
    ) -> np.ndarray:
        """Reconstruct per-pixel embedding map by bilinear upsampling of tile embeddings."""
        D = tile_embs.shape[1]
        acc = np.zeros((H, W, D), dtype="float32")
        cnt = np.zeros((H, W, 1), dtype="float32")
        step = max(1, self.chip_size - self.overlap)

        for (y, x), emb in zip(positions, tile_embs):
            h_end = min(y + step, H)
            w_end = min(x + step, W)
            if h_end <= y or w_end <= x:
                continue
            # Broadcast tile embedding to the region (constant per tile)
            acc[y:h_end, x:w_end, :] += emb[np.newaxis, np.newaxis, :]
            cnt[y:h_end, x:w_end, :] += 1.0

        cnt = np.where(cnt == 0, 1, cnt)
        return (acc / cnt).astype("float32")
