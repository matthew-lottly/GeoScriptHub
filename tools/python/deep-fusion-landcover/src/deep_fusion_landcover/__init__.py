"""deep_fusion_landcover — Deep-Fusion Multi-Sensor Landcover Mapper.

Fuses Landsat 4/5/7/8/9 (1990-2025), Sentinel-1/2 (2014-2025),
NAIP (2003-2023), 3DEP LiDAR, GEDI canopy height, and NLCD reference
data into annual landcover maps for Austin, TX at dual resolution
(10 m change history + 1 m present-day NAIP layer).

Novel contributions
-------------------
- 5-branch stacked ensemble: Random Forest, Gradient Boosting,
  8-qubit pseudo-quantum VQC, TorchGeo CNN (ONNX), SAM2 OBIA
- Sub-canopy detection: LiDAR penetration + SAR double-bounce +
  thermal anomaly + HAND-based inundation
- Cross-sensor temporal harmonisation (Roy et al. 2016 OLI→TM)
- Medoid cloud-free annual compositing
- 12-class Austin-specific scheme (Ashe Juniper, cedar brush,
  Hill Country oak, limestone quarry, etc.)

Submodules
----------
constants           Class definitions, sensor configs, colour maps, thresholds.
aoi                 AOI builder — WGS-84 + UTM 14N projection.
acquisition         Multi-sensor STAC fetcher (PC + USGS + NASA EarthData).
preprocessing       Cloud masking, band harmonisation, co-registration.
temporal_compositing  Annual medoid compositing, SLC-off gap fill, STARFM fusion.
feature_engineering Spectral indices, SAR, texture, LiDAR, terrain, phenology.
lidar_processor     3DEP point-cloud DTM/DSM/nDSM/penetration-ratio extraction.
sub_canopy_detector Under-canopy structure detection (4 evidence lines).
sam_segmenter       SAM2 object-based segmentation on NAIP tiles.
cnn_encoder         TorchGeo ResNet-50 / SSL4EO encoder + ONNX export.
quantum_classifier  8-qubit pseudo-quantum VQC feature encoder.
ensemble            5-branch stacked meta-classifier (LightGBM stacker).
change_detection    Annual transition matrices, trajectory analysis, hotspots.
mosaic              Dask-parallel tiled mosaic engine for large AOI.
accuracy            NLCD cross-validation, Kappa, confusion matrix.
viz                 Folium dual-panel map, Sankey diagrams, change charts.
flowchart           Pipeline architecture HTML diagram.
cli                 Click command-line interface.
"""
from __future__ import annotations

from .aoi import AOIBuilder, AOIResult, AustinMetroAOI
from .acquisition import MultiSensorAcquisition, SensorStack
from .preprocessing import Preprocessor, AlignedStack
from .temporal_compositing import TemporalCompositor, AnnualComposite
from .feature_engineering import FeatureEngineer, FeatureStack
from .lidar_processor import LiDARProcessor, LiDARProducts
from .sub_canopy_detector import SubCanopyDetector, SubCanopyResult
from .sam_segmenter import SAMSegmenter, SegmentFeatures
from .cnn_encoder import CNNEncoder, EmbeddingResult
from .quantum_classifier import QuantumVQCClassifier
from .ensemble import EnsembleClassifier, ClassificationResult
from .change_detection import ChangeDetector, ChangeResult
from .mosaic import TiledMosaicEngine, MosaicResult
from .accuracy import AccuracyAssessor, AccuracyReport
from .viz import build_dual_map, build_sankey_html, generate_all_outputs
from .flowchart import generate_flowchart

__version__ = "1.0.0"
__all__ = [
    "AOIBuilder", "AOIResult", "AustinMetroAOI",
    "MultiSensorAcquisition", "SensorStack",
    "Preprocessor", "AlignedStack",
    "TemporalCompositor", "AnnualComposite",
    "FeatureEngineer", "FeatureStack",
    "LiDARProcessor", "LiDARProducts",
    "SubCanopyDetector", "SubCanopyResult",
    "SAMSegmenter", "SegmentFeatures",
    "CNNEncoder", "EmbeddingResult",
    "QuantumVQCClassifier",
    "EnsembleClassifier", "ClassificationResult",
    "ChangeDetector", "ChangeResult",
    "TiledMosaicEngine", "MosaicResult",
    "AccuracyAssessor", "AccuracyReport",
    "build_dual_map", "build_sankey_html", "generate_all_outputs",
    "generate_flowchart",
]
