"""
quantum_flood_frequency
=======================
Pseudo-Quantum Hybrid AI Flood Frequency Mapper v2.0 for Houston, TX.

Fuses multi-temporal Landsat-8/9, Sentinel-2, and NAIP imagery from
Microsoft Planetary Computer to produce per-pixel flood inundation
frequency maps at **10 m** resolution using super-resolution upsampling.

A novel 3-qubit pseudo-quantum-inspired classification engine employs
8-state superposition probability amplitudes, variational quantum
circuits, quantum kernel estimation, and ensemble AI with meta-learner
stacking — then aggregates observations into a temporal frequency surface.

Submodules
----------
aoi                 -- Define study-area AOI (Houston, TX)
acquisition         -- Multi-sensor STAC imagery acquisition
preprocessing       -- Super-resolution upsampling, co-registration, cloud masking
super_resolution    -- Multi-sensor SR engine (bicubic, spectral-guided, learned SISR)
model_optimization  -- ONNX export, INT8 quantisation, tiled inference
quantum_classifier  -- 3-qubit pseudo-quantum + AI hybrid water/flood classifier
flood_engine        -- Temporal frequency aggregation and statistics
fema                -- FEMA National Flood Hazard Layer overlay
viz                 -- Cartographic output: frequency maps, FEMA comparison
cli                 -- Click command-line interface
"""

from .aoi import AOIBuilder, AOIResult
from .acquisition import MultiSensorAcquisition, SensorStack
from .preprocessing import ImagePreprocessor, AlignedStack
from .super_resolution import SuperResolutionEngine, SRMethod, SRResult
from .model_optimization import ONNXOptimizer, TiledInferenceEngine, BatchPredictor
from .quantum_classifier import QuantumHybridClassifier, ClassificationResult
from .flood_engine import FloodFrequencyEngine, FrequencyResult
from .fema import FEMAFloodZones
from .viz import FloodMapper

__version__ = "2.0.0"
__all__ = [
    "AOIBuilder",
    "AOIResult",
    "MultiSensorAcquisition",
    "SensorStack",
    "ImagePreprocessor",
    "AlignedStack",
    "SuperResolutionEngine",
    "SRMethod",
    "SRResult",
    "ONNXOptimizer",
    "TiledInferenceEngine",
    "BatchPredictor",
    "QuantumHybridClassifier",
    "ClassificationResult",
    "FloodFrequencyEngine",
    "FrequencyResult",
    "FEMAFloodZones",
    "FloodMapper",
]
