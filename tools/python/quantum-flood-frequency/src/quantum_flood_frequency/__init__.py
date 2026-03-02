"""
quantum_flood_frequency
=======================
Pseudo-Quantum Hybrid AI Flood Frequency Mapper v3.0 for Houston, TX.

Fuses multi-temporal Landsat-8/9, Sentinel-2, NAIP, **Sentinel-1 SAR**,
and **Copernicus DEM** imagery from Microsoft Planetary Computer to
produce per-pixel flood inundation frequency maps at **1 m** resolution
using super-resolution upsampling, SAR-based water/building discrimination,
and DEM-based terrain constraints.

A novel 3-qubit pseudo-quantum-inspired classification engine employs
8-state superposition probability amplitudes, variational quantum
circuits, quantum kernel estimation, and ensemble AI with meta-learner
stacking — enhanced with SAR, terrain, NDBI urban masking, and
morphological refinement — then aggregates observations into a temporal
frequency surface.

Submodules
----------
aoi                 -- Define study-area AOI (Houston, TX)
acquisition         -- Multi-sensor STAC imagery acquisition (Landsat, S2, NAIP, S1, DEM)
preprocessing       -- Super-resolution upsampling, co-registration, cloud masking
super_resolution    -- Multi-sensor SR engine (bicubic, spectral-guided, learned SISR)
model_optimization  -- ONNX export, INT8 quantisation, tiled inference
sar_processor       -- Sentinel-1 SAR speckle filtering, backscatter, water/building masks
terrain             -- DEM slope, HAND computation, flood susceptibility
mosaic              -- Temporal mosaicking engine (median, mean, best-pixel composites)
tiled_pipeline      -- Tiled processing orchestrator for large 1 m rasters
quantum_classifier  -- 3-qubit pseudo-quantum + AI + SAR + terrain classifier
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
from .sar_processor import SARProcessor, SARFeatures
from .terrain import TerrainProcessor, TerrainFeatures
from .mosaic import TemporalMosaicker, MosaicResult
from .tiled_pipeline import TileGrid, TiledAccumulator, BatchScheduler
from .quantum_classifier import QuantumHybridClassifier, ClassificationResult
from .flood_engine import FloodFrequencyEngine, FrequencyResult
from .fema import FEMAFloodZones
from .viz import FloodMapper

__version__ = "3.0.0"
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
    "SARProcessor",
    "SARFeatures",
    "TerrainProcessor",
    "TerrainFeatures",
    "TemporalMosaicker",
    "MosaicResult",
    "TileGrid",
    "TiledAccumulator",
    "BatchScheduler",
    "QuantumHybridClassifier",
    "ClassificationResult",
    "FloodFrequencyEngine",
    "FrequencyResult",
    "FEMAFloodZones",
    "FloodMapper",
]
