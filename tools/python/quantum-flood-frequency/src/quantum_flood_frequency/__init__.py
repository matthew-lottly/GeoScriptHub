"""
quantum_flood_frequency
=======================
Pseudo-Quantum Hybrid AI Flood Frequency Mapper for the Mississippi River.

Fuses multi-temporal Landsat-8/9, Sentinel-2, and NAIP imagery from
Microsoft Planetary Computer to produce per-pixel flood inundation
frequency maps.  A novel pseudo-quantum-inspired classification engine
employs superposition-state probability amplitudes, quantum kernel
estimation, and ensemble AI to detect water with sub-pixel confidence —
then aggregates observations into a temporal frequency surface.

Submodules
----------
aoi             -- Define study-area AOI (Mississippi River delta)
acquisition     -- Multi-sensor STAC imagery acquisition
preprocessing   -- Resampling, co-registration, cloud masking, alignment
quantum_classifier -- Pseudo-quantum + AI hybrid water/flood classifier
flood_engine    -- Temporal frequency aggregation and statistics
fema            -- FEMA National Flood Hazard Layer overlay
viz             -- Cartographic output: frequency maps, FEMA comparison
cli             -- Click command-line interface
"""

from .aoi import AOIBuilder, AOIResult
from .acquisition import MultiSensorAcquisition, SensorStack
from .preprocessing import ImagePreprocessor, AlignedStack
from .quantum_classifier import QuantumHybridClassifier, ClassificationResult
from .flood_engine import FloodFrequencyEngine, FrequencyResult
from .fema import FEMAFloodZones
from .viz import FloodMapper

__version__ = "1.0.0"
__all__ = [
    "AOIBuilder",
    "AOIResult",
    "MultiSensorAcquisition",
    "SensorStack",
    "ImagePreprocessor",
    "AlignedStack",
    "QuantumHybridClassifier",
    "ClassificationResult",
    "FloodFrequencyEngine",
    "FrequencyResult",
    "FEMAFloodZones",
    "FloodMapper",
]
