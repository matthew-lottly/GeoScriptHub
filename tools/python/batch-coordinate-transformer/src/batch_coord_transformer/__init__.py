"""
Batch Coordinate Transformer
=============================
A GeoScriptHub tool for reprojecting CSV coordinate data between any two
coordinate reference systems supported by pyproj.

Public API::

    from src.batch_coord_transformer import CoordinateTransformer, TransformerConfig
"""

from src.batch_coord_transformer.transformer import CoordinateTransformer, TransformerConfig

__all__ = ["CoordinateTransformer", "TransformerConfig"]
__version__ = "1.0.0"
