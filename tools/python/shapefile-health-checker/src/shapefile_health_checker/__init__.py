"""
Shapefile Health Checker
=========================
A GeoScriptHub tool for validating vector geospatial files against a
configurable suite of health checks.

Public API::

    from src.shapefile_health_checker import ShapefileHealthChecker
"""

from src.shapefile_health_checker.checker import (
    CheckResult,
    CheckStatus,
    CheckStrategy,
    DEFAULT_CHECKS,
    HealthReport,
    ShapefileHealthChecker,
)

__all__ = [
    "ShapefileHealthChecker",
    "HealthReport",
    "CheckResult",
    "CheckStatus",
    "CheckStrategy",
    "DEFAULT_CHECKS",
]
__version__ = "1.0.0"
