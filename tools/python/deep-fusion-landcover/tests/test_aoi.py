"""test_aoi.py — Tests for AOIBuilder and tile_aoi."""
from __future__ import annotations

from shapely.geometry import box

from deep_fusion_landcover.aoi import AOIBuilder, AustinMetroAOI
from deep_fusion_landcover.constants import AUSTIN_CENTER_LAT, AUSTIN_CENTER_LON


def test_aoi_builder_returns_result():
    result = AOIBuilder(
        center_lat=AUSTIN_CENTER_LAT,
        center_lon=AUSTIN_CENTER_LON,
        buffer_km=5.0,
    ).build()
    assert result.target_crs == "EPSG:32614"
    assert result.utm_poly.is_valid
    assert result.utm_poly.area > 0


def test_bbox_wgs84_contains_austin(austin_aoi_result):
    minx, miny, maxx, maxy = austin_aoi_result.bbox_wgs84
    assert minx < AUSTIN_CENTER_LON < maxx
    assert miny < AUSTIN_CENTER_LAT < maxy


def test_austin_metro_aoi_convenience():
    result = AustinMetroAOI()
    # Buffer is 55 km → expect very large polygon
    area_km2 = result.utm_poly.area / 1e6
    assert area_km2 > 1000  # at least 1000 km²


def test_tile_aoi_produces_tiles(austin_aoi_result):
    tiles = AOIBuilder().tile_aoi(austin_aoi_result, tile_size_px=256, resolution_m=10)
    assert len(tiles) > 0
    # Each tile must have a valid UTM bbox
    for t in tiles:
        minx, miny, maxx, maxy = t.bbox_utm
        assert maxx > minx
        assert maxy > miny


def test_tile_aoi_covers_full_extent(austin_aoi_result):
    """Union of tile boxes should cover the AOI bounding box."""
    tiles = AOIBuilder().tile_aoi(austin_aoi_result, tile_size_px=256, resolution_m=10)
    from shapely.ops import unary_union

    tile_polys = [box(*t.bbox_utm) for t in tiles]
    union = unary_union(tile_polys)
    aoi_box = box(*austin_aoi_result.bbox_utm)
    # Allow ±5% coverage tolerance
    coverage = union.intersection(aoi_box).area / aoi_box.area
    assert coverage > 0.90
