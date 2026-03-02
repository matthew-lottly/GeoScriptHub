"""
gauge_data.py
=============
USGS National Water Information System (NWIS) gauge data integration.

Fetches real-time and historical water-level (gage height) and
discharge data from USGS stream gauges near the study area for
accuracy assessment of flood classification.

For each satellite observation date, retrieves gauge readings from
the day before, the day of, and the day after to capture flood
conditions contemporaneous with the imagery.

Data source: USGS NWIS Instantaneous Values Web Service
    https://waterservices.usgs.gov/nwis/iv/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger("geoscripthub.quantum_flood_frequency.gauge_data")

# USGS NWIS endpoints
NWIS_SITE_URL = "https://waterservices.usgs.gov/nwis/site/"
NWIS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
NWIS_DV_URL = "https://waterservices.usgs.gov/nwis/dv/"

# USGS parameter codes
GAGE_HEIGHT_CODE = "00065"   # Gage height (ft)
DISCHARGE_CODE = "00060"     # Discharge (ft³/s)
WATER_LEVEL_CODE = "72019"   # Water level below land surface (ft)

# Maximum search radius for gauge stations (degrees)
GAUGE_SEARCH_RADIUS_DEG = 0.15  # ~16 km


@dataclass
class GaugeReading:
    """A single gauge reading for a specific date."""
    site_id: str
    site_name: str
    latitude: float
    longitude: float
    date: str
    gage_height_ft: Optional[float] = None
    discharge_cfs: Optional[float] = None
    gage_height_m: Optional[float] = None
    flood_stage_ft: Optional[float] = None
    is_flooding: bool = False


@dataclass
class GaugeStation:
    """USGS gauge station metadata."""
    site_id: str
    site_name: str
    latitude: float
    longitude: float
    site_type: str = ""
    drainage_area_sqmi: Optional[float] = None


@dataclass
class GaugeValidation:
    """Validation data pairing gauge readings with satellite observations."""
    observation_date: str
    sensor: str
    stations: list[GaugeStation] = field(default_factory=list)
    readings_before: list[GaugeReading] = field(default_factory=list)
    readings_dayof: list[GaugeReading] = field(default_factory=list)
    readings_after: list[GaugeReading] = field(default_factory=list)
    max_gage_height_ft: Optional[float] = None
    any_flooding: bool = False

    def summary(self) -> str:
        """Human-readable summary of gauge conditions."""
        if not self.readings_dayof:
            return f"{self.observation_date}: no gauge data"
        heights = [r.gage_height_ft for r in self.readings_dayof if r.gage_height_ft is not None]
        if not heights:
            return f"{self.observation_date}: gauge data but no gage height"
        return (
            f"{self.observation_date}: "
            f"gage_ht={max(heights):.2f} ft "
            f"({max(heights)*0.3048:.2f} m), "
            f"n_stations={len(self.readings_dayof)}, "
            f"flooding={'YES' if self.any_flooding else 'no'}"
        )


class USGSGaugeData:
    """Fetch and manage USGS gauge data for flood validation.

    Parameters
    ----------
    center_lat:
        Centre latitude of the study area (WGS-84).
    center_lon:
        Centre longitude of the study area (WGS-84).
    search_radius_deg:
        Search radius in degrees for finding nearby gauges.
    """

    def __init__(
        self,
        center_lat: float,
        center_lon: float,
        search_radius_deg: float = GAUGE_SEARCH_RADIUS_DEG,
    ) -> None:
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.search_radius_deg = search_radius_deg
        self._stations: list[GaugeStation] = []
        self._session = None

    def _get_session(self):
        """Lazy-init a requests session with retry strategy."""
        if self._session is None:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            session = requests.Session()
            retry = Retry(total=3, backoff_factor=1.0, status_forcelist=[500, 502, 503])
            session.mount("https://", HTTPAdapter(max_retries=retry))
            self._session = session
        return self._session

    def discover_stations(self) -> list[GaugeStation]:
        """Find USGS gauge stations near the study area.

        Returns:
            List of GaugeStation objects within the search radius.
        """
        logger.info(
            "Searching for USGS gauge stations near (%.4f, %.4f) "
            "within %.3f°",
            self.center_lat, self.center_lon, self.search_radius_deg,
        )

        bbox = (
            self.center_lon - self.search_radius_deg,
            self.center_lat - self.search_radius_deg,
            self.center_lon + self.search_radius_deg,
            self.center_lat + self.search_radius_deg,
        )

        session = self._get_session()

        params = {
            "format": "rdb",
            "bBox": f"{bbox[0]:.6f},{bbox[1]:.6f},{bbox[2]:.6f},{bbox[3]:.6f}",
            "siteType": "ST",  # Stream sites
            "siteStatus": "all",
            "hasDataTypeCd": "iv",  # sites with instantaneous values
            "parameterCd": GAGE_HEIGHT_CODE,
        }

        try:
            resp = session.get(NWIS_SITE_URL, params=params, timeout=30)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("USGS station query failed: %s", exc)
            return []

        stations = self._parse_site_rdb(resp.text)
        self._stations = stations
        logger.info("Found %d USGS gauge stations", len(stations))
        for s in stations:
            logger.debug("  %s: %s (%.4f, %.4f)", s.site_id, s.site_name, s.latitude, s.longitude)

        return stations

    def fetch_readings_for_date(
        self,
        date_str: str,
        window_days: int = 1,
    ) -> GaugeValidation:
        """Fetch gauge readings for a date ± window.

        Args:
            date_str: ISO date string (YYYY-MM-DD).
            window_days: Days before/after to fetch (default 1).

        Returns:
            GaugeValidation with readings from day-before, day-of, day-after.
        """
        if not self._stations:
            self.discover_stations()

        if not self._stations:
            return GaugeValidation(
                observation_date=date_str,
                sensor="unknown",
            )

        dt = datetime.strptime(date_str, "%Y-%m-%d")
        start = dt - timedelta(days=window_days)
        end = dt + timedelta(days=window_days)

        site_ids = [s.site_id for s in self._stations]

        # Fetch daily values (more reliable for historical data)
        readings = self._fetch_daily_values(
            site_ids,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        )

        # Split into before/dayof/after
        before = [r for r in readings if r.date < date_str]
        dayof = [r for r in readings if r.date == date_str]
        after = [r for r in readings if r.date > date_str]

        max_ht = None
        any_flood = False
        all_readings = before + dayof + after
        if all_readings:
            heights = [r.gage_height_ft for r in all_readings if r.gage_height_ft is not None]
            if heights:
                max_ht = max(heights)
            any_flood = any(r.is_flooding for r in all_readings)

        return GaugeValidation(
            observation_date=date_str,
            sensor="unknown",
            stations=self._stations,
            readings_before=before,
            readings_dayof=dayof,
            readings_after=after,
            max_gage_height_ft=max_ht,
            any_flooding=any_flood,
        )

    def fetch_all_observation_dates(
        self,
        observation_dates: list[tuple[str, str]],
    ) -> list[GaugeValidation]:
        """Fetch gauge data for all observation dates.

        Args:
            observation_dates: List of (date_str, sensor_name) tuples.

        Returns:
            List of GaugeValidation, one per observation date.
        """
        if not self._stations:
            self.discover_stations()

        logger.info(
            "Fetching gauge data for %d observation dates from %d stations",
            len(observation_dates), len(self._stations),
        )

        # Batch by month to reduce API calls
        results = []
        unique_dates = sorted(set(d for d, _ in observation_dates))

        if not unique_dates:
            return results

        # Fetch the full date range at once for efficiency
        start_date = min(unique_dates)
        end_date = max(unique_dates)

        # Expand window by 1 day on each side
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=1)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        site_ids = [s.site_id for s in self._stations]

        all_readings = self._fetch_daily_values(
            site_ids,
            start_dt.strftime("%Y-%m-%d"),
            end_dt.strftime("%Y-%m-%d"),
        )

        # Index readings by date for fast lookup
        readings_by_date: dict[str, list[GaugeReading]] = {}
        for r in all_readings:
            readings_by_date.setdefault(r.date, []).append(r)

        for date_str, sensor in observation_dates:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            day_before = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
            day_after = (dt + timedelta(days=1)).strftime("%Y-%m-%d")

            before = readings_by_date.get(day_before, [])
            dayof = readings_by_date.get(date_str, [])
            after = readings_by_date.get(day_after, [])

            max_ht = None
            any_flood = False
            all_r = before + dayof + after
            if all_r:
                heights = [r.gage_height_ft for r in all_r if r.gage_height_ft is not None]
                if heights:
                    max_ht = max(heights)
                any_flood = any(r.is_flooding for r in all_r)

            results.append(GaugeValidation(
                observation_date=date_str,
                sensor=sensor,
                stations=self._stations,
                readings_before=before,
                readings_dayof=dayof,
                readings_after=after,
                max_gage_height_ft=max_ht,
                any_flooding=any_flood,
            ))

        logger.info(
            "Gauge data retrieved: %d dates, %d total readings",
            len(results), len(all_readings),
        )

        return results

    def _fetch_daily_values(
        self,
        site_ids: list[str],
        start_date: str,
        end_date: str,
    ) -> list[GaugeReading]:
        """Fetch daily mean values from USGS NWIS.

        Uses the daily values (dv) service which has better coverage
        for historical data than the instantaneous values service.
        """
        if not site_ids:
            return []

        session = self._get_session()
        station_map = {s.site_id: s for s in self._stations}

        # NWIS limits site list length; batch if needed
        batch_size = 100
        all_readings: list[GaugeReading] = []

        for i in range(0, len(site_ids), batch_size):
            batch = site_ids[i:i + batch_size]

            params = {
                "format": "rdb",
                "sites": ",".join(batch),
                "startDT": start_date,
                "endDT": end_date,
                "parameterCd": f"{GAGE_HEIGHT_CODE},{DISCHARGE_CODE}",
                "statCd": "00003",  # Mean daily
            }

            try:
                resp = session.get(NWIS_DV_URL, params=params, timeout=60)
                resp.raise_for_status()
                readings = self._parse_dv_rdb(resp.text, station_map)
                all_readings.extend(readings)
            except Exception as exc:
                logger.warning(
                    "USGS daily values fetch failed for batch %d: %s",
                    i // batch_size, exc,
                )

        return all_readings

    def _parse_site_rdb(self, text: str) -> list[GaugeStation]:
        """Parse USGS RDB-format site data."""
        stations = []
        header_found = False
        col_names: list[str] = []

        for line in text.splitlines():
            if line.startswith("#"):
                continue
            if not header_found:
                col_names = line.split("\t")
                header_found = True
                continue
            if line.startswith("5s") or line.startswith("15s") or all(c in "-s\td" for c in line.replace("\t", "")):
                continue  # skip type-definition line

            parts = line.split("\t")
            if len(parts) < 4:
                continue

            try:
                col_map = {name: parts[i] if i < len(parts) else "" for i, name in enumerate(col_names)}
                site_id = col_map.get("site_no", "").strip()
                site_name = col_map.get("station_nm", "").strip()
                lat_str = col_map.get("dec_lat_va", "")
                lon_str = col_map.get("dec_long_va", "")

                if not site_id or not lat_str or not lon_str:
                    continue

                stations.append(GaugeStation(
                    site_id=site_id,
                    site_name=site_name,
                    latitude=float(lat_str),
                    longitude=float(lon_str),
                    site_type=col_map.get("site_tp_cd", ""),
                    drainage_area_sqmi=float(col_map["drain_area_va"]) if col_map.get("drain_area_va", "").strip() else None,
                ))
            except (ValueError, KeyError) as exc:
                logger.debug("Skipping malformed station line: %s", exc)
                continue

        return stations

    def _parse_dv_rdb(
        self,
        text: str,
        station_map: dict[str, GaugeStation],
    ) -> list[GaugeReading]:
        """Parse USGS RDB-format daily values data."""
        readings = []
        header_found = False
        col_names: list[str] = []

        for line in text.splitlines():
            if line.startswith("#"):
                continue
            if not header_found:
                col_names = line.split("\t")
                header_found = True
                continue
            # Skip type-def line (e.g., "5s\t15s\t20d\t...")
            if all(c in "0123456789sdnta\t " for c in line) and "\t" in line:
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue

            try:
                col_map = {name: parts[i] if i < len(parts) else "" for i, name in enumerate(col_names)}
                site_id = col_map.get("site_no", "").strip()
                date_str = col_map.get("datetime", "").strip()

                if not site_id or not date_str:
                    continue

                station = station_map.get(site_id)
                if station is None:
                    continue

                # Parse gage height — try multiple column name patterns
                gage_ht = None
                for key in col_map:
                    if GAGE_HEIGHT_CODE in key and "_00003" in key and "cd" not in key.lower():
                        try:
                            val = col_map[key].strip()
                            if val and val not in ("", "Ice", "Eqp", "Ssn", "***"):
                                gage_ht = float(val)
                        except ValueError:
                            pass
                        break

                # Parse discharge
                discharge = None
                for key in col_map:
                    if DISCHARGE_CODE in key and "_00003" in key and "cd" not in key.lower():
                        try:
                            val = col_map[key].strip()
                            if val and val not in ("", "Ice", "Eqp", "Ssn", "***"):
                                discharge = float(val)
                        except ValueError:
                            pass
                        break

                if gage_ht is None and discharge is None:
                    continue

                readings.append(GaugeReading(
                    site_id=site_id,
                    site_name=station.site_name,
                    latitude=station.latitude,
                    longitude=station.longitude,
                    date=date_str,
                    gage_height_ft=gage_ht,
                    discharge_cfs=discharge,
                    gage_height_m=gage_ht * 0.3048 if gage_ht is not None else None,
                ))

            except (ValueError, KeyError) as exc:
                logger.debug("Skipping malformed DV line: %s", exc)
                continue

        return readings

    def save_gauge_report(
        self,
        validations: list[GaugeValidation],
        output_path,
    ) -> None:
        """Save gauge validation data as a CSV report."""
        from pathlib import Path
        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "observation_date", "sensor", "site_id", "site_name",
                "latitude", "longitude", "window",
                "gage_height_ft", "gage_height_m", "discharge_cfs",
                "any_flooding",
            ])

            for v in validations:
                for label, readings in [
                    ("day_before", v.readings_before),
                    ("day_of", v.readings_dayof),
                    ("day_after", v.readings_after),
                ]:
                    for r in readings:
                        writer.writerow([
                            v.observation_date, v.sensor,
                            r.site_id, r.site_name,
                            f"{r.latitude:.6f}", f"{r.longitude:.6f}",
                            label,
                            f"{r.gage_height_ft:.2f}" if r.gage_height_ft is not None else "",
                            f"{r.gage_height_m:.3f}" if r.gage_height_m is not None else "",
                            f"{r.discharge_cfs:.1f}" if r.discharge_cfs is not None else "",
                            v.any_flooding,
                        ])

        logger.info("Gauge validation report saved → %s (%d dates)", output_path, len(validations))
