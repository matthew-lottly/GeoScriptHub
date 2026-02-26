"""
OSM Change Monitor — Scheduler
================================
Wraps :class:`~osm_change_monitor.monitor.OSMChangeMonitor` in a ``schedule``-
based polling loop so it can run continuously as a background service or cron
replacement.

Usage::

    from pathlib import Path
    from osm_change_monitor.monitor import OSMChangeMonitor, BoundingBox
    from osm_change_monitor.scheduler import MonitorScheduler

    monitor = OSMChangeMonitor(
        bbox=BoundingBox(51.47, -0.15, 51.52, -0.08),
        osm_tag="amenity=hospital",
        output_dir=Path("data/monitor"),
    )
    # Poll every 60 minutes until Ctrl-C
    MonitorScheduler(monitor, interval_minutes=60).start()
"""

from __future__ import annotations

import logging
import signal
import sys
import time

try:
    import schedule  # type: ignore[import-untyped]
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("schedule is required: pip install schedule") from exc

from osm_change_monitor.monitor import OSMChangeMonitor

logger = logging.getLogger("geoscripthub.osm_change_monitor.scheduler")


class MonitorScheduler:
    """Schedule repeated polling of an :class:`OSMChangeMonitor`.

    Intercepts SIGINT/SIGTERM for graceful shutdown so that the in-progress
    poll is allowed to complete before the process exits.

    Args:
        monitor: The configured :class:`OSMChangeMonitor` to poll.
        interval_minutes: How often to poll, in minutes.
                          <!-- PLACEHOLDER: Set to your desired poll frequency,
                               e.g. 30 for every 30 minutes.
                               Note: Overpass public API asks for ≥ 1 minute intervals. -->
        run_immediately: If ``True`` (default), execute one poll immediately on
                         :meth:`start` before scheduling subsequent runs.
                         <!-- PLACEHOLDER: set to False to skip the initial poll -->
    """

    def __init__(
        self,
        monitor: OSMChangeMonitor,
        interval_minutes: int = 60,  # PLACEHOLDER: adjust to your poll frequency
        *,
        run_immediately: bool = True,
    ) -> None:
        if interval_minutes < 1:
            raise ValueError("interval_minutes must be ≥ 1")
        self.monitor = monitor
        self.interval_minutes = interval_minutes
        self.run_immediately = run_immediately
        self._running = False

    def _safe_run(self) -> None:
        """Execute a single monitor poll, catching all exceptions to keep the loop alive."""
        try:
            self.monitor.run()
        except Exception as exc:  # noqa: BLE001
            logger.error("Scheduled poll failed: %s", exc)

    def start(self) -> None:
        """Begin the scheduling loop (blocking).

        Runs until interrupted by SIGINT (Ctrl-C) or SIGTERM.
        """
        self._running = True

        # Register graceful shutdown handlers
        def _shutdown(signum: int, frame: object) -> None:
            logger.info("Received signal %d, stopping scheduler...", signum)
            self._running = False

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        if self.run_immediately:
            logger.info("Running initial poll...")
            self._safe_run()

        schedule.every(self.interval_minutes).minutes.do(self._safe_run)
        logger.info(
            "Scheduler started: polling every %d minute(s). Press Ctrl-C to stop.",
            self.interval_minutes,
        )

        while self._running:
            schedule.run_pending()
            time.sleep(10)

        logger.info("Scheduler stopped.")
        schedule.clear()
