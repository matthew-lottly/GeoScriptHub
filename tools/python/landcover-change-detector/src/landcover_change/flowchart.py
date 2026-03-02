"""Generate pipeline flowchart in HTML, PNG, and PDF formats.

v1.0 — Quantum Land-Cover Change Detector
Produces a complete pipeline architecture diagram showing all data
flows, processing steps, classification methods, and outputs.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger("geoscripthub.landcover_change.flowchart")

# Absolute path to the static HTML flowchart shipped with the package
_HTML_SRC = Path(__file__).resolve().parent.parent.parent / "pipeline_flowchart.html"


def generate_flowchart(
    output_dir: Path,
    *,
    html: bool = True,
    png: bool = True,
    pdf: bool = True,
) -> dict[str, Optional[Path]]:
    """Copy/render the pipeline flowchart into *output_dir*.

    Parameters
    ----------
    output_dir:
        Target directory (created if missing).
    html:
        Emit ``pipeline_flowchart.html``.
    png:
        Emit ``pipeline_flowchart.png`` via Playwright or fall back to
        a note that a browser screenshot is required.
    pdf:
        Emit ``pipeline_flowchart.pdf`` via Playwright or wkhtmltopdf.

    Returns
    -------
    dict mapping format name to the file path produced (or ``None``
    if that format could not be generated).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Optional[Path]] = {"html": None, "png": None, "pdf": None}

    # ── HTML ──────────────────────────────────────────────────────────
    if html:
        dst = output_dir / "pipeline_flowchart.html"
        if _HTML_SRC.exists():
            shutil.copy2(_HTML_SRC, dst)
        else:
            # Fallback: write a minimal redirect
            dst.write_text(
                "<html><body><p>Flowchart source not found. "
                "See <code>pipeline_flowchart.html</code> in the "
                "package root.</p></body></html>",
                encoding="utf-8",
            )
        result["html"] = dst
        logger.info("HTML flowchart → %s", dst)

    html_path = result["html"] or _HTML_SRC
    if not html_path or not html_path.exists():
        logger.warning("No HTML source available — skipping PNG/PDF")
        return result

    # ── PNG via Playwright ────────────────────────────────────────────
    if png:
        png_path = output_dir / "pipeline_flowchart.png"
        ok = _render_playwright(html_path, png_path, fmt="png")
        if not ok:
            ok = _render_wkhtmltoimage(html_path, png_path)
        if ok:
            result["png"] = png_path
            logger.info("PNG flowchart → %s", png_path)
        else:
            logger.warning(
                "PNG generation unavailable — install playwright "
                "(`pip install playwright && playwright install chromium`) "
                "or wkhtmltopdf to enable PNG export."
            )

    # ── PDF via Playwright or wkhtmltopdf ─────────────────────────────
    if pdf:
        pdf_path = output_dir / "pipeline_flowchart.pdf"
        ok = _render_playwright(html_path, pdf_path, fmt="pdf")
        if not ok:
            ok = _render_wkhtmltopdf(html_path, pdf_path)
        if ok:
            result["pdf"] = pdf_path
            logger.info("PDF flowchart → %s", pdf_path)
        else:
            logger.warning(
                "PDF generation unavailable — install playwright "
                "(`pip install playwright && playwright install chromium`) "
                "or wkhtmltopdf to enable PDF export."
            )

    return result


# ── Renderer back-ends ────────────────────────────────────────────────

def _render_playwright(
    html_path: Path, out_path: Path, *, fmt: str = "png",
) -> bool:
    """Render HTML → PNG or PDF using Playwright (headless Chromium)."""
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-untyped]
    except ImportError:
        return False

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1600, "height": 900})
            page.goto(html_path.as_uri())
            page.wait_for_load_state("networkidle")

            if fmt == "pdf":
                page.pdf(
                    path=str(out_path),
                    format="A2",
                    landscape=False,
                    print_background=True,
                    margin={"top": "12mm", "bottom": "12mm",
                            "left": "12mm", "right": "12mm"},
                )
            else:
                page.screenshot(
                    path=str(out_path),
                    full_page=True,
                    type="png",
                )
            browser.close()
        return True
    except Exception as exc:
        logger.debug("Playwright rendering failed: %s", exc)
        return False


def _render_wkhtmltoimage(html_path: Path, out_path: Path) -> bool:
    """Render HTML → PNG via wkhtmltoimage (if installed)."""
    exe = shutil.which("wkhtmltoimage")
    if not exe:
        return False
    try:
        subprocess.run(
            [exe, "--width", "1600", "--quality", "95",
             str(html_path), str(out_path)],
            check=True, capture_output=True, timeout=60,
        )
        return True
    except Exception:
        return False


def _render_wkhtmltopdf(html_path: Path, out_path: Path) -> bool:
    """Render HTML → PDF via wkhtmltopdf (if installed)."""
    exe = shutil.which("wkhtmltopdf")
    if not exe:
        return False
    try:
        subprocess.run(
            [exe, "--page-size", "A2", "--orientation", "Portrait",
             "--enable-local-file-access",
             str(html_path), str(out_path)],
            check=True, capture_output=True, timeout=60,
        )
        return True
    except Exception:
        return False
