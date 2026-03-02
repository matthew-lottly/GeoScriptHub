"""
status_flowchart.py
===================
Generate a modern interactive HTML pipeline architecture flowchart.

Each pipeline step is rendered as a card node with colour status:
  - 🟢 Green  = step succeeded
  - 🟡 Yellow = ran but no data / partial success
  - 🔴 Red    = step failed
  - ⬜ Grey   = not executed

Reads the pipeline_status.json written by cli.py and produces
a self-contained HTML file with a polished model-flow-diagram layout.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("geoscripthub.quantum_flood_frequency.status_flowchart")

# Pipeline flow — defines the node order, connections, and groups
PIPELINE_NODES = [
    {"id": "aoi", "label": "AOI Builder", "key": "AOI", "icon": "🌐", "group": "input"},
    {"id": "landsat", "label": "Landsat 30 m", "key": "Landsat", "icon": "🛰️", "group": "acquisition"},
    {"id": "sentinel2", "label": "Sentinel-2 10 m", "key": "Sentinel-2", "icon": "🛰️", "group": "acquisition"},
    {"id": "naip", "label": "NAIP 1 m", "key": "NAIP", "icon": "✈️", "group": "acquisition"},
    {"id": "sar_acq", "label": "Sentinel-1 SAR", "key": "Sentinel-1 SAR", "icon": "📡", "group": "acquisition"},
    {"id": "dem_acq", "label": "Copernicus DEM", "key": "Copernicus DEM", "icon": "⛰️", "group": "acquisition"},
    {"id": "preproc", "label": "30 m Aggregate &amp; Align", "key": "Preprocessing", "icon": "🔧", "group": "processing"},
    {"id": "sar_proc", "label": "SAR Processing", "key": "SAR Processing", "icon": "📊", "group": "processing"},
    {"id": "terrain", "label": "Terrain / HAND", "key": "Terrain/DEM", "icon": "🏔️", "group": "processing"},
    {"id": "classify", "label": "QIEC v4.0 Classifier", "key": "Classification", "icon": "⚛️", "group": "model"},
    {"id": "flood", "label": "Flood Frequency", "key": "Flood Frequency", "icon": "🌊", "group": "model"},
    {"id": "gauge", "label": "USGS Gauge Data", "key": "USGS Gauge Data", "icon": "📈", "group": "validation"},
    {"id": "fema", "label": "FEMA Zones", "key": "FEMA Zones", "icon": "🗺️", "group": "validation"},
    {"id": "viz", "label": "Visualisation", "key": "Visualisation", "icon": "🖼️", "group": "output"},
]

# Group definitions for the layout
GROUPS = [
    {"id": "input", "label": "Input", "sublabel": "Study Area"},
    {"id": "acquisition", "label": "Data Acquisition", "sublabel": "Multi-Sensor Imagery"},
    {"id": "processing", "label": "Preprocessing", "sublabel": "Alignment & Feature Extraction"},
    {"id": "model", "label": "Classification & Analysis", "sublabel": "Quantum-Hybrid Ensemble"},
    {"id": "validation", "label": "Validation", "sublabel": "External Reference Data"},
    {"id": "output", "label": "Output", "sublabel": "Maps & Reports"},
]


def generate_flowchart(
    status_json_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Generate an HTML flowchart from pipeline_status.json.

    Args:
        status_json_path: Path to the JSON status file.
        output_path: Path for the HTML output (default: same dir).

    Returns:
        Path to the generated HTML file.
    """
    with open(status_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps = data.get("steps", {})
    elapsed = data.get("elapsed_seconds", 0)
    timestamp = data.get("timestamp", "unknown")

    if output_path is None:
        output_path = status_json_path.parent / "pipeline_flowchart.html"

    # Build grouped node HTML
    colour_map = {
        "green": ("#10B981", "#065F46", "#D1FAE5"),
        "yellow": ("#F59E0B", "#78350F", "#FEF3C7"),
        "red": ("#EF4444", "#7F1D1D", "#FEE2E2"),
        "grey": ("#6B7280", "#374151", "#F3F4F6"),
    }

    def _node_html(node: dict) -> str:
        key = node["key"]
        step_info = steps.get(key, {"status": "grey", "detail": "not executed"})
        status = step_info.get("status", "grey")
        detail = step_info.get("detail", "")
        accent, _, _ = colour_map.get(status, colour_map["grey"])
        status_label = {
            "green": "Passed", "yellow": "Warning", "red": "Failed", "grey": "Pending"
        }.get(status, "Unknown")
        return (
            f'  <div class="node status-{status}" data-status="{status}">\n'
            f'    <div class="node-accent" style="background:{accent};"></div>\n'
            f'    <div class="node-body">\n'
            f'      <div class="node-icon">{node.get("icon", "")}</div>\n'
            f'      <div class="node-content">\n'
            f'        <div class="node-label">{node["label"]}</div>\n'
            f'        <div class="node-detail" title="{detail}">{detail[:80] if detail else status_label}</div>\n'
            f'      </div>\n'
            f'      <div class="node-badge">{status_label}</div>\n'
            f'    </div>\n'
            f'  </div>'
        )

    # Group nodes
    grouped: dict[str, list[str]] = {g["id"]: [] for g in GROUPS}
    for node in PIPELINE_NODES:
        g = node.get("group", "input")
        grouped[g].append(_node_html(node))

    # Compute group statuses
    def _group_status(group_id: str) -> str:
        nodes_in_group = [n for n in PIPELINE_NODES if n.get("group") == group_id]
        statuses = [steps.get(n["key"], {}).get("status", "grey") for n in nodes_in_group]
        if "red" in statuses:
            return "red"
        if all(s == "green" for s in statuses):
            return "green"
        if any(s == "green" for s in statuses):
            return "yellow"
        return "grey"

    # Build section HTML
    sections_html = []
    for g in GROUPS:
        gstatus = _group_status(g["id"])
        accent, _, _ = colour_map.get(gstatus, colour_map["grey"])
        nodes_block = "\n".join(grouped[g["id"]])
        sections_html.append(
            f'<div class="group group-{gstatus}">\n'
            f'  <div class="group-header">\n'
            f'    <div class="group-indicator" style="background:{accent};"></div>\n'
            f'    <div class="group-titles">\n'
            f'      <div class="group-label">{g["label"]}</div>\n'
            f'      <div class="group-sublabel">{g["sublabel"]}</div>\n'
            f'    </div>\n'
            f'  </div>\n'
            f'  <div class="group-nodes">\n'
            f'{nodes_block}\n'
            f'  </div>\n'
            f'</div>'
        )

    # Insert arrows between groups
    body_parts = []
    for i, sec in enumerate(sections_html):
        body_parts.append(sec)
        if i < len(sections_html) - 1:
            body_parts.append('<div class="connector"><svg viewBox="0 0 24 24" width="24" height="24"><path d="M12 4v14M6 14l6 6 6-6" stroke="#475569" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg></div>')

    body_joined = "\n".join(body_parts)

    # Statistics
    total = len(PIPELINE_NODES)
    n_green = sum(1 for n in PIPELINE_NODES if steps.get(n["key"], {}).get("status") == "green")
    n_yellow = sum(1 for n in PIPELINE_NODES if steps.get(n["key"], {}).get("status") == "yellow")
    n_red = sum(1 for n in PIPELINE_NODES if steps.get(n["key"], {}).get("status") == "red")
    n_grey = total - n_green - n_yellow - n_red
    pct_ok = int(100 * n_green / total) if total else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QIEC v4.0 — Pipeline Architecture</title>
<style>
  :root {{
    --bg: #0F172A;
    --surface: #1E293B;
    --surface-hover: #334155;
    --border: #334155;
    --text: #F1F5F9;
    --text-muted: #94A3B8;
    --green: #10B981;
    --green-bg: rgba(16, 185, 129, 0.08);
    --yellow: #F59E0B;
    --yellow-bg: rgba(245, 158, 11, 0.08);
    --red: #EF4444;
    --red-bg: rgba(239, 68, 68, 0.08);
    --grey: #6B7280;
    --grey-bg: rgba(107, 114, 128, 0.06);
    --radius: 12px;
    --radius-sm: 8px;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 2rem;
    min-height: 100vh;
  }}

  /* --- Header --- */
  .header {{
    text-align: center;
    margin-bottom: 2.5rem;
  }}
  .header h1 {{
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60A5FA, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
  }}
  .header .subtitle {{
    color: var(--text-muted);
    font-size: 0.9rem;
  }}

  /* --- Stats bar --- */
  .stats {{
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-bottom: 2rem;
    flex-wrap: wrap;
  }}
  .stat {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.6rem 1.2rem;
  }}
  .stat-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }}
  .stat-value {{
    font-weight: 700;
    font-size: 1.1rem;
  }}
  .stat-label {{
    font-size: 0.8rem;
    color: var(--text-muted);
  }}

  /* --- Progress bar --- */
  .progress-wrap {{
    max-width: 600px;
    margin: 0 auto 2rem;
    background: var(--surface);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
  }}
  .progress-fill {{
    height: 100%;
    border-radius: 999px;
    transition: width 1s ease;
    background: linear-gradient(90deg, var(--green), #34D399);
  }}

  /* --- Pipeline container --- */
  .pipeline {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0;
    max-width: 720px;
    margin: 0 auto;
  }}

  /* --- Connector arrows --- */
  .connector {{
    display: flex;
    align-items: center;
    justify-content: center;
    height: 32px;
    opacity: 0.5;
  }}

  /* --- Group --- */
  .group {{
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    transition: border-color 0.3s;
  }}
  .group:hover {{
    border-color: #475569;
  }}
  .group-green {{ border-left: 3px solid var(--green); }}
  .group-yellow {{ border-left: 3px solid var(--yellow); }}
  .group-red {{ border-left: 3px solid var(--red); }}
  .group-grey {{ border-left: 3px solid var(--grey); }}

  .group-header {{
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.8rem 1.2rem;
    border-bottom: 1px solid var(--border);
  }}
  .group-indicator {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }}
  .group-label {{
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
    text-transform: uppercase;
  }}
  .group-sublabel {{
    font-size: 0.75rem;
    color: var(--text-muted);
  }}

  .group-nodes {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    padding: 0.8rem;
  }}

  /* --- Node --- */
  .node {{
    flex: 1 1 calc(50% - 0.5rem);
    min-width: 200px;
    border-radius: var(--radius-sm);
    overflow: hidden;
    display: flex;
    transition: transform 0.15s, box-shadow 0.15s;
    cursor: default;
  }}
  .node:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  }}
  .node-accent {{
    width: 4px;
    flex-shrink: 0;
  }}
  .node-body {{
    flex: 1;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.6rem 0.8rem;
  }}
  .status-green .node-body {{ background: var(--green-bg); }}
  .status-yellow .node-body {{ background: var(--yellow-bg); }}
  .status-red .node-body {{ background: var(--red-bg); }}
  .status-grey .node-body {{ background: var(--grey-bg); }}

  .node-icon {{
    font-size: 1.3rem;
    flex-shrink: 0;
    width: 1.6rem;
    text-align: center;
  }}
  .node-content {{
    flex: 1;
    min-width: 0;
  }}
  .node-label {{
    font-weight: 600;
    font-size: 0.85rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .node-detail {{
    font-size: 0.7rem;
    color: var(--text-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 240px;
  }}
  .node-badge {{
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 0.2rem 0.5rem;
    border-radius: 999px;
    white-space: nowrap;
    flex-shrink: 0;
  }}
  .status-green .node-badge {{ background: rgba(16,185,129,0.15); color: var(--green); }}
  .status-yellow .node-badge {{ background: rgba(245,158,11,0.15); color: var(--yellow); }}
  .status-red .node-badge {{ background: rgba(239,68,68,0.15); color: var(--red); }}
  .status-grey .node-badge {{ background: rgba(107,114,128,0.15); color: var(--grey); }}

  /* --- Footer --- */
  .footer {{
    text-align: center;
    margin-top: 2rem;
    color: var(--text-muted);
    font-size: 0.8rem;
  }}
  .footer a {{
    color: #60A5FA;
    text-decoration: none;
  }}

  /* --- Animation --- */
  .group, .stat {{ animation: fadeUp 0.5s ease both; }}
  .group:nth-child(1) {{ animation-delay: 0.05s; }}
  .group:nth-child(3) {{ animation-delay: 0.15s; }}
  .group:nth-child(5) {{ animation-delay: 0.25s; }}
  .group:nth-child(7) {{ animation-delay: 0.35s; }}
  .group:nth-child(9) {{ animation-delay: 0.45s; }}
  .group:nth-child(11) {{ animation-delay: 0.55s; }}
  @keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(12px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>QIEC v4.0 — Pipeline Architecture</h1>
  <div class="subtitle">
    Quantum-Inspired Ensemble Classifier &bull; 30 m Aggregate Resolution &bull; Multi-Sensor Fusion
  </div>
</div>

<div class="stats">
  <div class="stat">
    <div class="stat-dot" style="background:var(--green);"></div>
    <div><span class="stat-value">{n_green}</span> <span class="stat-label">passed</span></div>
  </div>
  <div class="stat">
    <div class="stat-dot" style="background:var(--yellow);"></div>
    <div><span class="stat-value">{n_yellow}</span> <span class="stat-label">warnings</span></div>
  </div>
  <div class="stat">
    <div class="stat-dot" style="background:var(--red);"></div>
    <div><span class="stat-value">{n_red}</span> <span class="stat-label">failed</span></div>
  </div>
  <div class="stat">
    <div class="stat-dot" style="background:var(--grey);"></div>
    <div><span class="stat-value">{n_grey}</span> <span class="stat-label">pending</span></div>
  </div>
  <div class="stat">
    <div><span class="stat-value">{elapsed:.0f}s</span> <span class="stat-label">elapsed</span></div>
  </div>
</div>

<div class="progress-wrap">
  <div class="progress-fill" style="width:{pct_ok}%;"></div>
</div>

<div class="pipeline">
{body_joined}
</div>

<div class="footer">
  Generated {timestamp} &bull; GeoScriptHub &bull;
  Quantum Flood Frequency v4.0
</div>

</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("Pipeline flowchart saved → %s", output_path)
    return output_path
