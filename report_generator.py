"""
Incident Report Generation Engine for GridShield AI Smart Grid Cybersecurity Simulator.
Generates comprehensive HTML & Markdown reports of telemetry status, attack simulations,
model evaluation benchmarks, and prescriptive mitigation actions based on actual application data.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import html
import pandas as pd


def generate_incident_report_html(
    project_name: str,
    telemetry_summary: Dict[str, Any],
    attack_history: List[Dict[str, Any]],
    model_comparison_df: Optional[pd.DataFrame],
    mitigation_actions: List[str],
    active_threat_level: str = "ELEVATED",
    timestamp: Optional[str] = None,
) -> str:
    """
    Generates a standalone, styled, professional HTML report ready for download/printing/PDF export.
    Uses actual application telemetry, attack simulation records, and ML metrics.
    """
    report_time = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Format telemetry metrics
    total_samples = telemetry_summary.get("total_samples", 0)
    suspicious_readings = telemetry_summary.get("suspicious_readings", 0)
    suspicious_pct = telemetry_summary.get("suspicious_pct", 0.0)
    mean_voltage = telemetry_summary.get("mean_voltage", 230.0)
    mean_frequency = telemetry_summary.get("mean_frequency", 50.0)
    mean_current = telemetry_summary.get("mean_current", 0.0)
    mean_load = telemetry_summary.get("mean_load", 0.0)
    
    # Attack simulation rows
    attack_rows_html = ""
    if attack_history:
        for idx, atk in enumerate(attack_history, start=1):
            atk_type = html.escape(str(atk.get("type", "Unknown")))
            sensor = html.escape(str(atk.get("target_sensor", atk.get("sensor", "N/A"))))
            mag = html.escape(str(atk.get("magnitude", "Default")))
            dur = html.escape(str(atk.get("duration", "N/A")))
            indices_count = len(atk.get("indices", []))
            attack_rows_html += f"""
            <tr>
                <td>#{idx}</td>
                <td><span class="badge-attack">{atk_type}</span></td>
                <td>{sensor}</td>
                <td>{mag}</td>
                <td>{dur} hrs ({indices_count} steps)</td>
                <td>Simulated Injected Event</td>
            </tr>
            """
    else:
        attack_rows_html = """
        <tr>
            <td colspan="6" style="text-align: center; color: #8892b0; padding: 16px;">
                No manual attack scenarios injected during this session. Baseline telemetry only.
            </td>
        </tr>
        """

    # Model Benchmark Table
    model_rows_html = ""
    if model_comparison_df is not None and not model_comparison_df.empty:
        for _, row in model_comparison_df.iterrows():
            m_name = html.escape(str(row.get("Model", "Unknown")))
            acc = f"{row.get('Accuracy', 0.0):.1%}"
            prec = f"{row.get('Precision', 0.0):.3f}"
            rec = f"{row.get('Recall', 0.0):.3f}"
            f1 = f"{row.get('F1-score', 0.0):.3f}"
            roc = f"{row.get('ROC-AUC', 0.0):.3f}"
            fpr = f"{row.get('False Positive Rate', 0.0):.1%}"
            tp = row.get('TP', 0)
            fp = row.get('FP', 0)
            tn = row.get('TN', 0)
            fn = row.get('FN', 0)
            
            model_rows_html += f"""
            <tr>
                <td style="font-weight: 600; color: #00d2ff;">{m_name}</td>
                <td>{acc}</td>
                <td>{prec}</td>
                <td>{rec}</td>
                <td><strong>{f1}</strong></td>
                <td>{roc}</td>
                <td>{fpr}</td>
                <td>TP:{tp} | FP:{fp} | TN:{tn} | FN:{fn}</td>
            </tr>
            """
    else:
        model_rows_html = """
        <tr>
            <td colspan="8" style="text-align: center; color: #8892b0; padding: 16px;">
                Model training benchmarks have not been executed for this session yet.
            </td>
        </tr>
        """

    # Mitigation recommendations
    mitigation_html = ""
    if mitigation_actions:
        for action in mitigation_actions:
            mitigation_html += f"<li style='margin-bottom: 8px;'>{html.escape(action)}</li>"
    else:
        mitigation_html = "<li>System telemetry normal. Maintain routine SCADA surveillance and standard PMU cryptographic heartbeat.</li>"

    # Threat status badge color
    status_bg = "#ff0055" if active_threat_level in ["CRITICAL", "HIGH", "ALERT"] else "#00e676" if active_threat_level == "NORMAL" else "#ffb300"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{html.escape(project_name)} - Cyber Incident & Audit Report</title>
<style>
    @media print {{
        body {{ background: #ffffff !important; color: #111111 !important; }}
        .header, .card, table {{ border-color: #cccccc !important; }}
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background-color: #0b111e;
        color: #e2e8f0;
        margin: 0;
        padding: 40px;
        line-height: 1.6;
    }}
    .container {{
        max-width: 1000px;
        margin: 0 auto;
        background: #111a2e;
        border: 1px solid #1f2d4d;
        border-radius: 12px;
        padding: 40px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }}
    .header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2px solid #1f2d4d;
        padding-bottom: 24px;
        margin-bottom: 30px;
    }}
    .title-area h1 {{
        margin: 0 0 6px 0;
        font-size: 28px;
        color: #00e5ff;
        letter-spacing: 0.5px;
    }}
    .title-area p {{
        margin: 0;
        color: #94a3b8;
        font-size: 14px;
    }}
    .threat-tag {{
        background: {status_bg};
        color: #ffffff;
        font-weight: 700;
        padding: 8px 18px;
        border-radius: 20px;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .grid-metrics {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 30px;
    }}
    .metric-card {{
        background: #17233d;
        border: 1px solid #23365d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }}
    .metric-val {{
        font-size: 22px;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }}
    .metric-lbl {{
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }}
    h2 {{
        font-size: 18px;
        color: #f1f5f9;
        margin-top: 32px;
        margin-bottom: 14px;
        border-left: 4px solid #00e5ff;
        padding-left: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 24px;
        font-size: 13px;
    }}
    th, td {{
        padding: 10px 14px;
        text-align: left;
        border-bottom: 1px solid #1f2d4d;
    }}
    th {{
        background: #17233d;
        color: #94a3b8;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 0.8px;
    }}
    tr:hover td {{
        background: rgba(56, 189, 248, 0.04);
    }}
    .badge-attack {{
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 12px;
    }}
    ul.advisory {{
        background: #17233d;
        border: 1px solid #23365d;
        border-radius: 8px;
        padding: 20px 20px 20px 36px;
        color: #cbd5e1;
        font-size: 14px;
    }}
    .footer {{
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #1f2d4d;
        text-align: center;
        font-size: 12px;
        color: #64748b;
    }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <div class="title-area">
            <h1>⚡ {html.escape(project_name)}</h1>
            <p>Smart Grid Cyber-Physical Anomaly Audit & Incident Advisory Report</p>
            <p>Generated: <strong>{report_time}</strong></p>
        </div>
        <div>
            <span class="threat-tag">Status: {active_threat_level}</span>
        </div>
    </div>

    <h2>1. Executive Telemetry Overview</h2>
    <div class="grid-metrics">
        <div class="metric-card">
            <div class="metric-lbl">Total Time Intervals</div>
            <div class="metric-val">{total_samples:,} hrs</div>
        </div>
        <div class="metric-card">
            <div class="metric-lbl">Compromised Intervals</div>
            <div class="metric-val" style="color: {'#f87171' if suspicious_readings > 0 else '#34d399'};">{suspicious_readings} ({suspicious_pct:.1f}%)</div>
        </div>
        <div class="metric-card">
            <div class="metric-lbl">Mean System Voltage</div>
            <div class="metric-val">{mean_voltage:.2f} V</div>
        </div>
        <div class="metric-card">
            <div class="metric-lbl">Mean Frequency</div>
            <div class="metric-val">{mean_frequency:.3f} Hz</div>
        </div>
    </div>

    <h2>2. Cyberattack Injection & Simulation Log</h2>
    <table>
        <thead>
            <tr>
                <th>Event</th>
                <th>Attack Vector</th>
                <th>Target Sensor</th>
                <th>Magnitude / Shape</th>
                <th>Duration / Scope</th>
                <th>Nature</th>
            </tr>
        </thead>
        <tbody>
            {attack_rows_html}
        </tbody>
    </table>

    <h2>3. Machine Learning Detection Benchmarks (Chronological Test Holdout)</h2>
    <table>
        <thead>
            <tr>
                <th>Model Architecture</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>ROC-AUC</th>
                <th>FPR</th>
                <th>Confusion Matrix</th>
            </tr>
        </thead>
        <tbody>
            {model_rows_html}
        </tbody>
    </table>

    <h2>4. Prescriptive Incident Mitigation Advisory</h2>
    <ul class="advisory">
        {mitigation_html}
    </ul>

    <div class="footer">
        GridShield AI Simulator • Automated Security Audit Document • Confirmatory Telemetry via Zero-Trust Sensor Verification
    </div>
</div>
</body>
</html>
"""
    return html_content
