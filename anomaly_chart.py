import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Domain-specific metric definitions
METRIC_CONFIG = {
    "treatment": {
        "label":     "Patient Readmission Rate (%)",
        "baseline":  12.0,
        "threshold": 18.0,
        "unit":      "%",
    },
    "payment": {
        "label":     "Claim Denial Rate (%)",
        "baseline":  8.0,
        "threshold": 15.0,
        "unit":      "%",
    },
    "operations": {
        "label":     "ICU Bed Utilization (%)",
        "baseline":  72.0,
        "threshold": 88.0,
        "unit":      "%",
    },
}

# Anomaly injection by risk level
_ANOMALY_PROFILE = {
    "low":    {"multiplier": 1.25, "spike_days": [27, 29]},
    "medium": {"multiplier": 1.55, "spike_days": [20, 25, 28]},
    "high":   {"multiplier": 1.90, "spike_days": [14, 19, 23, 26, 29]},
}


def build_anomaly_chart(domain: str, risk_level: str, color: str) -> go.Figure:
    cfg     = METRIC_CONFIG.get(domain, METRIC_CONFIG["operations"])
    profile = _ANOMALY_PROFILE.get(risk_level, _ANOMALY_PROFILE["low"])

    days  = 30
    dates = [datetime.today() - timedelta(days=days - i) for i in range(days)]

    np.random.seed(7)
    noise  = np.random.normal(0, cfg["baseline"] * 0.04, days)
    values = np.array([cfg["baseline"]] * days, dtype=float) + noise

    # Inject spikes
    for d in profile["spike_days"]:
        if d < days:
            values[d] = cfg["baseline"] * profile["multiplier"] + np.random.normal(0, cfg["baseline"] * 0.02)

    threshold    = cfg["threshold"]
    anomaly_mask = values > threshold

    fig = go.Figure()

    # Main trend line
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode="lines+markers",
        name=cfg["label"],
        line=dict(color=color, width=2.5),
        marker=dict(size=5, color=color),
        hovertemplate="%{x|%b %d}<br>" + cfg["label"] + ": %{y:.1f}" + cfg["unit"] + "<extra></extra>",
    ))

    # Anomaly markers
    if any(anomaly_mask):
        fig.add_trace(go.Scatter(
            x=[d for d, m in zip(dates, anomaly_mask) if m],
            y=[v for v, m in zip(values, anomaly_mask) if m],
            mode="markers",
            name="Anomaly Detected",
            marker=dict(color="#B71C1C", size=12, symbol="x-thin", line=dict(width=2.5, color="#B71C1C")),
            hovertemplate="%{x|%b %d}<br><b>ANOMALY: %{y:.1f}" + cfg["unit"] + "</b><extra></extra>",
        ))

    # Alert threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#E65100",
        line_width=1.5,
        annotation_text=f"Alert Threshold ({threshold}{cfg['unit']})",
        annotation_position="top left",
        annotation_font_color="#E65100",
    )

    fig.update_layout(
        title=dict(
            text=f"30-Day Trend · {cfg['label']}  —  Anomaly Detection",
            font=dict(size=14),
        ),
        xaxis=dict(title="Date", tickformat="%b %d", showgrid=False),
        yaxis=dict(title=cfg["label"], gridcolor="#eeeeee"),
        height=320,
        margin=dict(t=55, b=35, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
        hovermode="x unified",
    )

    return fig
