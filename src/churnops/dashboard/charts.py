"""Reusable Plotly chart builders — pure functions (DataFrame in, Figure out).

No Streamlit import here either: these are unit-testable with plain pandas
DataFrames, and every builder returns a valid (if empty-state) figure rather
than raising when given no data, so a section of the dashboard can never
turn into a stack trace.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, showarrow=False, font={"size": 16, "color": "#888"},
        xref="paper", yref="paper", x=0.5, y=0.5,
    )
    fig.update_layout(
        xaxis={"visible": False}, yaxis={"visible": False},
        plot_bgcolor="rgba(0,0,0,0)", height=280,
    )
    return fig


def churn_rate_bar(
    df: pd.DataFrame,
    x: str = "contract",
    y: str = "churn_rate_pct",
    title: str = "Predicted churn rate by contract type",
) -> go.Figure:
    """Bar chart of churn rate by segment. Expects columns [x, y]."""
    if df.empty or x not in df.columns or y not in df.columns:
        return _empty_figure("No churn-by-segment data yet")

    fig = px.bar(
        df, x=x, y=y, text=y, title=title, color=y, color_continuous_scale="Reds",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        yaxis_title="Churn rate (%)", xaxis_title=x.replace("_", " ").title(),
        coloraxis_showscale=False,
    )
    return fig


def probability_histogram(
    df: pd.DataFrame,
    column: str = "churn_probability",
    title: str = "Churn probability distribution",
) -> go.Figure:
    """Histogram of predicted churn probabilities across the current window."""
    if df.empty or column not in df.columns:
        return _empty_figure("No predictions in this window yet")

    fig = px.histogram(df, x=column, nbins=30, title=title)
    fig.update_layout(xaxis_title="P(churn)", yaxis_title="Count", bargap=0.05)
    return fig


def volume_timeseries(
    df: pd.DataFrame,
    x: str = "bucket",
    y: str = "count",
    title: str = "Prediction volume over time",
) -> go.Figure:
    """Line chart of prediction counts per time bucket — spots pipeline stalls."""
    if df.empty or x not in df.columns or y not in df.columns:
        return _empty_figure("No volume data yet")

    fig = px.line(df, x=x, y=y, markers=True, title=title)
    fig.update_layout(xaxis_title="Time bucket", yaxis_title="Predictions")
    return fig


def drift_heatmap(
    df: pd.DataFrame,
    title: str = "Per-feature data drift (latest cycle)",
) -> go.Figure:
    """Heatmap of per-feature drift statistics.

    Expects columns [feature_name, statistic_name, statistic_value] — rows
    with a null feature_name (overall prediction-drift rows, not per-feature)
    are excluded; this chart is specifically about the feature-level picture.
    """
    required = {"feature_name", "statistic_name", "statistic_value"}
    if df.empty or not required.issubset(df.columns):
        return _empty_figure("No drift data yet")

    feature_rows = df[df["feature_name"].notna()]
    if feature_rows.empty:
        return _empty_figure("No per-feature drift rows in the latest cycle")

    pivot = feature_rows.pivot_table(
        index="feature_name", columns="statistic_name", values="statistic_value", aggfunc="last",
    )
    fig = px.imshow(
        pivot, text_auto=".3f", color_continuous_scale="Reds", aspect="auto", title=title,
    )
    fig.update_layout(xaxis_title="Statistic", yaxis_title="Feature")
    return fig


def risk_gauge(probability: float, mid_threshold: float = 0.40, high_threshold: float = 0.70) -> go.Figure:
    """Gauge for the optional live-scoring widget's single-prediction result."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%"},
        title={"text": "P(churn)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#333"},
            "steps": [
                {"range": [0, mid_threshold * 100], "color": "#2CA02C"},
                {"range": [mid_threshold * 100, high_threshold * 100], "color": "#E8A33D"},
                {"range": [high_threshold * 100, 100], "color": "#D62728"},
            ],
        },
    ))
    fig.update_layout(height=250, margin={"t": 40, "b": 10, "l": 30, "r": 30})
    return fig
