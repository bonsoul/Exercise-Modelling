"""Plotly chart builders styled to match the 'field log' theme."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Theme palette (mirrors the CSS custom properties in the original design)
INK = "#16241D"
INK_SOFT = "#4B5A50"
LINE = "#C7D0C3"
PAPER = "#F4F7F1"
PRIMARY = "#2F6F4E"
AMBER = "#E0A526"
CORAL = "#C1443B"
PALETTE = ["#2F6F4E", "#E0A526", "#4B5A50", "#7FA88C", "#C1443B", "#9CAF88"]

FONT_BODY = "Inter, sans-serif"
FONT_MONO = "JetBrains Mono, monospace"

_BASE_LAYOUT = dict(
    paper_bgcolor=PAPER,
    plot_bgcolor=PAPER,
    font=dict(family=FONT_BODY, color=INK, size=12),
    margin=dict(l=10, r=20, t=10, b=10),
)


def horizontal_bar(df: pd.DataFrame, color: str = PRIMARY, max_bars: int = 10,
                    value_col: str = "value", name_col: str = "name") -> go.Figure:
    """Horizontal bar chart, largest value at top (matches recharts layout='vertical')."""
    trimmed = df.head(max_bars).iloc[::-1]  # reverse so biggest sits at top
    fig = go.Figure(
        go.Bar(
            x=trimmed[value_col],
            y=trimmed[name_col],
            orientation="h",
            marker_color=color,
            text=trimmed[value_col],
            textposition="outside",
            textfont=dict(family=FONT_MONO, size=10, color=INK_SOFT),
        )
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        height=max(220, len(trimmed) * 32),
        xaxis=dict(showgrid=True, gridcolor=LINE, zeroline=False, tickfont=dict(size=11, color=INK_SOFT)),
        yaxis=dict(showgrid=False, tickfont=dict(size=11, color=INK, family=FONT_MONO)),
        bargap=0.35,
    )
    return fig


def donut(df: pd.DataFrame, value_col: str = "value", name_col: str = "name") -> go.Figure:
    """Donut / pie chart matching the category-share visualization."""
    fig = go.Figure(
        go.Pie(
            labels=df[name_col],
            values=df[value_col],
            hole=0.55,
            marker=dict(colors=PALETTE * 3, line=dict(color=PAPER, width=2)),
            textinfo="none",
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        height=320,
        showlegend=True,
        legend=dict(font=dict(family=FONT_MONO, size=11, color=INK_SOFT), orientation="v"),
    )
    return fig


def intensity_colorscale(base_hex: str):
    """Build a light-to-base_hex colorscale for heatmaps (mimics intensityColor())."""
    r, g, b = int(base_hex[1:3], 16), int(base_hex[3:5], 16), int(base_hex[5:7], 16)
    return [
        [0.0, "rgba(255,255,255,0)"],
        [0.05, f"rgba({r},{g},{b},0.12)"],
        [1.0, f"rgba({r},{g},{b},0.9)"],
    ]


def heatmap(pivot: pd.DataFrame, base_color: str = PRIMARY, height: int = 420,
            text_size: int = 10) -> go.Figure:
    """Generic annotated heatmap, columns on top, rows down the left (mimics the CSS-grid heatmap)."""
    z = pivot.values
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale=intensity_colorscale(base_color),
            showscale=False,
            xgap=3,
            ygap=3,
            hovertemplate="%{y} \u00d7 %{x}: %{z}<extra></extra>",
        )
    )
    annotations = []
    for i, row in enumerate(pivot.index):
        for j, col in enumerate(pivot.columns):
            v = pivot.iloc[i, j]
            if v and v > 0:
                annotations.append(
                    dict(
                        x=col, y=row, text=str(int(v)), showarrow=False,
                        font=dict(family=FONT_MONO, size=text_size, color=INK),
                    )
                )
    fig.update_layout(
        **_BASE_LAYOUT,
        height=height,
        annotations=annotations,
        xaxis=dict(side="top", tickfont=dict(size=10, color=INK_SOFT, family=FONT_MONO), tickangle=-45),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10, color=INK, family=FONT_MONO)),
    )
    return fig


def confusion_heatmap(pivot: pd.DataFrame, height: int = 520) -> go.Figure:
    """Confusion matrix heatmap: green diagonal, coral off-diagonal."""
    classes = list(pivot.index)
    z = pivot.values
    max_v = max(z.max(), 1)

    def cell_color(v, is_diag):
        base = PRIMARY if is_diag else CORAL
        r, g, b = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
        if v == 0:
            return "rgba(0,0,0,0)"
        alpha = 0.12 + min(1.0, v / max_v) * 0.78
        return f"rgba({r},{g},{b},{alpha:.2f})"

    colors = [
        [cell_color(z[i, j], classes[i] == classes[j]) for j in range(len(classes))]
        for i in range(len(classes))
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=[[1 if classes[i] == classes[j] else 0 for j in range(len(classes))] for i in range(len(classes))],
            x=classes, y=classes,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
            showscale=False,
            xgap=2, ygap=2,
        )
    )
    # overlay colored rectangles via shapes for precise diagonal/off-diagonal coloring
    shapes = []
    annotations = []
    n = len(classes)
    for i in range(n):
        for j in range(n):
            v = z[i, j]
            shapes.append(dict(
                type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                fillcolor=colors[i][j], line=dict(color=LINE, width=1),
            ))
            if v > 0:
                annotations.append(dict(
                    x=j, y=i, text=str(int(v)), showarrow=False,
                    font=dict(family=FONT_MONO, size=8, color=INK),
                ))
    fig.update_layout(
        **_BASE_LAYOUT,
        height=height,
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(side="top", tickvals=list(range(n)), ticktext=[c[:4] for c in classes],
                    tickfont=dict(size=9, color=INK_SOFT, family=FONT_MONO), tickangle=-60,
                    range=[-0.5, n - 0.5]),
        yaxis=dict(tickvals=list(range(n)), ticktext=classes, autorange="reversed",
                    tickfont=dict(size=9, color=INK, family=FONT_MONO),
                    range=[n - 0.5, -0.5]),
    )
    return fig


def f1_bar(df: pd.DataFrame) -> go.Figure:
    """Per-class F1 horizontal bar, colored by performance tier."""
    trimmed = df.iloc[::-1]

    def tier_color(f1):
        if f1 >= 0.75:
            return PRIMARY
        if f1 >= 0.5:
            return AMBER
        return CORAL

    colors = [tier_color(v) for v in trimmed["f1"]]
    fig = go.Figure(
        go.Bar(
            x=trimmed["f1"], y=trimmed["name"], orientation="h",
            marker_color=colors,
            text=trimmed["f1"].round(3),
            textposition="outside",
            textfont=dict(family=FONT_MONO, size=10, color=INK_SOFT),
            customdata=trimmed["support"],
            hovertemplate="%{y}: F1 %{x} \u00b7 support %{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        height=max(240, len(trimmed) * 26),
        xaxis=dict(range=[0, 1.08], showgrid=True, gridcolor=LINE, zeroline=False,
                    tickfont=dict(size=11, color=INK_SOFT)),
        yaxis=dict(showgrid=False, tickfont=dict(size=11, color=INK, family=FONT_MONO)),
        bargap=0.4,
    )
    return fig
