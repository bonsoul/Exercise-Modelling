"""Streamlit conversion of the exercise dashboard JSX app."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "exercise_dashboard_data.json"

BG = "#E4EAE3"
PAPER = "#F4F7F1"
INK = "#16241D"
INK_SOFT = "#4B5A50"
LINE = "#C7D0C3"
PRIMARY = "#2F6F4E"
AMBER = "#E0A526"
CORAL = "#C1443B"
PALETTE = ["#2F6F4E", "#E0A526", "#4B5A50", "#7FA88C", "#C1443B", "#9CAF88"]
PAGE_SIZE = 15


def running_under_streamlit() -> bool:
    """Detect whether the file is running inside Streamlit runtime."""

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False

    return get_script_run_ctx(suppress_warning=True) is not None


def cache_data(*decorator_args, **decorator_kwargs):
    """Use Streamlit cache only when the app is launched via Streamlit."""

    if running_under_streamlit():
        return st.cache_data(*decorator_args, **decorator_kwargs)

    def decorator(func):
        return func

    return decorator


@cache_data(show_spinner=False)
def load_dashboard_payload() -> dict[str, Any]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing dashboard data file: {DATA_FILE}")
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def intensity_color(value: float, max_value: float, base_color: str) -> str:
    if max_value <= 0:
        alpha = 0.12
    else:
        alpha = 0.12 + (value / max_value) * 0.78
    r, g, b = hex_to_rgb(base_color)
    return f"rgba({r}, {g}, {b}, {alpha:.3f})"


def apply_theme() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

        .stApp {{
          background:
            linear-gradient({LINE} 1px, transparent 1px),
            linear-gradient(90deg, {LINE} 1px, transparent 1px),
            {BG};
          background-size: 28px 28px;
          background-position: -1px -1px;
          color: {INK};
          font-family: 'Inter', sans-serif;
        }}
        .block-container {{
          padding-top: 2rem;
          padding-bottom: 2rem;
        }}
        .dashboard-root {{
          color: {INK};
        }}
        .masthead {{
          border-bottom: 3px solid {INK};
          padding-bottom: 16px;
          margin-bottom: 24px;
        }}
        .eyebrow {{
          font-family: 'JetBrains Mono', monospace;
          font-size: 10.5px;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: {INK_SOFT};
        }}
        .masthead-title {{
          font-family: 'Oswald', sans-serif;
          font-weight: 700;
          font-size: clamp(28px, 4vw, 44px);
          letter-spacing: 0.01em;
          text-transform: uppercase;
          line-height: 1.02;
          margin: 4px 0 6px;
        }}
        .masthead-sub {{
          font-size: 13.5px;
          color: {INK_SOFT};
          max-width: 720px;
          margin: 0;
        }}
        .section-heading {{
          display: flex;
          align-items: baseline;
          gap: 10px;
          margin: 40px 0 16px;
        }}
        .section-heading .eyebrow {{
          font-size: 11px;
        }}
        .section-heading::after {{
          content: "";
          flex: 1;
          height: 1px;
          background: {LINE};
        }}
        .fig-heading {{
          display: flex;
          align-items: baseline;
          gap: 8px;
          margin: 0 0 10px;
        }}
        .fig-tag {{
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px;
          letter-spacing: 0.08em;
          color: {PRIMARY};
          background: rgba(47, 111, 78, 0.08);
          padding: 2px 6px;
          border-radius: 3px;
          white-space: nowrap;
        }}
        .chart-title {{
          font-family: 'Oswald', sans-serif;
          font-weight: 500;
          font-size: 15px;
          text-transform: uppercase;
          letter-spacing: 0.02em;
          color: {INK};
          margin: 0;
        }}
        .stat-card,
        .model-card,
        .log-card {{
          background: {PAPER};
          border: 1px solid {LINE};
          border-radius: 6px;
        }}
        .stat-card {{
          padding: 18px 18px 16px;
          height: 100%;
          display: flex;
          flex-direction: column;
          justify-content: space-between;
        }}
        .stat-number {{
          font-family: 'Oswald', sans-serif;
          font-weight: 600;
          font-size: 34px;
          line-height: 1;
          color: {INK};
          margin-top: 6px;
        }}
        .stat-sub {{
          font-size: 12px;
          color: {INK_SOFT};
          margin-top: 4px;
        }}
        .model-card {{
          padding: 18px 18px 16px;
          display: flex;
          flex-direction: column;
          min-height: 188px;
        }}
        .model-note {{
          font-size: 12px;
          color: {INK_SOFT};
          margin: 4px 0 16px;
        }}
        .metric-row {{
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          margin-bottom: 6px;
          gap: 10px;
        }}
        .metric-label {{
          font-family: 'JetBrains Mono', monospace;
          font-size: 10.5px;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          color: {INK_SOFT};
        }}
        .metric-value {{
          font-family: 'JetBrains Mono', monospace;
          font-size: 13px;
          color: {INK};
        }}
        .meter-track {{
          height: 6px;
          background: {LINE};
          border-radius: 3px;
          overflow: hidden;
        }}
        .meter-fill {{
          height: 100%;
          border-radius: 3px;
        }}
        .explorer-panel {{
          background: {PAPER};
          border: 1px solid {LINE};
          border-radius: 6px;
          padding: 14px;
          margin-bottom: 12px;
        }}
        .filter-input, .filter-select {{
          font-family: 'Inter', sans-serif;
          font-size: 13px;
          background: {PAPER};
          border: 1px solid {LINE};
          border-radius: 5px;
          padding: 8px 10px;
          color: {INK};
        }}
        .filter-input:focus, .filter-select:focus {{
          outline: 2px solid {PRIMARY};
          outline-offset: 1px;
        }}
        .table-wrap {{
          background: {PAPER};
          border: 1px solid {LINE};
          border-radius: 6px;
          padding: 8px;
          overflow-x: auto;
        }}
        table.data-table {{
          width: 100%;
          border-collapse: collapse;
          font-size: 12.5px;
        }}
        table.data-table th {{
          font-family: 'JetBrains Mono', monospace;
          font-size: 10.5px;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          color: {INK_SOFT};
          text-align: left;
          padding: 8px 10px;
          border-bottom: 2px solid {INK};
        }}
        table.data-table td {{
          padding: 7px 10px;
          border-bottom: 1px solid {LINE};
          color: {INK};
        }}
        table.data-table tr:hover td {{
          background: rgba(47, 111, 78, 0.05);
        }}
        .tag-pill {{
          font-family: 'JetBrains Mono', monospace;
          font-size: 10.5px;
          background: rgba(224, 165, 38, 0.18);
          color: #8a6410;
          padding: 1px 7px;
          border-radius: 999px;
          display: inline-block;
        }}
        .page-btn {{
          font-family: 'JetBrains Mono', monospace;
          font-size: 12px;
          border: 1px solid {LINE};
          background: {PAPER};
          padding: 5px 12px;
          border-radius: 4px;
          cursor: pointer;
          color: {INK};
        }}
        .page-btn:disabled {{
          opacity: 0.35;
          cursor: not-allowed;
        }}
        .page-btn:not(:disabled):hover {{
          border-color: {PRIMARY};
          color: {PRIMARY};
        }}
        .heatmap-table {{
          border-collapse: separate;
          border-spacing: 3px;
        }}
        .heatmap-table th {{
          font-family: 'JetBrains Mono', monospace;
          font-size: 9.5px;
          letter-spacing: 0.02em;
          color: {INK_SOFT};
          text-transform: uppercase;
          font-weight: 500;
          writing-mode: vertical-rl;
          transform: rotate(180deg);
          padding: 2px;
          max-width: 22px;
          text-align: left;
        }}
        .heatmap-table.confusion th {{
          font-size: 9px;
        }}
        .heatmap-table .row-label {{
          font-family: 'JetBrains Mono', monospace;
          font-size: 10.5px;
          color: {INK};
          white-space: nowrap;
          padding-right: 8px;
          text-align: right;
        }}
        .heat-cell {{
          width: 26px;
          height: 26px;
          border-radius: 3px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-family: 'JetBrains Mono', monospace;
          font-size: 9.5px;
          color: {INK};
          border: 1px solid {LINE};
        }}
        .heatmap-table.confusion .heat-cell {{
          width: 22px;
          height: 22px;
          font-size: 8.5px;
        }}
        .note-box {{
          font-size: 12px;
          color: {INK_SOFT};
          margin: 0 0 10px;
        }}
        @media (max-width: 640px) {{
          .block-container {{
            padding-top: 1.25rem;
            padding-bottom: 1.25rem;
          }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_heading(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="masthead">
          <span class="eyebrow">Training Log - Dataset Analytics</span>
          <h1 class="masthead-title">{html.escape(title)}</h1>
          <p class="masthead-sub">{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(title: str) -> None:
    st.markdown(
        f"""
        <div class="section-heading">
          <span class="eyebrow">{html.escape(title)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(label: str, value: str, sub: str | None = None) -> str:
    sub_html = f'<div class="stat-sub">{html.escape(sub)}</div>' if sub else ""
    return f"""
    <div class="stat-card">
      <div>
        <div class="eyebrow">{html.escape(label)}</div>
        <div class="stat-number">{html.escape(value)}</div>
      </div>
      {sub_html}
    </div>
    """


def render_model_card(
    tag: str,
    title: str,
    note: str,
    metric_label: str,
    metric_value: str,
    bar_pct: int,
    color: str,
) -> str:
    return f"""
    <div class="model-card" style="--accent: {color};">
      <span class="fig-tag">{html.escape(tag)}</span>
      <h3 class="chart-title" style="margin-top: 6px;">{html.escape(title)}</h3>
      <div class="model-note">{html.escape(note)}</div>
      <div style="margin-top: auto;">
        <div class="metric-row">
          <span class="metric-label">{html.escape(metric_label)}</span>
          <span class="metric-value">{html.escape(metric_value)}</span>
        </div>
        <div class="meter-track">
          <div class="meter-fill" style="width: {bar_pct}%; background: {color};"></div>
        </div>
      </div>
    </div>
    """


def bar_figure(data: list[dict[str, Any]], color: str, max_bars: int = 10) -> go.Figure:
    frame = pd.DataFrame(data).head(max_bars).copy()
    frame["value"] = pd.to_numeric(frame["value"])

    fig = go.Figure(
        go.Bar(
            x=frame["value"],
            y=frame["name"],
            orientation="h",
            marker=dict(color=color),
            text=frame["value"],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(220, len(frame) * 28),
        margin=dict(l=0, r=14, t=0, b=0),
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=dict(family="Inter, sans-serif", color=INK),
        showlegend=False,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=LINE,
        zeroline=False,
        tickfont=dict(size=11, color=INK_SOFT),
        title=None,
    )
    fig.update_yaxes(
        autorange="reversed",
        tickfont=dict(size=11, color=INK, family="JetBrains Mono, monospace"),
        title=None,
    )
    return fig


def donut_figure(data: list[dict[str, Any]]) -> go.Figure:
    frame = pd.DataFrame(data).copy()
    frame["value"] = pd.to_numeric(frame["value"])
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(frame))]
    total = int(frame["value"].sum())

    fig = go.Figure(
        go.Pie(
            labels=frame["name"],
            values=frame["value"],
            hole=0.62,
            sort=False,
            direction="clockwise",
            marker=dict(colors=colors, line=dict(color=PAPER, width=2)),
            textinfo="percent",
            hovertemplate="%{label}: %{value}<extra></extra>",
        )
    )
    fig.add_annotation(
        x=0.5,
        y=0.5,
        text=f"<b>{total:,}</b><br><span style='font-size:10px;color:{INK_SOFT};'>total</span>",
        showarrow=False,
        font=dict(size=28, color=INK),
    )
    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=dict(family="Inter, sans-serif", color=INK),
        legend=dict(font=dict(size=11), orientation="v"),
    )
    return fig


def f1_figure(per_class_f1: list[dict[str, Any]]) -> go.Figure:
    frame = pd.DataFrame(per_class_f1).copy()
    frame["f1"] = pd.to_numeric(frame["f1"])
    frame["color"] = frame["f1"].apply(
        lambda v: PRIMARY if v >= 0.75 else AMBER if v >= 0.5 else CORAL
    )

    fig = go.Figure(
        go.Bar(
            x=frame["f1"],
            y=frame["name"],
            orientation="h",
            marker=dict(color=frame["color"]),
            text=frame["f1"].map(lambda v: f"{v:.3f}"),
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(240, len(frame) * 24),
        margin=dict(l=0, r=24, t=0, b=0),
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=dict(family="Inter, sans-serif", color=INK),
        showlegend=False,
    )
    fig.update_xaxes(
        range=[0, 1],
        showgrid=True,
        gridcolor=LINE,
        zeroline=False,
        tickfont=dict(size=11, color=INK_SOFT),
        title=None,
    )
    fig.update_yaxes(
        autorange="reversed",
        tickfont=dict(size=11, color=INK, family="JetBrains Mono, monospace"),
        title=None,
    )
    return fig


def html_table(
    headers: list[str],
    rows_html: list[str],
    classes: str = "",
) -> str:
    header_html = "".join(f"<th>{html.escape(col)}</th>" for col in headers)
    return f"""
    <div class="table-wrap">
      <table class="heatmap-table {classes}">
        <thead><tr><th></th>{header_html}</tr></thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>
    """


def render_heatmap_table(
    fig_num: str,
    title: str,
    equipment_list: list[str],
    category_list: list[str],
    cells: list[dict[str, Any]],
) -> None:
    st.markdown(
        f"""
        <div class="fig-heading">
          <span class="fig-tag">FIG. {html.escape(fig_num)}</span>
          <h3 class="chart-title">{html.escape(title)}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    lookup = {f"{cell['equipment']}|{cell['category']}": int(cell["count"]) for cell in cells}
    max_value = max((int(cell["count"]) for cell in cells), default=0)

    rows_html: list[str] = []
    for equipment in equipment_list:
        cells_html = [f"<td class='row-label'>{html.escape(equipment)}</td>"]
        for category in category_list:
            value = lookup.get(f"{equipment}|{category}", 0)
            bg = intensity_color(value, max_value, PRIMARY)
            content = str(value) if value > 0 else ""
            cells_html.append(
                f"""
                <td title="{html.escape(equipment)} x {html.escape(category)}: {value}">
                  <div class="heat-cell" style="background: {bg};">{content}</div>
                </td>
                """
            )
        rows_html.append(f"<tr>{''.join(cells_html)}</tr>")

    st.markdown(
        html_table(category_list, rows_html),
        unsafe_allow_html=True,
    )


def render_confusion_matrix(
    fig_num: str,
    title: str,
    classes: list[str],
    confusion: list[dict[str, Any]],
) -> None:
    st.markdown(
        f"""
        <div class="fig-heading">
          <span class="fig-tag">FIG. {html.escape(fig_num)}</span>
          <h3 class="chart-title">{html.escape(title)}</h3>
        </div>
        <p class="note-box">
          Rows = true target muscle, columns = predicted. Diagonal cells are correct predictions; off-diagonal cells show the confusions.
        </p>
        """,
        unsafe_allow_html=True,
    )

    lookup = {f"{cell['true']}|{cell['pred']}": int(cell["count"]) for cell in confusion}
    max_value = max((int(cell["count"]) for cell in confusion), default=0)

    rows_html: list[str] = []
    for true_class in classes:
        cells_html = [f"<td class='row-label'>{html.escape(true_class)}</td>"]
        for pred_class in classes:
            value = lookup.get(f"{true_class}|{pred_class}", 0)
            bg = intensity_color(value, max_value, PRIMARY if true_class == pred_class else CORAL)
            content = str(value) if value > 0 else ""
            cells_html.append(
                f"""
                <td title="true: {html.escape(true_class)}, pred: {html.escape(pred_class)} -> {value}">
                  <div class="heat-cell" style="background: {bg};">{content}</div>
                </td>
                """
            )
        rows_html.append(f"<tr>{''.join(cells_html)}</tr>")

    abbreviations = [cls[:4] for cls in classes]
    st.markdown(
        html_table(abbreviations, rows_html, classes="confusion"),
        unsafe_allow_html=True,
    )


def render_explorer(rows: list[dict[str, Any]]) -> None:
    if "explorer_page" not in st.session_state:
        st.session_state.explorer_page = 0

    def reset_page() -> None:
        st.session_state.explorer_page = 0

    st.markdown(
        """
        <div class="explorer-panel">
          <div style="display:flex; flex-direction:column; gap:10px;">
            <div class="fig-heading" style="margin-bottom:0;">
              <span class="fig-tag">EXPLORER</span>
              <h3 class="chart-title">Exercise Explorer</h3>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    search_col, category_col, equipment_col = st.columns([1.4, 0.8, 0.8])
    categories = ["all", *sorted({row["category"] for row in rows})]
    equipment_list = ["all", *sorted({row["equipment"] for row in rows})]

    with search_col:
        search = st.text_input(
            "Search",
            key="explorer_search",
            placeholder="Search by exercise name or target muscle...",
            on_change=reset_page,
        )
    with category_col:
        category = st.selectbox(
            "Category",
            options=categories,
            key="explorer_category",
            on_change=reset_page,
        )
    with equipment_col:
        equipment = st.selectbox(
            "Equipment",
            options=equipment_list,
            key="explorer_equipment",
            on_change=reset_page,
        )

    q = search.strip().lower()
    filtered_rows = []
    for row in rows:
        if category != "all" and row["category"] != category:
            continue
        if equipment != "all" and row["equipment"] != equipment:
            continue
        if q and q not in row["name"].lower() and q not in row["target"].lower():
            continue
        filtered_rows.append(row)

    page_count = max(1, math.ceil(len(filtered_rows) / PAGE_SIZE))
    st.session_state.explorer_page = min(st.session_state.explorer_page, page_count - 1)
    page = st.session_state.explorer_page
    page_rows = filtered_rows[page * PAGE_SIZE : page * PAGE_SIZE + PAGE_SIZE]

    html_rows: list[str] = []
    for row in page_rows:
        html_rows.append(
            f"""
            <tr>
              <td style="text-transform:capitalize;">{html.escape(row['name'])}</td>
              <td style="text-transform:capitalize;">{html.escape(row['category'])}</td>
              <td style="text-transform:capitalize;">{html.escape(row['equipment'])}</td>
              <td><span class="tag-pill">{html.escape(row['target'])}</span></td>
              <td style="text-transform:capitalize; color:{INK_SOFT};">{html.escape(row['muscle_group'])}</td>
            </tr>
            """
        )

    if not page_rows:
        html_rows.append(
            """
            <tr>
              <td colspan="5" style="text-align:center; padding:24px 0; color:#4B5A50;">
                No exercises match those filters.
              </td>
            </tr>
            """
        )

    st.markdown(
        f"""
        <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Category</th>
              <th>Equipment</th>
              <th>Target</th>
              <th>Muscle Group</th>
            </tr>
          </thead>
          <tbody>
            {''.join(html_rows)}
          </tbody>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.2, 0.8])
    start = 0 if len(filtered_rows) == 0 else page * PAGE_SIZE + 1
    end = min((page + 1) * PAGE_SIZE, len(filtered_rows))
    with left:
        st.markdown(
            f"""
            <div class="mono-number" style="color:{INK_SOFT}; margin-top: 10px;">
              Showing {start}-{end} of {len(filtered_rows)}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        prev_col, next_col = st.columns(2)

        def go_prev() -> None:
            st.session_state.explorer_page = max(0, st.session_state.explorer_page - 1)

        def go_next() -> None:
            st.session_state.explorer_page = min(page_count - 1, st.session_state.explorer_page + 1)

        with prev_col:
            st.button("← Prev", use_container_width=True, disabled=page == 0, on_click=go_prev)
        with next_col:
            st.button("Next →", use_container_width=True, disabled=page >= page_count - 1, on_click=go_next)


def main() -> None:
    st.set_page_config(page_title="Exercise Data Log", layout="wide")
    apply_theme()

    payload = load_dashboard_payload()
    data = payload["DATA"]
    extra = payload["EXTRA"]

    render_heading(
        "Exercise Data Log",
        f"A field survey of {data['total']:,} logged exercises - category, equipment, and target-muscle distribution, plus model notes and a searchable exercise explorer.",
    )

    stat_cols = st.columns(4)
    stat_cols[0].markdown(render_stat_card("Exercises Logged", f"{data['total']:,}"), unsafe_allow_html=True)
    stat_cols[1].markdown(render_stat_card("Categories", str(len(data["category"]))), unsafe_allow_html=True)
    stat_cols[2].markdown(render_stat_card("Equipment Types", str(len(data["equipment"]))), unsafe_allow_html=True)
    stat_cols[3].markdown(render_stat_card("Target Muscles", str(len(data["target"]))), unsafe_allow_html=True)

    render_section("Distribution")
    row1 = st.columns(2)
    with row1[0]:
        st.markdown(
            """
            <div class="fig-heading">
              <span class="fig-tag">FIG. 1</span>
              <h3 class="chart-title">By Category</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(bar_figure(data["category"], PRIMARY), use_container_width=True, config={"displayModeBar": False})
    with row1[1]:
        st.markdown(
            """
            <div class="fig-heading">
              <span class="fig-tag">FIG. 2</span>
              <h3 class="chart-title">By Equipment (Top 10)</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(bar_figure(data["equipment"], AMBER), use_container_width=True, config={"displayModeBar": False})

    row2 = st.columns(2)
    with row2[0]:
        st.markdown(
            """
            <div class="fig-heading">
              <span class="fig-tag">FIG. 3</span>
              <h3 class="chart-title">By Target Muscle (Top 10)</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(bar_figure(data["target"], PRIMARY), use_container_width=True, config={"displayModeBar": False})
    with row2[1]:
        st.markdown(
            """
            <div class="fig-heading">
              <span class="fig-tag">FIG. 4</span>
              <h3 class="chart-title">Secondary Muscles Engaged (Top 10)</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(bar_figure(data["secondary_muscles"], AMBER), use_container_width=True, config={"displayModeBar": False})

    row3 = st.columns(2)
    with row3[0]:
        st.markdown(
            """
            <div class="fig-heading">
              <span class="fig-tag">FIG. 5</span>
              <h3 class="chart-title">Category Share</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(donut_figure(data["category"]), use_container_width=True, config={"displayModeBar": False})
    with row3[1]:
        render_heatmap_table(
            fig_num="6",
            title="Equipment x Category (Top 10 Equipment)",
            equipment_list=extra["heatmap"]["equipment_list"],
            category_list=extra["heatmap"]["category_list"],
            cells=extra["heatmap"]["cells"],
        )

    render_section("Model Performance")
    model_cols = st.columns(3)
    model_cols[0].markdown(
        render_model_card(
            tag="EXP. 01",
            title="Target Muscle Classifier",
            note="TF-IDF + Logistic Regression, predicting target muscle from instruction text alone.",
            metric_label="Macro F1",
            metric_value="0.71",
            bar_pct=71,
            color=PRIMARY,
        ),
        unsafe_allow_html=True,
    )
    model_cols[1].markdown(
        render_model_card(
            tag="EXP. 02",
            title="Secondary Muscle Prediction",
            note="Multi-label One-vs-Rest Logistic Regression over text + equipment + target.",
            metric_label="Micro F1 - Jaccard 0.63",
            metric_value="0.71",
            bar_pct=71,
            color=AMBER,
        ),
        unsafe_allow_html=True,
    )
    model_cols[2].markdown(
        render_model_card(
            tag="EXP. 03",
            title="Content-Based Recommender",
            note="TF-IDF cosine similarity, filtered by equipment + target. Near-duplicate instructions return similarity 1.0.",
            metric_label="Evaluation",
            metric_value="qualitative",
            bar_pct=100,
            color=INK_SOFT,
        ),
        unsafe_allow_html=True,
    )

    perf_cols = st.columns(2)
    with perf_cols[0]:
        st.markdown(
            """
            <div class="fig-heading">
              <span class="fig-tag">FIG. 7</span>
              <h3 class="chart-title">Classifier F1 by Target Muscle</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            f1_figure(extra["confusion"]["per_class_f1"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with perf_cols[1]:
        render_confusion_matrix(
            fig_num="8",
            title="Classifier Confusion Matrix",
            classes=extra["confusion"]["classes"],
            confusion=extra["confusion"]["confusion"],
        )

    render_section("Exercise Explorer")
    render_explorer(data["rows"])


if __name__ == "__main__":
    if running_under_streamlit():
        main()
    else:
        print("Run this app with: streamlit run streamlit_app.py")
