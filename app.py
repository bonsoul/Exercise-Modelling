"""
Exercise Data Log — Streamlit Dashboard
A field-survey style analytics dashboard for an exercise dataset:
category/equipment/target-muscle distributions, model performance notes,
and a searchable/filterable exercise explorer.
"""
import streamlit as st
import pandas as pd

from src.data_loader import (
    load_summary, load_rows, load_model_results, dict_list_to_df, heatmap_df, confusion_df,
)
from src.charts import horizontal_bar, donut, heatmap, confusion_heatmap, f1_bar, PRIMARY, AMBER, INK_SOFT

st.set_page_config(
    page_title="Exercise Data Log",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------
# THEME / CSS — "field log" aesthetic mirrored from the original design
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg: #E4EAE3;
        --paper: #F4F7F1;
        --ink: #16241D;
        --ink-soft: #4B5A50;
        --line: #C7D0C3;
        --primary: #2F6F4E;
        --amber: #E0A526;
        --coral: #C1443B;
    }

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

    .stApp {
        background-color: var(--bg);
        background-image:
            linear-gradient(var(--line) 1px, transparent 1px),
            linear-gradient(90deg, var(--line) 1px, transparent 1px);
        background-size: 28px 28px;
    }

    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

    .masthead {
        border-bottom: 3px solid var(--ink);
        padding-bottom: 14px;
        margin-bottom: 20px;
    }
    .eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--ink-soft);
    }
    .masthead-title {
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        font-size: 42px;
        letter-spacing: 0.01em;
        text-transform: uppercase;
        line-height: 1.05;
        color: var(--ink);
        margin: 4px 0 6px;
    }
    .masthead-sub { font-size: 14px; color: var(--ink-soft); max-width: 680px; }

    .log-card {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 18px 20px;
        margin-bottom: 4px;
    }

    .stat-number {
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        font-size: 32px;
        line-height: 1;
        color: var(--ink);
    }
    .stat-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10.5px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--ink-soft);
    }

    .fig-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        letter-spacing: 0.08em;
        color: var(--primary);
        background: rgba(47,111,78,0.08);
        padding: 2px 7px;
        border-radius: 3px;
        margin-right: 8px;
    }
    .chart-title {
        font-family: 'Oswald', sans-serif;
        font-weight: 500;
        font-size: 15px;
        text-transform: uppercase;
        letter-spacing: 0.02em;
        color: var(--ink);
        display: inline;
    }
    .chart-note { font-size: 12px; color: var(--ink-soft); margin: 4px 0 10px; }

    .section-heading {
        display: flex;
        align-items: baseline;
        gap: 10px;
        margin: 34px 0 14px;
    }
    .section-heading .eyebrow { font-size: 11.5px; white-space: nowrap; }
    .section-heading .rule { flex: 1; height: 1px; background: var(--line); }

    .model-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        letter-spacing: 0.08em;
        color: var(--primary);
        background: rgba(47,111,78,0.08);
        padding: 2px 7px;
        border-radius: 3px;
    }
    .model-title {
        font-family: 'Oswald', sans-serif;
        font-weight: 500;
        font-size: 15px;
        text-transform: uppercase;
        margin: 6px 0 4px;
        color: var(--ink);
    }
    .model-note { font-size: 12.5px; color: var(--ink-soft); margin-bottom: 12px; min-height: 54px; }
    .meter-track { height: 6px; background: var(--line); border-radius: 3px; overflow: hidden; }
    .meter-fill { height: 100%; border-radius: 3px; }
    .meter-row { display: flex; justify-content: space-between; font-size: 11.5px; color: var(--ink-soft); margin-bottom: 4px; }
    .mono-number { font-family: 'JetBrains Mono', monospace; font-size: 12.5px; color: var(--ink); }

    .tag-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10.5px;
        background: rgba(224,165,38,0.18);
        color: #8a6410;
        padding: 1px 8px;
        border-radius: 999px;
    }

    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------
summary = load_summary()
rows_df = load_rows()
model_results = load_model_results()

category_df = dict_list_to_df(summary["category"])
equipment_df = dict_list_to_df(summary["equipment"])
target_df = dict_list_to_df(summary["target"])
secondary_df = dict_list_to_df(summary["secondary_muscles"])


def section_heading(label: str):
    st.markdown(
        f'<div class="section-heading"><span class="eyebrow">{label}</span><div class="rule"></div></div>',
        unsafe_allow_html=True,
    )


def fig_title(num: str, title: str):
    st.markdown(f'<span class="fig-tag">FIG. {num}</span><span class="chart-title">{title}</span>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# MASTHEAD
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="masthead">
        <span class="eyebrow">Training Log — Dataset Analytics</span>
        <div class="masthead-title">Exercise Data Log</div>
        <div class="masthead-sub">
            A field survey of {summary['total']:,} logged exercises — category, equipment,
            and target-muscle distribution, plus performance notes from three modelling experiments run against
            the instruction text.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# STAT STRIP
# --------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
stats = [
    ("Exercises Logged", f"{summary['total']:,}", c1),
    ("Categories", str(len(summary["category"])), c2),
    ("Equipment Types", str(len(summary["equipment"])), c3),
    ("Target Muscles", str(len(summary["target"])), c4),
]
for label, value, col in stats:
    with col:
        st.markdown(
            f'<div class="log-card"><div class="stat-label">{label}</div>'
            f'<div class="stat-number">{value}</div></div>',
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------
# DISTRIBUTIONS
# --------------------------------------------------------------------------
section_heading("Distribution")

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="log-card">', unsafe_allow_html=True)
    fig_title("1", "By Category")
    st.plotly_chart(horizontal_bar(category_df, color=PRIMARY), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    st.markdown('<div class="log-card">', unsafe_allow_html=True)
    fig_title("2", "By Equipment (Top 10)")
    st.plotly_chart(horizontal_bar(equipment_df, color=AMBER), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    st.markdown('<div class="log-card">', unsafe_allow_html=True)
    fig_title("3", "By Target Muscle (Top 10)")
    st.plotly_chart(horizontal_bar(target_df, color=PRIMARY), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)
with col4:
    st.markdown('<div class="log-card">', unsafe_allow_html=True)
    fig_title("4", "Secondary Muscles Engaged (Top 10)")
    st.plotly_chart(horizontal_bar(secondary_df, color=AMBER), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    st.markdown('<div class="log-card">', unsafe_allow_html=True)
    fig_title("5", "Category Share")
    st.plotly_chart(donut(category_df), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)
with col6:
    st.markdown('<div class="log-card">', unsafe_allow_html=True)
    fig_title("6", "Equipment × Category (Top 10 Equipment)")
    st.plotly_chart(heatmap(heatmap_df(model_results)), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------------------------------
section_heading("Model Performance")

color_map = {"primary": "var(--primary)", "amber": "var(--amber)", "ink_soft": "var(--ink-soft)"}
m1, m2, m3 = st.columns(3)
for col, model in zip((m1, m2, m3), model_results["models"]):
    with col:
        st.markdown(
            f"""
            <div class="log-card" style="min-height: 260px; display:flex; flex-direction:column; justify-content:space-between;">
                <div>
                    <span class="model-tag">{model['tag']}</span>
                    <div class="model-title">{model['title']}</div>
                    <div class="model-note">{model['note']}</div>
                </div>
                <div>
                    <div class="meter-row"><span>{model['metric_label']}</span><span class="mono-number">{model['metric_value']}</span></div>
                    <div class="meter-track"><div class="meter-fill" style="width:{model['bar_pct']}%; background:{color_map.get(model['color'], 'var(--primary)')};"></div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

col7, col8 = st.columns(2)
with col7:
    st.markdown('<div class="log-card">', unsafe_allow_html=True)
    fig_title("7", "Classifier F1 by Target Muscle")
    f1_df = dict_list_to_df(model_results["per_class_f1"])
    st.plotly_chart(f1_bar(f1_df), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)
with col8:
    st.markdown('<div class="log-card">', unsafe_allow_html=True)
    fig_title("8", "Classifier Confusion Matrix")
    st.markdown(
        '<div class="chart-note">Rows = true target muscle, columns = predicted. '
        'Diagonal is correct; off-diagonal cells show where the classifier confuses muscle groups.</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(confusion_heatmap(confusion_df(model_results)), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# EXERCISE EXPLORER
# --------------------------------------------------------------------------
section_heading("Exercise Explorer")

st.markdown('<div class="log-card">', unsafe_allow_html=True)
f1, f2, f3 = st.columns([2, 1, 1])
with f1:
    search = st.text_input("Search", placeholder="Search by exercise name or target muscle…", label_visibility="collapsed")
with f2:
    category_options = ["All categories"] + sorted(rows_df["category"].unique().tolist())
    category_filter = st.selectbox("Category", category_options, label_visibility="collapsed")
with f3:
    equipment_options = ["All equipment"] + sorted(rows_df["equipment"].unique().tolist())
    equipment_filter = st.selectbox("Equipment", equipment_options, label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

filtered = rows_df.copy()
if category_filter != "All categories":
    filtered = filtered[filtered["category"] == category_filter]
if equipment_filter != "All equipment":
    filtered = filtered[filtered["equipment"] == equipment_filter]
if search.strip():
    q = search.strip().lower()
    filtered = filtered[
        filtered["name"].str.lower().str.contains(q) | filtered["target"].str.lower().str.contains(q)
    ]

PAGE_SIZE = 15
total_rows = len(filtered)
page_count = max(1, -(-total_rows // PAGE_SIZE))

if "page" not in st.session_state:
    st.session_state.page = 0
# reset page if filters shrink the result set below current page
if st.session_state.page >= page_count:
    st.session_state.page = 0

start = st.session_state.page * PAGE_SIZE
end = start + PAGE_SIZE
page_rows = filtered.iloc[start:end].copy()

st.markdown('<div class="log-card">', unsafe_allow_html=True)
if page_rows.empty:
    st.markdown(
        '<div style="text-align:center; padding: 24px 0; color: var(--ink-soft);">No exercises match those filters.</div>',
        unsafe_allow_html=True,
    )
else:
    display_df = page_rows.rename(columns={
        "name": "Name", "category": "Category", "equipment": "Equipment",
        "target": "Target", "muscle_group": "Muscle Group",
    })
    for col in ["Name", "Category", "Equipment", "Muscle Group"]:
        display_df[col] = display_df[col].str.capitalize()
    st.dataframe(
        display_df[["Name", "Category", "Equipment", "Target", "Muscle Group"]],
        use_container_width=True,
        hide_index=True,
    )
st.markdown("</div>", unsafe_allow_html=True)

foot1, foot2 = st.columns([3, 1])
with foot1:
    shown_start = 0 if total_rows == 0 else start + 1
    shown_end = min(end, total_rows)
    st.markdown(
        f'<span class="mono-number" style="color:var(--ink-soft);">'
        f'Showing {shown_start}–{shown_end} of {total_rows}</span>',
        unsafe_allow_html=True,
    )
with foot2:
    p1, p2 = st.columns(2)
    with p1:
        if st.button("← Prev", disabled=st.session_state.page == 0, use_container_width=True):
            st.session_state.page -= 1
            st.rerun()
    with p2:
        if st.button("Next →", disabled=st.session_state.page >= page_count - 1, use_container_width=True):
            st.session_state.page += 1
            st.rerun()
