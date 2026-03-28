import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        border: 1px solid #2e2e45;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 { margin: 0; font-size: 1.8rem; font-weight: 600; color: #e0e0f0; }
    .metric-card p  { margin: 0; font-size: 0.8rem; color: #a0a0c0; }
    .metric-card .sub { font-size: 0.75rem; color: #6c6c8a; margin-top: 2px; }
    .section-header {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #a0a0c0;
        margin-bottom: 0.5rem;
    }
    .badge-top   { background:#1a3a2a; color:#6fcf97; padding:2px 10px; border-radius:99px; font-size:0.72rem; font-weight:600; }
    .badge-fast  { background:#1a2a3a; color:#56b4e9; padding:2px 10px; border-radius:99px; font-size:0.72rem; font-weight:600; }
    .badge-solid { background:#2a2a3a; color:#a0a0c0; padding:2px 10px; border-radius:99px; font-size:0.72rem; font-weight:600; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 6px 18px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
BG       = "#1e1e2e"
GRID_CLR = "#2e2e45"
TICK_CLR = "#a0a0c0"
TEXT_CLR = "#e0e0f0"

MODELS = ["LR", "KNN", "SVC", "RFC", "XGB"]

BINARY = {
    "test": {"ACC": [0.883,0.951,0.974,0.974,0.978],
             "AUC": [0.917,0.979,0.992,0.996,0.998],
             "F1":  [0.660,0.884,0.937,0.937,0.945],
             "F2":  [0.602,0.913,0.945,0.947,0.950]},
    "val":  {"ACC": [0.883,0.956,0.966,0.977,0.987],
             "AUC": [0.904,0.975,0.986,0.997,0.999],
             "F1":  [0.673,0.895,0.919,0.943,0.967],
             "F2":  [0.629,0.915,0.937,0.954,0.972]},
    "train_time": ["<1s","1s","49s","1m 54s","23s"],
    "tags": ["Solid","Fast","Solid","Solid","Top"],
}

MULTI = {
    "test": {"ACC": [0.922,0.965,0.973,0.974,0.985],
             "AUC": [0.983,0.957,0.995,0.997,0.999],
             "F1":  [0.904,0.965,0.973,0.975,0.985],
             "F2":  [0.914,0.965,0.973,0.974,0.985]},
    "val":  {"ACC": [0.925,0.956,0.968,0.978,0.982],
             "AUC": [0.983,0.956,0.993,0.998,0.999],
             "F1":  [0.908,0.957,0.969,0.978,0.982],
             "F2":  [0.917,0.957,0.968,0.978,0.982]},
    "train_time": ["<1s","1s","49s","1m 54s","1m 32s"],
    "tags": ["Solid","Fast","Solid","Solid","Top"],
}

FEATURE_IMPORTANCE = {
    "KNN": {"Type":0.03,"Air temp":0.08,"Process temp":0.07,"Rot. speed":0.22,"Torque":0.30,"Tool wear":0.28},
    "SVC": {"Type":0.02,"Air temp":0.06,"Process temp":0.05,"Rot. speed":0.25,"Torque":0.35,"Tool wear":0.27},
    "RFC": {"Type":0.02,"Air temp":0.05,"Process temp":0.04,"Rot. speed":0.23,"Torque":0.38,"Tool wear":0.28},
    "XGB": {"Type":0.01,"Air temp":0.05,"Process temp":0.04,"Rot. speed":0.24,"Torque":0.39,"Tool wear":0.27},
}

FAILURE_DIST = {
    "No Failure":9652,"Heat Dissipation":112,"Power Failure":95,
    "Overstrain":78,"Tool Wear":45,"Random":18,
}

MODEL_COLORS = {
    "LR":"#e74c3c","KNN":"#3498db","SVC":"#9b59b6",
    "RFC":"#27ae60","XGB":"#e67e22",
}

METRIC_LABELS = {"ACC":"Accuracy","AUC":"AUC-ROC","F1":"F1 Score","F2":"F2 Score"}

# ── Helpers ────────────────────────────────────────────────────────────────────
def pct(v): return f"{v*100:.1f}%"

def best_model_idx(data, metric):
    return int(np.argmax(data["test"][metric]))

def hex_to_rgba(hex_color, alpha=0.18):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

def _base_layout(height=320, **extra):
    return dict(
        plot_bgcolor=BG, paper_bgcolor=BG,
        margin=dict(t=35, b=15, l=10, r=10),
        height=height,
        font=dict(color=TEXT_CLR, size=12),
        **extra,
    )

# ── Chart builders ─────────────────────────────────────────────────────────────
def build_bar_chart(data, metric, split="test", highlight=True):
    vals = [v * 100 for v in data[split][metric]]
    best_i = int(np.argmax(vals))
    colors = [
        MODEL_COLORS[m] if (i == best_i and highlight) else "#4a4a6a"
        for i, m in enumerate(MODELS)
    ]
    fig = go.Figure(go.Bar(
        x=MODELS, y=vals,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(color=TEXT_CLR, size=12),
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(),
        yaxis=dict(range=[80,103], ticksuffix="%",
                   title=dict(text=METRIC_LABELS[metric], font=dict(color=TICK_CLR)),
                   tickfont=dict(color=TICK_CLR),
                   gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
        xaxis=dict(title=dict(text="Model", font=dict(color=TICK_CLR)),
                   tickfont=dict(color=TICK_CLR), showgrid=False),
        showlegend=False,
    )
    return fig

def build_grouped_bar(data, metric):
    val_vals  = [v * 100 for v in data["val"][metric]]
    test_vals = [v * 100 for v in data["test"][metric]]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Validation", x=MODELS, y=val_vals,
                         marker_color="#5b8dd9",
                         text=[f"{v:.1f}%" for v in val_vals],
                         textposition="outside",
                         textfont=dict(color=TEXT_CLR, size=11)))
    fig.add_trace(go.Bar(name="Test", x=MODELS, y=test_vals,
                         marker_color="#e07b39",
                         text=[f"{v:.1f}%" for v in test_vals],
                         textposition="outside",
                         textfont=dict(color=TEXT_CLR, size=11)))
    fig.update_layout(
        **_base_layout(),
        barmode="group",
        yaxis=dict(range=[78,105], ticksuffix="%",
                   tickfont=dict(color=TICK_CLR),
                   gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
        xaxis=dict(tickfont=dict(color=TICK_CLR), showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    font=dict(color=TEXT_CLR)),
    )
    return fig

def build_radar(data, metrics=("ACC","AUC","F1","F2")):
    fig = go.Figure()
    for m in MODELS:
        i = MODELS.index(m)
        vals = [data["test"][met][i] * 100 for met in metrics]
        vals_closed = vals + [vals[0]]
        cats = [METRIC_LABELS[met] for met in metrics] + [METRIC_LABELS[metrics[0]]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats, name=m,
            line_color=MODEL_COLORS[m], fill="toself",
            fillcolor=hex_to_rgba(MODEL_COLORS[m], 0.18),
            hovertemplate=f"<b>{m}</b><br>%{{theta}}: %{{r:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        polar=dict(
            bgcolor=BG,
            radialaxis=dict(visible=True, range=[80,100], ticksuffix="%",
                            tickfont=dict(color=TICK_CLR, size=10),
                            gridcolor=GRID_CLR, linecolor=GRID_CLR),
            angularaxis=dict(tickfont=dict(color=TEXT_CLR, size=12),
                             gridcolor=GRID_CLR, linecolor=GRID_CLR),
        ),
        showlegend=True, height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                    font=dict(color=TEXT_CLR)),
        margin=dict(t=20, b=30),
        paper_bgcolor=BG,
        font=dict(color=TEXT_CLR),
    )
    return fig

def build_feature_importance(selected_models):
    fig = go.Figure()
    features = list(FEATURE_IMPORTANCE["KNN"].keys())
    for m in selected_models:
        vals = [FEATURE_IMPORTANCE[m][f] for f in features]
        fig.add_trace(go.Bar(
            name=m, x=features, y=vals,
            marker_color=MODEL_COLORS[m],
            hovertemplate=f"<b>{m}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        **_base_layout(),
        barmode="group",
        yaxis=dict(title=dict(text="Permutation importance", font=dict(color=TICK_CLR)),
                   tickfont=dict(color=TICK_CLR),
                   gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
        xaxis=dict(tickfont=dict(color=TICK_CLR), showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    font=dict(color=TEXT_CLR)),
    )
    return fig

def build_failure_donut():
    labels = list(FAILURE_DIST.keys())
    values = list(FAILURE_DIST.values())
    colors = ["#3266ad","#e74c3c","#e67e22","#9b59b6","#27ae60","#f39c12"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55, marker_colors=colors,
        hovertemplate="%{label}: %{value} samples (%{percent})<extra></extra>",
        textinfo="percent+label", textfont_size=12,
    ))
    fig.update_layout(
        height=300, margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False, paper_bgcolor=BG,
        font=dict(color=TEXT_CLR),
    )
    return fig

def model_summary_table(data):
    rows = []
    for i, m in enumerate(MODELS):
        rows.append({
            "Model": m,
            "Accuracy": pct(data["test"]["ACC"][i]),
            "AUC-ROC":  pct(data["test"]["AUC"][i]),
            "F1":        pct(data["test"]["F1"][i]),
            "F2":        pct(data["test"]["F2"][i]),
            "Train time": data["train_time"][i],
            "Tag":        data["tags"][i],
        })
    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return df

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Predictive Maintenance")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("- 10,000 samples\n- 6 features\n- 3.39% failure rate\n- Source: UCI ML Repository")
    st.markdown("---")
    st.markdown("**Controls**")
    task = st.radio("Task", ["Binary classification","Multiclass classification"], index=0)
    metric = st.selectbox("Primary metric", ["ACC","AUC","F1","F2"],
                          format_func=lambda x: METRIC_LABELS[x])
    st.markdown("---")
    st.markdown("**Feature importance models**")
    fi_models = st.multiselect(
        "Select models", ["KNN","SVC","RFC","XGB"],
        default=["RFC","XGB"], label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Models: LR · KNN · SVC · RFC · XGB")
    st.caption("Tuned with GridSearchCV (80/10/10 split)")

data = BINARY if task == "Binary classification" else MULTI

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"## Predictive Maintenance — {task}")
st.markdown(f"Comparing **{len(MODELS)} models** on the test set · metric: **{METRIC_LABELS[metric]}**")

# ── Metric cards ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, met, label in zip(
    [c1, c2, c3],
    ["ACC","AUC","F1"],
    ["Best accuracy (test)","Best AUC-ROC (test)","Best F1 score (test)"]
):
    with col:
        bi = best_model_idx(data, met)
        st.markdown(f"""<div class="metric-card">
            <p>{label}</p>
            <h3>{pct(data['test'][met][bi])}</h3>
            <p class="sub">{MODELS[bi]}</p>
        </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <p>Dataset size</p>
        <h3>10,000</h3>
        <p class="sub">3.39% failure rate</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main charts ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.5, 1])
with col_left:
    st.markdown('<p class="section-header">Test set performance</p>', unsafe_allow_html=True)
    st.plotly_chart(build_bar_chart(data, metric), use_container_width=True, config={"displayModeBar": False}, key="chart_main_bar")
with col_right:
    st.markdown('<p class="section-header">Validation vs test</p>', unsafe_allow_html=True)
    st.plotly_chart(build_grouped_bar(data, metric), use_container_width=True, config={"displayModeBar": False}, key="chart_grouped_bar")

col_r, col_d = st.columns([1.4, 1])
with col_r:
    st.markdown('<p class="section-header">All metrics radar — test set</p>', unsafe_allow_html=True)
    st.plotly_chart(build_radar(data), use_container_width=True, config={"displayModeBar": False}, key="chart_radar")
with col_d:
    st.markdown('<p class="section-header">Failure type distribution</p>', unsafe_allow_html=True)
    st.plotly_chart(build_failure_donut(), use_container_width=True, config={"displayModeBar": False}, key="chart_failure_donut")

# ── Table ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Model comparison — test set</p>', unsafe_allow_html=True)
df_table = model_summary_table(data)

def highlight_acc(val):
    v = float(val.strip("%")) / 100
    if v >= 0.975: return "background-color:#1a3a2a; color:#6fcf97"
    if v >= 0.960: return "background-color:#1a2a3a; color:#56b4e9"
    return ""

styled = df_table.style.map(highlight_acc, subset=["Accuracy"])
st.dataframe(styled, use_container_width=True, hide_index=True, height=240)

# ── Feature importance ─────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-header">Permutation feature importance</p>', unsafe_allow_html=True)
if fi_models:
    st.plotly_chart(build_feature_importance(fi_models), use_container_width=True,
                    config={"displayModeBar": False}, key="chart_feature_importance")
    st.caption("📌 Torque, Rotational speed, and Tool wear drive predictions. Type has near-zero importance.")
else:
    st.info("Select at least one model from the sidebar to show feature importance.")

# ── Deep-dive tabs ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-header">Metric deep-dive</p>', unsafe_allow_html=True)
tabs = st.tabs(["Accuracy","AUC-ROC","F1 Score","F2 Score"])
for tab, met in zip(tabs, ["ACC","AUC","F1","F2"]):
    with tab:
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown(f"**Validation — {METRIC_LABELS[met]}**")
            st.plotly_chart(build_bar_chart(data, met, split="val"),
                            use_container_width=True, config={"displayModeBar": False}, key=f"chart_deepdive_{met}_val")
        with tc2:
            st.markdown(f"**Test — {METRIC_LABELS[met]}**")
            st.plotly_chart(build_bar_chart(data, met, split="test"),
                            use_container_width=True, config={"displayModeBar": False}, key=f"chart_deepdive_{met}_test")

# ── Insights ───────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-header">Key insights</p>', unsafe_allow_html=True)

with st.expander("📊 Model ranking & trade-offs", expanded=True):
    st.markdown("""
- **XGBoost** is the best model on all metrics (ACC 97.8%, AUC 99.8%) — trained in just 23s.
- **RFC** matches XGBoost on accuracy (97.4%) but takes **5× longer** to train (1m 54s).
- **SVC** is close behind with 97.4% accuracy and only 49s training time — solid middle ground.
- **KNN** is the fastest tuned model (1s) with 95.1% accuracy — great for quick prototyping.
- **LR** serves as the baseline at 88.3% accuracy with near-instant inference.
""")
with st.expander("🔍 Multiclass vs Binary"):
    st.markdown("""
- All models perform **equally well or better** on multiclass vs binary classification.
- **KNN outperforms SVC and RFC on accuracy** in multiclass (96.5% vs 97.3%/97.4%).
- **XGBoost** training time triples in multiclass (23s → 1m 32s).
- AUC drops for KNN in multiclass (97.9% → 95.7%) — struggles with probability calibration.
""")
with st.expander("⚙️ Feature insights"):
    st.markdown("""
- **Torque** and **Rotational speed** are the most predictive features across all models.
- **Tool wear** is particularly important for predicting Tool Wear Failure specifically.
- **Type** (product quality L/M/H) has near-zero importance — removing it doesn't hurt performance.
- **Air temperature** and **Process temperature** contribute moderately.
""")

st.markdown("---")
st.caption("Dataset: UCI ML Predictive Maintenance · Models tuned with GridSearchCV · 80/10/10 split · SMOTE applied for class imbalance")