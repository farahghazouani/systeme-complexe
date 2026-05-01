"""
MetaSafe-Industrial — Plateforme de Maintenance Prédictive & Détection d'Anomalies
Projet Systèmes Complexes | 2ING1 | 2025-2026
Farah Ghazouani, Siwar Mrayhi, Safa Douma
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MetaSafe-Industrial | Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS  — dark industrial aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Syne:wght@400;700;800&display=swap');

:root {
  --bg:       #0a0c10;
  --surface:  #111318;
  --border:   #1e2230;
  --accent:   #00e5b3;
  --accent2:  #ff5e5e;
  --accent3:  #ffb300;
  --text:     #e2e8f0;
  --muted:    #6b7280;
  --card:     #13161f;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Syne', sans-serif;
}
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Typography ── */
h1 { font-family: 'Syne', sans-serif; font-weight: 800; letter-spacing: -1px; }
h2, h3 { font-family: 'Syne', sans-serif; font-weight: 700; }
code, pre { font-family: 'JetBrains Mono', monospace; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 20px 18px !important;
  position: relative;
  overflow: hidden;
}
[data-testid="metric-container"]::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--accent), transparent);
}
[data-testid="stMetricValue"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 26px !important;
  color: var(--accent) !important;
  font-weight: 700;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricDelta"] { font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; }

/* ── Sidebar ── */
.css-1d391kg, section[data-testid="stSidebar"] > div { padding-top: 1rem; }
[data-testid="stSidebar"] .stSelectbox label { color: var(--accent) !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1px; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
}

/* ── Number inputs ── */
input[type=number] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
  font-family: 'JetBrains Mono', monospace;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, var(--accent) 0%, #00b890 100%) !important;
  color: #000 !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 8px !important;
  letter-spacing: 1px;
  text-transform: uppercase;
  padding: 12px 32px !important;
  transition: all 0.2s ease;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 8px 24px rgba(0,229,179,0.3) !important; }

/* ── Section dividers ── */
hr { border-color: var(--border) !important; }

/* ── Info / warning / error boxes ── */
.stAlert { border-radius: 10px !important; border: 1px solid var(--border) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: var(--surface) !important; border-radius: 10px; padding: 4px; gap: 4px; border: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--muted) !important; border-radius: 8px !important; font-family: 'Syne', sans-serif; font-weight: 600; border: none !important; }
.stTabs [aria-selected="true"] { background: var(--accent) !important; color: #000 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Plotly dark bg override ── */
.js-plotly-plot .plotly { background: transparent !important; }

/* ── Section header ── */
.section-header {
  font-family: 'Syne', sans-serif;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 8px;
  padding-bottom: 4px;
  border-bottom: 1px solid var(--border);
}

/* ── Status badge ── */
.badge-ok   { background: rgba(0,229,179,.15); color: #00e5b3; padding: 4px 12px; border-radius: 100px; font-size: 12px; font-weight: 700; border: 1px solid rgba(0,229,179,.3); }
.badge-warn { background: rgba(255,179,0,.15); color: #ffb300; padding: 4px 12px; border-radius: 100px; font-size: 12px; font-weight: 700; border: 1px solid rgba(255,179,0,.3); }
.badge-err  { background: rgba(255,94,94,.15); color: #ff5e5e; padding: 4px 12px; border-radius: 100px; font-size: 12px; font-weight: 700; border: 1px solid rgba(255,94,94,.3); }

/* ── Anomaly row highlight ── */
.anomaly-card {
  background: rgba(255,94,94,.08);
  border: 1px solid rgba(255,94,94,.25);
  border-radius: 10px;
  padding: 16px;
  margin: 8px 0;
}

/* ── Logo area ── */
.logo-title { font-family:'Syne',sans-serif; font-size:20px; font-weight:800; color:var(--accent); letter-spacing:-0.5px; }
.logo-sub   { font-family:'JetBrains Mono',monospace; font-size:10px; color:var(--muted); letter-spacing:2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PLOTLY TEMPLATE
# ─────────────────────────────────────────────
PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,19,24,0.6)",
        font=dict(family="JetBrains Mono", color="#e2e8f0", size=11),
        title=dict(font=dict(family="Syne", size=14, color="#e2e8f0")),
        xaxis=dict(gridcolor="#1e2230", linecolor="#1e2230", zerolinecolor="#1e2230"),
        yaxis=dict(gridcolor="#1e2230", linecolor="#1e2230", zerolinecolor="#1e2230"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2230"),
        colorway=["#00e5b3", "#ff5e5e", "#ffb300", "#7c83fd", "#f97316", "#a855f7"],
        margin=dict(t=40, b=30, l=40, r=20),
    )
)

def apply_template(fig):
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    return fig

# ─────────────────────────────────────────────
#  DATA & MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model   = joblib.load(os.path.join(current_dir, "modele_maintenance_predictive.pkl"))
    encoder = joblib.load(os.path.join(current_dir, "label_encoder_type.pkl"))
    df      = pd.read_csv(os.path.join(current_dir, "ai4i2020.csv"))
    return model, encoder, df

@st.cache_data(show_spinner=False)
def enrich_dataframe(df_raw):
    df = df_raw.copy()
    # Feature engineering
    df["Temp_Delta"]       = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Power_kW"]         = (df["Torque [Nm]"] * df["Rotational speed [rpm]"]) / 9550
    df["Stress_Index"]     = df["Torque [Nm]"] * df["Tool wear [min]"] / 10000
    df["Failure_Type"]     = "No Failure"
    for col, label in [("TWF","Tool Wear"), ("HDF","Heat Dissipation"), ("PWF","Power"), ("OSF","Over-Strain"), ("RNF","Random")]:
        if col in df.columns:
            df.loc[df[col] == 1, "Failure_Type"] = label
    return df

@st.cache_data(show_spinner=False)
def run_anomaly_detection(df):
    feat_cols = ["Air temperature [K]", "Process temperature [K]",
                 "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
                 "Temp_Delta", "Power_kW", "Stress_Index"]
    X = df[feat_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
    df["Anomaly_IF"]    = iso.fit_predict(X_scaled)  # -1 = anomaly
    df["Anomaly_Score"] = -iso.score_samples(X_scaled)  # higher = more anomalous

    # Z-Score anomaly
    z_scores = np.abs(stats.zscore(X))
    df["Anomaly_ZScore"] = (z_scores > 3).any(axis=1).astype(int)

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X_scaled)
    df["PC1"] = pcs[:, 0]
    df["PC2"] = pcs[:, 1]
    df["PCA_Explained"] = pca.explained_variance_ratio_.sum()

    return df

with st.spinner(""):
    try:
        model, encoder, df_raw = load_resources()
        df = enrich_dataframe(df_raw)
        df = run_anomaly_detection(df)
        LOAD_OK = True
    except Exception as e:
        LOAD_OK = False
        LOAD_ERR = str(e)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 24px">
      <div class="logo-title">⚙ MetaSafe</div>
      <div class="logo-sub">INDUSTRIAL AI PLATFORM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
    page = st.selectbox("", [
        "🏠  Vue d'Ensemble",
        "🔍  Détection d'Anomalies",
        "📊  Analyse des Variables",
        "⚠️  Types de Pannes",
        "🤖  Diagnostic Prédictif",
        "🧪  Analyse de Risque",
    ], label_visibility="collapsed")

    if LOAD_OK:
        st.markdown("---")
        st.markdown('<div class="section-header">Filtres Globaux</div>', unsafe_allow_html=True)
        machine_types = ["Tous"] + sorted(df["Type"].unique().tolist())
        selected_type = st.selectbox("Type Machine", machine_types)
        failure_filter = st.selectbox("Statut", ["Tous", "En Panne", "Opérationnel"])
        
        df_filtered = df.copy()
        if selected_type != "Tous":
            df_filtered = df_filtered[df_filtered["Type"] == selected_type]
        if failure_filter == "En Panne":
            df_filtered = df_filtered[df_filtered["Machine failure"] == 1]
        elif failure_filter == "Opérationnel":
            df_filtered = df_filtered[df_filtered["Machine failure"] == 0]

        n_anomalies = (df_filtered["Anomaly_IF"] == -1).sum()
        n_failures  = df_filtered["Machine failure"].sum()
        total       = len(df_filtered)

        st.markdown("---")
        st.markdown('<div class="section-header">Statut Système</div>', unsafe_allow_html=True)
        
        health_pct = 100 - (n_failures / total * 100)
        color = "#00e5b3" if health_pct > 92 else "#ffb300" if health_pct > 85 else "#ff5e5e"
        badge = "badge-ok" if health_pct > 92 else "badge-warn" if health_pct > 85 else "badge-err"
        status_label = "NOMINAL" if health_pct > 92 else "ATTENTION" if health_pct > 85 else "CRITIQUE"

        st.markdown(f"""
        <div style="text-align:center; margin: 12px 0">
          <div style="font-family:'JetBrains Mono',monospace; font-size:32px; color:{color}; font-weight:700">{health_pct:.1f}%</div>
          <div style="font-size:11px; color:#6b7280; letter-spacing:1px; text-transform:uppercase">Santé du Parc</div>
          <br>
          <span class="{badge}">{status_label}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace; font-size:11px; color:#6b7280; margin-top:16px; line-height:2">
        Machines analysées : <span style="color:#e2e8f0">{total:,}</span><br>
        Pannes détectées : <span style="color:#ff5e5e">{n_failures:,}</span><br>
        Anomalies IF : <span style="color:#ffb300">{n_anomalies:,}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_filtered = pd.DataFrame()

    st.markdown("---")
    st.markdown('<div style="font-size:10px; color:#374151; font-family:JetBrains Mono; text-align:center">v2.0 · Systèmes Complexes · 2ING1</div>', unsafe_allow_html=True)

if not LOAD_OK:
    st.error(f"⛔ Erreur de chargement des ressources : {LOAD_ERR}")
    st.info("Vérifiez que `ai4i2020.csv`, `modele_maintenance_predictive.pkl` et `label_encoder_type.pkl` sont dans le même dossier que `app.py`.")
    st.stop()

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def kpi_row(metrics: list):
    cols = st.columns(len(metrics))
    for col, (label, value, delta, delta_color) in zip(cols, metrics):
        with col:
            st.metric(label, value, delta)

def section(title, icon=""):
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:12px; margin: 28px 0 16px">
      <span style="font-size:22px">{icon}</span>
      <div>
        <div style="font-family:'Syne',sans-serif; font-size:20px; font-weight:800; color:#e2e8f0; line-height:1">{title}</div>
        <div style="height:2px; background:linear-gradient(90deg,#00e5b3,transparent); margin-top:4px; border-radius:2px; width:60px"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
#  PAGE 1 — VUE D'ENSEMBLE
# ═══════════════════════════════════════════════
if "Vue d'Ensemble" in page:
    st.markdown("""
    <div style="margin-bottom:32px">
      <div style="font-family:'Syne',sans-serif; font-size:36px; font-weight:800; line-height:1; letter-spacing:-1px">
        Tableau de Bord <span style="color:#00e5b3">Industriel</span>
      </div>
      <div style="font-family:'JetBrains Mono',monospace; font-size:12px; color:#6b7280; margin-top:6px; letter-spacing:1px">
        DATASET AI4I 2020 — ANALYSE EN TEMPS RÉEL — 10 000 MACHINES
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    total = len(df_filtered)
    n_fail = df_filtered["Machine failure"].sum()
    n_anom = (df_filtered["Anomaly_IF"] == -1).sum()
    fail_rate = n_fail / total * 100 if total else 0
    anom_rate = n_anom / total * 100 if total else 0
    mean_torque = df_filtered["Torque [Nm]"].mean()
    mean_wear   = df_filtered["Tool wear [min]"].mean()

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1: st.metric("Parc Total", f"{total:,}", "machines")
    with col2: st.metric("Taux de Panne", f"{fail_rate:.2f}%", f"{int(n_fail)} arrêts")
    with col3: st.metric("Anomalies IF", f"{anom_rate:.1f}%", f"{int(n_anom)} détectées")
    with col4: st.metric("Couple Moyen", f"{mean_torque:.1f} Nm")
    with col5: st.metric("Usure Moy. Outil", f"{mean_wear:.0f} min")
    with col6:
        power_mean = df_filtered["Power_kW"].mean()
        st.metric("Puissance Moy.", f"{power_mean:.2f} kW")

    st.markdown("---")

    # Row 1: Failure distribution + Type breakdown
    c1, c2, c3 = st.columns([2, 2, 1.5])

    with c1:
        section("Distribution des Pannes par Type", "🛠️")
        fail_type_counts = df_filtered["Failure_Type"].value_counts().reset_index()
        fail_type_counts.columns = ["Type", "Count"]
        fig = px.bar(fail_type_counts, x="Type", y="Count",
                     color="Type",
                     color_discrete_sequence=["#00e5b3","#ff5e5e","#ffb300","#7c83fd","#f97316","#a855f7"],
                     title="Répartition des modes de défaillance")
        apply_template(fig)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        section("Carte de Chaleur — Corrélations", "🌡️")
        corr_cols = ["Air temperature [K]", "Process temperature [K]",
                     "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
                     "Temp_Delta", "Power_kW", "Stress_Index", "Machine failure"]
        corr = df_filtered[corr_cols].corr()
        short_labels = ["T.Air","T.Proc","RPM","Torque","Usure","ΔTemp","Puiss","Stress","Panne"]
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=short_labels, y=short_labels,
            colorscale=[[0,"#ff5e5e"],[0.5,"#111318"],[1,"#00e5b3"]],
            zmid=0, text=np.round(corr.values,2),
            texttemplate="%{text}", textfont=dict(size=9),
            showscale=True
        ))
        apply_template(fig)
        fig.update_layout(title="Matrice de corrélation")
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        section("Score de Risque", "📈")
        risk_by_type = df_filtered.groupby("Type")["Anomaly_Score"].mean().sort_values(ascending=False)
        for t, score in risk_by_type.items():
            norm = min(score / (risk_by_type.max() + 1e-9), 1.0)
            color = "#ff5e5e" if norm > 0.7 else "#ffb300" if norm > 0.4 else "#00e5b3"
            st.markdown(f"""
            <div style="margin:10px 0">
              <div style="display:flex; justify-content:space-between; margin-bottom:4px">
                <span style="font-family:'JetBrains Mono',monospace; font-size:12px">Type {t}</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:12px; color:{color}">{score:.3f}</span>
              </div>
              <div style="height:6px; background:#1e2230; border-radius:3px; overflow:hidden">
                <div style="height:100%; width:{norm*100:.0f}%; background:{color}; border-radius:3px; transition:all .3s"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Row 2: Sunburst + Scatter Power
    c4, c5 = st.columns(2)

    with c4:
        section("Répartition Type × Statut", "🔵")
        fig = px.sunburst(df_filtered, path=["Type", "Failure_Type"],
                          color="Type",
                          color_discrete_sequence=["#00e5b3","#7c83fd","#ffb300"],
                          title="Vue hiérarchique Type → Mode de panne")
        apply_template(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c5:
        section("Puissance vs Usure Outil", "⚡")
        sample = df_filtered.sample(min(3000, len(df_filtered)), random_state=42)
        fig = px.scatter(sample, x="Tool wear [min]", y="Power_kW",
                         color="Machine failure",
                         color_continuous_scale=["#00e5b3", "#ff5e5e"],
                         opacity=0.5, size_max=4,
                         title="Corrélation Puissance / Usure Outil",
                         labels={"Machine failure": "Panne"})
        apply_template(fig)
        fig.update_traces(marker=dict(size=4))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════
#  PAGE 2 — DÉTECTION D'ANOMALIES
# ═══════════════════════════════════════════════
elif "Anomalies" in page:
    st.markdown("""
    <div style="margin-bottom:32px">
      <div style="font-family:'Syne',sans-serif; font-size:36px; font-weight:800; line-height:1; letter-spacing:-1px">
        Détection d'<span style="color:#ff5e5e">Anomalies</span>
      </div>
      <div style="font-family:'JetBrains Mono',monospace; font-size:12px; color:#6b7280; margin-top:6px; letter-spacing:1px">
        ISOLATION FOREST + Z-SCORE — ANALYSE MULTIDIMENSIONNELLE
      </div>
    </div>
    """, unsafe_allow_html=True)

    n_anom_if = (df_filtered["Anomaly_IF"] == -1).sum()
    n_anom_z  = df_filtered["Anomaly_ZScore"].sum()
    overlap   = ((df_filtered["Anomaly_IF"] == -1) & (df_filtered["Anomaly_ZScore"] == 1)).sum()
    score_mean = df_filtered[df_filtered["Anomaly_IF"] == -1]["Anomaly_Score"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Anomalies Isolation Forest", f"{n_anom_if}", f"{n_anom_if/len(df_filtered)*100:.1f}% du parc")
    with c2: st.metric("Anomalies Z-Score (σ>3)", f"{n_anom_z}", f"{n_anom_z/len(df_filtered)*100:.1f}% du parc")
    with c3: st.metric("Chevauchement IF ∩ Z", f"{overlap}", "double détection")
    with c4: st.metric("Score Anomalie Moyen", f"{score_mean:.3f}", "machines critiques")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["  PCA 2D  ", "  Score Distribution  ", "  Heatmap Multivariée  ", "  Tableau d'Alertes  "])

    with tab1:
        section("Projection PCA — Espace des Anomalies", "🔭")
        st.info(f"📐 Variance expliquée par les 2 composantes principales : **{df_filtered['PCA_Explained'].iloc[0]*100:.1f}%**")
        
        sample = df_filtered.sample(min(4000, len(df_filtered)), random_state=42)
        sample["Statut"] = sample["Anomaly_IF"].map({1: "Normal", -1: "Anomalie IF"})
        
        fig = px.scatter(sample, x="PC1", y="PC2", color="Statut",
                         color_discrete_map={"Normal": "#00e5b3", "Anomalie IF": "#ff5e5e"},
                         symbol="Statut",
                         opacity=0.6,
                         hover_data=["Torque [Nm]", "Rotational speed [rpm]", "Tool wear [min]"],
                         title="Espace PCA — Détection Isolation Forest")
        apply_template(fig)
        fig.update_traces(marker=dict(size=5))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        section("Distribution du Score d'Anomalie", "📊")
        c_a, c_b = st.columns(2)
        
        with c_a:
            fig = go.Figure()
            # Normal machines
            normal_scores = df_filtered[df_filtered["Anomaly_IF"] == 1]["Anomaly_Score"]
            anom_scores   = df_filtered[df_filtered["Anomaly_IF"] == -1]["Anomaly_Score"]
            
            fig.add_trace(go.Histogram(x=normal_scores, name="Normal", nbinsx=60,
                                       marker_color="#00e5b3", opacity=0.7))
            fig.add_trace(go.Histogram(x=anom_scores, name="Anomalie", nbinsx=60,
                                       marker_color="#ff5e5e", opacity=0.8))
            fig.update_layout(barmode="overlay", title="Distribution des scores d'anomalie")
            apply_template(fig)
            st.plotly_chart(fig, use_container_width=True)

        with c_b:
            # Anomaly score vs failure
            sample = df_filtered.sample(min(2000, len(df_filtered)), random_state=42)
            fig = px.scatter(sample, x="Anomaly_Score", y="Stress_Index",
                             color="Machine failure",
                             color_continuous_scale=["#00e5b3","#ff5e5e"],
                             title="Score d'Anomalie vs Indice de Stress",
                             labels={"Machine failure": "Panne"})
            apply_template(fig)
            fig.update_traces(marker=dict(size=4, opacity=0.6))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        section("Heatmap Multivariée des Anomalies", "🗺️")
        feat_cols = ["Air temperature [K]", "Process temperature [K]",
                     "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
                     "Temp_Delta", "Power_kW", "Stress_Index"]
        
        anom_df   = df_filtered[df_filtered["Anomaly_IF"] == -1][feat_cols].describe().T
        normal_df = df_filtered[df_filtered["Anomaly_IF"] == 1][feat_cols].describe().T

        diff = ((anom_df["mean"] - normal_df["mean"]) / normal_df["mean"] * 100).reset_index()
        diff.columns = ["Feature", "Écart (%)"]
        diff["Abs"] = diff["Écart (%)"].abs()
        diff = diff.sort_values("Abs", ascending=True)

        fig = go.Figure(go.Bar(
            x=diff["Écart (%)"],
            y=diff["Feature"],
            orientation="h",
            marker=dict(
                color=diff["Écart (%)"],
                colorscale=[[0,"#ff5e5e"],[0.5,"#ffb300"],[1,"#00e5b3"]],
                cmid=0,
            )
        ))
        apply_template(fig)
        fig.update_layout(title="Écart moyen (%) : Anomalies vs Normaux")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        section("Tableau d'Alertes Critiques", "🚨")
        alert_df = df_filtered[df_filtered["Anomaly_IF"] == -1].copy()
        alert_df["Niveau"] = alert_df["Anomaly_Score"].apply(
            lambda s: "🔴 CRITIQUE" if s > alert_df["Anomaly_Score"].quantile(0.9)
                      else ("🟡 ÉLEVÉ" if s > alert_df["Anomaly_Score"].quantile(0.7) else "🟠 MODÉRÉ")
        )
        display_cols = ["UDI", "Type", "Torque [Nm]", "Rotational speed [rpm]",
                        "Tool wear [min]", "Anomaly_Score", "Machine failure", "Niveau"]
        existing = [c for c in display_cols if c in alert_df.columns]
        st.dataframe(
            alert_df[existing].sort_values("Anomaly_Score", ascending=False).head(50),
            use_container_width=True,
            hide_index=True,
        )


# ═══════════════════════════════════════════════
#  PAGE 3 — ANALYSE DES VARIABLES
# ═══════════════════════════════════════════════
elif "Variables" in page:
    st.markdown("""
    <div style="margin-bottom:32px">
      <div style="font-family:'Syne',sans-serif; font-size:36px; font-weight:800; line-height:1; letter-spacing:-1px">
        Analyse des <span style="color:#ffb300">Variables</span>
      </div>
      <div style="font-family:'JetBrains Mono',monospace; font-size:12px; color:#6b7280; margin-top:6px; letter-spacing:1px">
        DISTRIBUTIONS — BOÎTES À MOUSTACHES — ANALYSE DE SÉPARATION
      </div>
    </div>
    """, unsafe_allow_html=True)

    vars_map = {
        "Température Air [K]":      "Air temperature [K]",
        "Température Process [K]":  "Process temperature [K]",
        "Vitesse Rotation [RPM]":   "Rotational speed [rpm]",
        "Couple [Nm]":              "Torque [Nm]",
        "Usure Outil [min]":        "Tool wear [min]",
        "Delta Température [K]":    "Temp_Delta",
        "Puissance [kW]":           "Power_kW",
        "Indice de Stress":         "Stress_Index",
    }
    var_label = st.selectbox("Variable à analyser", list(vars_map.keys()))
    var_col   = vars_map[var_label]

    df_filtered["Statut_Label"] = df_filtered["Machine failure"].map({0: "✅ Opérationnel", 1: "❌ En Panne"})

    c1, c2 = st.columns(2)

    with c1:
        section("Distribution Comparative", "📊")
        fig = px.histogram(df_filtered, x=var_col, color="Statut_Label",
                           nbins=60, barmode="overlay",
                           marginal="violin",
                           color_discrete_map={"✅ Opérationnel": "#00e5b3", "❌ En Panne": "#ff5e5e"},
                           title=f"Distribution : {var_label}")
        apply_template(fig)
        fig.update_traces(opacity=0.75)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        section("Boîte à Moustaches par Type", "📦")
        fig = px.box(df_filtered, x="Type", y=var_col,
                     color="Statut_Label",
                     color_discrete_map={"✅ Opérationnel": "#00e5b3", "❌ En Panne": "#ff5e5e"},
                     title=f"Box plot : {var_label} par type de machine",
                     points="outliers")
        apply_template(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Separator power analysis
    section("Pouvoir de Séparation des Variables", "🎯")
    st.caption("Plus l'overlap KDE est faible, meilleur est l'indicateur de panne pour cette variable.")

    sep_cols = [c for c in vars_map.values() if c in df_filtered.columns]
    sep_results = []
    for col in sep_cols:
        g0 = df_filtered[df_filtered["Machine failure"] == 0][col].dropna()
        g1 = df_filtered[df_filtered["Machine failure"] == 1][col].dropna()
        if len(g1) > 10:
            t_stat, p_val = stats.ttest_ind(g0, g1, equal_var=False)
            effect = abs(g0.mean() - g1.mean()) / ((g0.std() + g1.std()) / 2 + 1e-9)
            sep_results.append({"Variable": col, "Cohen's d": round(effect, 3), "p-value": f"{p_val:.2e}"})

    sep_df = pd.DataFrame(sep_results).sort_values("Cohen's d", ascending=False)
    sep_df["Importance"] = sep_df["Cohen's d"] / sep_df["Cohen's d"].max()

    fig = px.bar(sep_df, x="Variable", y="Cohen's d",
                 color="Cohen's d",
                 color_continuous_scale=[[0,"#1e2230"],[0.4,"#ffb300"],[1,"#ff5e5e"]],
                 title="Cohen's d — Effet de séparation (Sain vs Panne)",
                 text="Cohen's d")
    apply_template(fig)
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════
#  PAGE 4 — TYPES DE PANNES
# ═══════════════════════════════════════════════
elif "Pannes" in page:
    st.markdown("""
    <div style="margin-bottom:32px">
      <div style="font-family:'Syne',sans-serif; font-size:36px; font-weight:800; line-height:1; letter-spacing:-1px">
        Analyse des <span style="color:#a855f7">Pannes</span>
      </div>
      <div style="font-family:'JetBrains Mono',monospace; font-size:12px; color:#6b7280; margin-top:6px; letter-spacing:1px">
        TWF · HDF · PWF · OSF · RNF — MODES DE DÉFAILLANCE
      </div>
    </div>
    """, unsafe_allow_html=True)

    failure_cols = {
        "TWF": "Tool Wear Failure",
        "HDF": "Heat Dissipation Failure",
        "PWF": "Power Failure",
        "OSF": "Over-Strain Failure",
        "RNF": "Random Failure",
    }
    existing_failures = {k: v for k, v in failure_cols.items() if k in df_filtered.columns}

    # Count each failure type
    counts = {v: df_filtered[k].sum() for k, v in existing_failures.items()}
    count_df = pd.DataFrame(list(counts.items()), columns=["Mode", "Count"]).sort_values("Count", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        section("Fréquence par Mode", "📉")
        colors = ["#ff5e5e","#ffb300","#7c83fd","#00e5b3","#f97316"]
        fig = px.bar(count_df, x="Mode", y="Count",
                     color="Mode",
                     color_discrete_sequence=colors,
                     title="Nombre d'occurrences par mode de panne",
                     text="Count")
        apply_template(fig)
        fig.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        section("Répartition Proportionnelle", "🥧")
        fig = go.Figure(go.Pie(
            labels=count_df["Mode"],
            values=count_df["Count"],
            hole=0.55,
            marker=dict(colors=colors, line=dict(color="#0a0c10", width=2)),
            textinfo="label+percent",
            textfont=dict(size=11),
        ))
        apply_template(fig)
        fig.update_layout(title="Part relative de chaque mode de défaillance",
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Parallel coordinates for multivariate view
    section("Coordonnées Parallèles — Profil des Pannes", "〰️")
    sample_fail = df_filtered[df_filtered["Machine failure"] == 1].sample(min(500, df_filtered["Machine failure"].sum()), random_state=42)
    sample_ok   = df_filtered[df_filtered["Machine failure"] == 0].sample(min(500, len(df_filtered[df_filtered["Machine failure"]==0])), random_state=42)
    combo = pd.concat([sample_fail, sample_ok])

    feat_pc = ["Air temperature [K]", "Process temperature [K]",
               "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Machine failure"]
    fig = px.parallel_coordinates(combo, color="Machine failure",
                                  dimensions=feat_pc,
                                  color_continuous_scale=["#00e5b3","#ff5e5e"],
                                  title="Profil multivariable : Opérationnel vs En Panne")
    apply_template(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Per-type failure rates
    section("Taux de Panne par Type de Machine", "🏭")
    fail_rate_type = df_filtered.groupby("Type")["Machine failure"].agg(["sum","count"]).reset_index()
    fail_rate_type["Taux (%)"] = fail_rate_type["sum"] / fail_rate_type["count"] * 100
    fail_rate_type.columns = ["Type","Pannes","Total","Taux (%)"]
    
    fig = px.bar(fail_rate_type, x="Type", y="Taux (%)", color="Taux (%)",
                 color_continuous_scale=[[0,"#00e5b3"],[0.5,"#ffb300"],[1,"#ff5e5e"]],
                 title="Taux de défaillance (%) par type de machine",
                 text="Taux (%)")
    apply_template(fig)
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════
#  PAGE 5 — DIAGNOSTIC PRÉDICTIF
# ═══════════════════════════════════════════════
elif "Diagnostic" in page:
    st.markdown("""
    <div style="margin-bottom:32px">
      <div style="font-family:'Syne',sans-serif; font-size:36px; font-weight:800; line-height:1; letter-spacing:-1px">
        Diagnostic <span style="color:#00e5b3">Prédictif</span>
      </div>
      <div style="font-family:'JetBrains Mono',monospace; font-size:12px; color:#6b7280; margin-top:6px; letter-spacing:1px">
        INTELLIGENCE ARTIFICIELLE — RANDOM FOREST — PRÉDICTION TEMPS RÉEL
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1.2])

    with col_form:
        section("Données Capteurs", "🔧")
        
        m_type   = st.selectbox("Type de Machine", ["L", "M", "H"],
                                help="L=Light, M=Medium, H=Heavy duty")
        
        c_a, c_b = st.columns(2)
        with c_a:
            air_temp  = st.number_input("Temp. Air [K]", 280.0, 320.0, 300.0, step=0.5)
            speed     = st.number_input("Vitesse [RPM]", 1000, 3000, 1500, step=50)
            wear      = st.number_input("Usure Outil [min]", 0, 300, 100, step=5)
        with c_b:
            proc_temp = st.number_input("Temp. Process [K]", 300.0, 340.0, 310.0, step=0.5)
            torque    = st.number_input("Couple [Nm]", 0.0, 100.0, 40.0, step=0.5)

        # Derived values preview
        delta_t = proc_temp - air_temp
        power   = (torque * speed) / 9550
        stress  = torque * wear / 10000

        st.markdown(f"""
        <div style="background:#111318; border:1px solid #1e2230; border-radius:10px; padding:16px; margin-top:12px">
          <div class="section-header" style="margin-bottom:10px">Valeurs Dérivées</div>
          <div style="font-family:'JetBrains Mono',monospace; font-size:12px; line-height:2; color:#9ca3af">
            ΔTempérature  : <span style="color:#00e5b3">{delta_t:.1f} K</span><br>
            Puissance      : <span style="color:#00e5b3">{power:.2f} kW</span><br>
            Indice Stress  : <span style="color:#ffb300">{stress:.3f}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        run_diag = st.button("⚡  LANCER LE DIAGNOSTIC", use_container_width=True)

    with col_result:
        if run_diag:
            type_enc = encoder.transform([m_type])[0]
            X_input  = np.array([[type_enc, air_temp, proc_temp, speed, torque, wear]])
            pred     = model.predict(X_input)[0]
            prob     = model.predict_proba(X_input)[0][1]

            # Gauge
            section("Résultat du Diagnostic", "🎯")
            gauge_color = "#ff5e5e" if prob > 0.7 else "#ffb300" if prob > 0.4 else "#00e5b3"
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                delta={"reference": 40, "suffix": "%", "increasing": {"color": "#ff5e5e"}},
                number={"suffix": "%", "font": {"family": "JetBrains Mono", "size": 36, "color": gauge_color}},
                title={"text": "Probabilité de Défaillance", "font": {"family": "Syne", "size": 14, "color": "#e2e8f0"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#6b7280",
                             "tickfont": {"size": 10, "family": "JetBrains Mono"}},
                    "bar":  {"color": gauge_color, "thickness": 0.25},
                    "bgcolor": "#111318",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40],  "color": "rgba(0,229,179,.08)"},
                        {"range": [40, 70], "color": "rgba(255,179,0,.08)"},
                        {"range": [70, 100],"color": "rgba(255,94,94,.08)"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 2}, "thickness": 0.8, "value": prob*100},
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                height=260,
                font={"color": "#e2e8f0"},
                margin=dict(t=30, b=10, l=30, r=30),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Decision banner
            if pred == 1:
                st.markdown(f"""
                <div class="anomaly-card" style="text-align:center">
                  <div style="font-size:28px; margin-bottom:6px">🚨</div>
                  <div style="font-family:'Syne',sans-serif; font-size:18px; font-weight:800; color:#ff5e5e">DÉFAILLANCE IMMINENTE</div>
                  <div style="font-family:'JetBrains Mono',monospace; font-size:13px; color:#9ca3af; margin-top:4px">
                    Probabilité de panne : <strong style="color:#ff5e5e">{prob:.1%}</strong>
                  </div>
                  <div style="font-size:12px; color:#6b7280; margin-top:8px">→ Intervention préventive recommandée immédiatement</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:rgba(0,229,179,.06); border:1px solid rgba(0,229,179,.2); border-radius:10px; padding:20px; text-align:center">
                  <div style="font-size:28px; margin-bottom:6px">✅</div>
                  <div style="font-family:'Syne',sans-serif; font-size:18px; font-weight:800; color:#00e5b3">MACHINE OPÉRATIONNELLE</div>
                  <div style="font-family:'JetBrains Mono',monospace; font-size:13px; color:#9ca3af; margin-top:4px">
                    Risque de panne : <strong style="color:#00e5b3">{prob:.1%}</strong>
                  </div>
                  <div style="font-size:12px; color:#6b7280; margin-top:8px">→ Prochain contrôle selon planning standard</div>
                </div>
                """, unsafe_allow_html=True)

            # Feature contribution radar
            section("Contribution des Capteurs", "📡")
            features_norm = {
                "T.Air":     (air_temp - 295) / 10,
                "T.Process": (proc_temp - 309) / 7,
                "RPM":       (speed - 1538) / 179,
                "Torque":    (torque - 39.9) / 9.9,
                "Usure":     (wear - 107) / 63,
            }
            cats  = list(features_norm.keys())
            vals  = [abs(v) for v in features_norm.values()]
            vals_norm = [min(v / (max(vals)+1e-9), 1.0) for v in vals]
            
            fig_radar = go.Figure(go.Scatterpolar(
                r=vals_norm + [vals_norm[0]],
                theta=cats + [cats[0]],
                fill="toself",
                fillcolor=f"rgba{tuple(int(gauge_color.lstrip('#')[i:i+2],16) for i in (0,2,4)) + (0.15,)}",
                line=dict(color=gauge_color, width=2),
                name="Déviation σ",
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(17,19,24,0.8)",
                    radialaxis=dict(visible=True, range=[0,1], gridcolor="#1e2230",
                                   tickfont=dict(size=8, family="JetBrains Mono")),
                    angularaxis=dict(gridcolor="#1e2230",
                                    tickfont=dict(size=11, family="Syne", color="#e2e8f0")),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                height=280,
                margin=dict(t=10, b=10, l=30, r=30),
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.markdown("""
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:400px; border:1px dashed #1e2230; border-radius:16px; color:#374151">
              <div style="font-size:48px; margin-bottom:16px">⚙️</div>
              <div style="font-family:'Syne',sans-serif; font-size:16px; font-weight:700">Entrez les paramètres capteurs</div>
              <div style="font-family:'JetBrains Mono',monospace; font-size:11px; margin-top:6px; letter-spacing:1px">puis lancez le diagnostic</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  PAGE 6 — ANALYSE DE RISQUE
# ═══════════════════════════════════════════════
elif "Risque" in page:
    st.markdown("""
    <div style="margin-bottom:32px">
      <div style="font-family:'Syne',sans-serif; font-size:36px; font-weight:800; line-height:1; letter-spacing:-1px">
        Analyse de <span style="color:#f97316">Risque</span>
      </div>
      <div style="font-family:'JetBrains Mono',monospace; font-size:12px; color:#6b7280; margin-top:6px; letter-spacing:1px">
        STRESS MÉCANIQUE — ZONES DE DANGER — CARTOGRAPHIE DES RISQUES
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        section("Zone de Danger Couple × RPM", "🌀")
        sample = df_filtered.sample(min(3000, len(df_filtered)), random_state=42)
        fig = px.scatter(sample, x="Torque [Nm]", y="Rotational speed [rpm]",
                         color="Machine failure",
                         color_continuous_scale=["#00e5b3","#ff5e5e"],
                         size="Tool wear [min]", size_max=10,
                         opacity=0.6,
                         hover_data=["Type","Failure_Type"],
                         title="Corrélation Couple / Vitesse — Zones de rupture")
        apply_template(fig)
        # Add reference lines
        fig.add_vline(x=df_filtered["Torque [Nm]"].quantile(0.9),
                      line_dash="dash", line_color="#ffb300", annotation_text="90e pct.")
        fig.add_hline(y=df_filtered["Rotational speed [rpm]"].quantile(0.9),
                      line_dash="dash", line_color="#ffb300")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        section("Carte de Risque — Indice de Stress", "🗺️")
        fig = px.density_heatmap(df_filtered, x="Stress_Index", y="Power_kW",
                                  z="Machine failure",
                                  histfunc="avg",
                                  nbinsx=30, nbinsy=30,
                                  color_continuous_scale=[[0,"#111318"],[0.3,"#1e2230"],[0.7,"#ffb300"],[1,"#ff5e5e"]],
                                  title="Probabilité de panne — Stress vs Puissance")
        apply_template(fig)
        st.plotly_chart(fig, use_container_width=True)

    # 3D scatter
    section("Vue 3D — Espace des États Machines", "🧊")
    sample3d = df_filtered.sample(min(2000, len(df_filtered)), random_state=42)
    fig3d = px.scatter_3d(sample3d,
                          x="Torque [Nm]", y="Rotational speed [rpm]", z="Tool wear [min]",
                          color="Machine failure",
                          color_continuous_scale=["#00e5b3","#ff5e5e"],
                          opacity=0.5, size_max=4,
                          title="Espace 3D : Torque × RPM × Usure Outil")
    fig3d.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="#111318",
            xaxis=dict(gridcolor="#1e2230", showbackground=False),
            yaxis=dict(gridcolor="#1e2230", showbackground=False),
            zaxis=dict(gridcolor="#1e2230", showbackground=False),
        ),
        font=dict(family="JetBrains Mono", color="#e2e8f0"),
        margin=dict(t=40, b=0, l=0, r=0),
        height=500,
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # Risk summary table
    section("Synthèse des Indicateurs de Risque", "📋")
    risk_cols = ["Torque [Nm]", "Rotational speed [rpm]", "Tool wear [min]", "Temp_Delta", "Power_kW", "Stress_Index"]
    available_risk = [c for c in risk_cols if c in df_filtered.columns]
    risk_summary = df_filtered.groupby("Machine failure")[available_risk].describe().T
    st.dataframe(risk_summary.style.background_gradient(cmap="RdYlGn_r", axis=1), use_container_width=True)
