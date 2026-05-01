# =====================================================================
# 🏭 INDUSTRIAL AI INSIGHTS – Dashboard de Maintenance Prédictive
# =====================================================================
# Secrets Streamlit Cloud (App Settings → Secrets) :
#
#   [google_auth]
#   client_id     = "....apps.googleusercontent.com"
#   client_secret = "GOCSPX-..."
#   redirect_uri  = "https://systeme-complexe-gpnqeejyqc5cbxsykqzyn6.streamlit.app"
# =====================================================================

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from scipy import stats as sp_stats

# =====================================================================
# 1. CONFIGURATION DE LA PAGE & STYLE
# =====================================================================
st.set_page_config(
    page_title="Industrial AI Insights – Maintenance Prédictive",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOR_OK      = "#00CC96"
COLOR_FAIL    = "#EF553B"
COLOR_WARNING = "#FFA500"
COLOR_PRIMARY = "#00B4D8"
TEMPLATE      = "plotly_dark"

SENSOR_VARS = {
    "Air temperature [K]":     "Température Air (K)",
    "Process temperature [K]": "Température Process (K)",
    "Rotational speed [rpm]":  "Vitesse de rotation (RPM)",
    "Torque [Nm]":             "Couple (Nm)",
    "Tool wear [min]":         "Usure outil (min)",
}

FAILURE_MODES = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 28px; color: #00B4D8 !important; font-weight: 700;}
[data-testid="stMetricLabel"] { font-size: 14px; color: #CCCCCC !important;}
.stMetric {
    background-color: rgba(255,255,255,0.04); border-radius: 12px;
    padding: 18px; border: 1px solid #2A2A3E; box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.stSelectbox label, .stNumberInput label, .stSlider label { color: #00B4D8 !important; font-weight: 600; }
h1 { color: #FFFFFF; border-bottom: 3px solid #00B4D8; padding-bottom: 10px;}
h2 { color: #00B4D8; }
h3 { color: #FFFFFF; }
.insight-box {
    background: linear-gradient(135deg,rgba(0,180,216,0.1) 0%,rgba(123,44,191,0.1) 100%);
    border-left: 4px solid #00B4D8; padding: 15px; border-radius: 8px; margin: 10px 0;
}
.alert-danger {
    background: linear-gradient(135deg,rgba(239,85,59,0.15) 0%,rgba(239,85,59,0.05) 100%);
    border-left: 4px solid #EF553B; padding: 15px; border-radius: 8px;
}
.alert-success {
    background: linear-gradient(135deg,rgba(0,204,150,0.15) 0%,rgba(0,204,150,0.05) 100%);
    border-left: 4px solid #00CC96; padding: 15px; border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. AUTHENTIFICATION – GOOGLE OAUTH2
# =====================================================================
from streamlit_google_auth import Authenticate

# ──────────────────────────────────────────────────────────────────
# On lit [google_auth] — c'est la section que tu as dans Streamlit Cloud
# ──────────────────────────────────────────────────────────────────
SECRET_SECTION = "google_auth"   # ← une seule ligne à changer si besoin

try:
    _s = st.secrets[SECRET_SECTION]
    auth = Authenticate(
        client_id=_s["client_id"],
        client_secret=_s["client_secret"],
        redirect_uri=_s["redirect_uri"],
    )
except KeyError as e:
    st.error(
        f"❌ Secret manquant : {e}\n\n"
        f"Dans **Streamlit Cloud → App Settings → Secrets**, ajoute :\n"
        f"```toml\n[{SECRET_SECTION}]\n"
        f'client_id     = "....apps.googleusercontent.com"\n'
        f'client_secret = "GOCSPX-..."\n'
        f'redirect_uri  = "https://systeme-complexe-gpnqeejyqc5cbxsykqzyn6.streamlit.app"\n```'
    )
    st.stop()

# Callback OAuth — DOIT être appelé avant tout affichage conditionnel
auth.check_authentification()

# ── Page de login ──
if not st.session_state.get("connected"):
    _, col_c, _ = st.columns([1, 2, 1])
    with col_c:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            "<h1 style='text-align:center;color:#00B4D8'>🏭 Industrial AI Insights</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:center;color:#aaa'>"
            "Connecte-toi avec ton compte Google pour accéder au dashboard.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        auth.login()
    st.stop()

# ── Infos utilisateur ──
user_info = st.session_state.get("connected_user", {})
fullname  = user_info.get("name", "Utilisateur")
email     = user_info.get("email", "")
avatar    = user_info.get("picture", "https://cdn-icons-png.flaticon.com/512/1067/1067357.png")
role      = "Administrateur" if email == "farahghazouani@gmail.com" else "Opérateur"

st.session_state["fullname"] = fullname
st.session_state["email"]    = email
st.session_state["role"]     = role

# =====================================================================
# 3. CHARGEMENT DES DONNÉES & MODÈLE
# =====================================================================
@st.cache_resource
def load_resources():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model   = joblib.load(os.path.join(current_dir, "modele_maintenance_predictive.pkl"))
    encoder = joblib.load(os.path.join(current_dir, "label_encoder_type.pkl"))
    df      = pd.read_csv(os.path.join(current_dir, "ai4i2020.csv"))
    df["Status"]           = df["Machine failure"].map({0: "Sain", 1: "En Panne"})
    df["Temp Diff"]        = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Mechanical Power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60
    return model, encoder, df

try:
    model, encoder, df = load_resources()
except Exception as e:
    st.error(f"❌ Erreur de chargement des ressources : {e}")
    st.stop()

# =====================================================================
# 4. BARRE LATÉRALE
# =====================================================================
with st.sidebar:
    st.image(avatar, width=70)
    st.title("Fleet Manager AI")
    st.caption("Maintenance Prédictive Industrielle")
    st.markdown("---")

    st.markdown(
        f"<div style='background:rgba(0,180,216,0.08);border:1px solid #2A2A3E;"
        f"border-radius:10px;padding:12px;margin-bottom:10px;'>"
        f"<span style='color:#888;font-size:12px;'>Connecté via Google SSO</span><br>"
        f"<b style='color:#00B4D8;'>👤 {st.session_state['fullname']}</b><br>"
        f"<span style='color:#AAA;font-size:12px;'>✉️ {st.session_state['email']}</span><br>"
        f"<span style='color:#AAA;font-size:12px;'>🏷️ {st.session_state['role']}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if st.button("🚪 Se déconnecter", use_container_width=True):
        auth.logout()
        st.rerun()

    st.markdown("---")
    page = st.radio(
        "📌 **Navigation**",
        [
            "🏠 Vue d'Ensemble",
            "📊 Distribution & Comportement",
            "🔬 Analyse Multivariée",
            "⚠️ Modes de Défaillance",
            "🤖 Diagnostic Prédictif",
        ],
    )

    st.markdown("---")
    st.markdown("### 🎛️ Filtres globaux")
    selected_types = st.multiselect(
        "Type de machine",
        options=df["Type"].unique().tolist(),
        default=df["Type"].unique().tolist(),
        help="L = Low, M = Medium, H = High quality",
    )
    df_filtered = df[df["Type"].isin(selected_types)] if selected_types else df

    st.markdown("---")
    st.caption(f"📁 **Dataset :** {len(df):,} observations")
    st.caption(f"🔍 **Filtré :** {len(df_filtered):,} observations")
    st.caption(f"⚙️ **Modèle :** {type(model).__name__}")

# =====================================================================
# PAGE 1 : VUE D'ENSEMBLE
# =====================================================================
if page == "🏠 Vue d'Ensemble":
    st.title("🏭 Vue d'Ensemble du Parc Machines")
    st.markdown("**Objectif :** Synthèse des indicateurs critiques de santé du parc industriel.")

    total      = len(df_filtered)
    fails      = int(df_filtered["Machine failure"].sum())
    fail_rate  = (fails / total * 100) if total > 0 else 0
    avg_wear   = df_filtered["Tool wear [min]"].mean()
    avg_torque = df_filtered["Torque [Nm]"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🏭 Parc total",   f"{total:,}",            "machines surveillées")
    col2.metric("🚨 Pannes",        f"{fails:,}",            f"{fail_rate:.2f} % du parc", delta_color="inverse")
    col3.metric("✅ Disponibilité", f"{100-fail_rate:.2f} %","Taux opérationnel")
    col4.metric("🔧 Usure moy.",    f"{avg_wear:.0f} min",   "Tool wear moyen")
    col5.metric("⚙️ Couple moy.",   f"{avg_torque:.1f} Nm",  "Torque moyen")

    st.markdown("---")
    c1, c2 = st.columns([1, 1.3])

    with c1:
        st.subheader("⚖️ Répartition Sain / Panne")
        status_counts = df_filtered["Status"].value_counts().reset_index()
        status_counts.columns = ["Statut", "Nombre"]
        fig_donut = px.pie(
            status_counts, values="Nombre", names="Statut", hole=0.55,
            color="Statut",
            color_discrete_map={"Sain": COLOR_OK, "En Panne": COLOR_FAIL},
            template=TEMPLATE,
        )
        fig_donut.update_traces(
            textposition="outside", textinfo="label+percent",
            marker=dict(line=dict(color="#000", width=2)),
        )
        fig_donut.update_layout(
            showlegend=True, legend=dict(orientation="h", y=-0.1),
            margin=dict(t=10, b=10, l=10, r=10),
            annotations=[dict(
                text=f"{fail_rate:.1f}%<br><span style='font-size:12px'>panne</span>",
                x=0.5, y=0.5, font_size=24, showarrow=False, font_color="white",
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown(
            f"<div class='insight-box'>📌 Sur {total:,} machines, <b>{fails}</b> sont en défaillance, "
            f"soit <b>{fail_rate:.2f}%</b>.</div>", unsafe_allow_html=True,
        )

    with c2:
        st.subheader("🎯 Taux de panne par type de machine")
        type_stats = df_filtered.groupby("Type").agg(
            Total=("Machine failure", "size"),
            Pannes=("Machine failure", "sum"),
        ).reset_index()
        type_stats["Taux (%)"]   = (type_stats["Pannes"] / type_stats["Total"] * 100).round(2)
        type_stats["Type Label"] = type_stats["Type"].map({"L":"Low","M":"Medium","H":"High"})
        fig_type = go.Figure()
        fig_type.add_trace(go.Bar(
            x=type_stats["Type Label"], y=type_stats["Total"] - type_stats["Pannes"],
            name="Machines saines", marker_color=COLOR_OK,
            text=type_stats["Total"] - type_stats["Pannes"], textposition="inside",
        ))
        fig_type.add_trace(go.Bar(
            x=type_stats["Type Label"], y=type_stats["Pannes"],
            name="Pannes", marker_color=COLOR_FAIL,
            text=type_stats["Pannes"], textposition="outside",
        ))
        fig_type.update_layout(
            barmode="group", template=TEMPLATE,
            xaxis_title="Qualité", yaxis_title="Nombre",
            legend=dict(orientation="h", y=1.1), margin=dict(t=30, b=10),
        )
        st.plotly_chart(fig_type, use_container_width=True)
        worst = type_stats.loc[type_stats["Taux (%)"].idxmax()]
        st.markdown(
            f"<div class='insight-box'>📌 Catégorie <b>{worst['Type Label']}</b> : "
            f"taux de panne le plus élevé à <b>{worst['Taux (%)']}%</b>.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("🔥 Décomposition des modes de défaillance")
    failure_data = [
        {"Mode": f"{c} – {l}", "Code": c, "Occurrences": int(df_filtered[c].sum())}
        for c, l in FAILURE_MODES.items() if c in df_filtered.columns
    ]
    fail_df = pd.DataFrame(failure_data).sort_values("Occurrences", ascending=True)
    fig_modes = px.bar(
        fail_df, x="Occurrences", y="Mode", orientation="h",
        color="Occurrences", color_continuous_scale="Reds",
        text="Occurrences", template=TEMPLATE,
    )
    fig_modes.update_traces(textposition="outside")
    fig_modes.update_layout(coloraxis_showscale=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig_modes, use_container_width=True)
    if len(fail_df) > 0:
        top_mode = fail_df.iloc[-1]
        st.markdown(
            f"<div class='insight-box'>🔍 Mode dominant : <b>{top_mode['Mode']}</b> "
            f"avec <b>{top_mode['Occurrences']}</b> occurrences.</div>", unsafe_allow_html=True,
        )

# =====================================================================
# PAGE 2 : DISTRIBUTION & COMPORTEMENT
# =====================================================================
elif page == "📊 Distribution & Comportement":
    st.title("📊 Distribution des Valeurs Capteurs")

    var_target = st.selectbox(
        "🎯 Choisir une variable capteur",
        list(SENSOR_VARS.keys()),
        format_func=lambda x: SENSOR_VARS[x],
    )

    stats_ok = df_filtered[df_filtered["Machine failure"] == 0][var_target].describe()
    stats_ko = df_filtered[df_filtered["Machine failure"] == 1][var_target].describe()

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("📗 Moyenne (Sain)",     f"{stats_ok['mean']:.2f}")
    col_b.metric("📕 Moyenne (Panne)",    f"{stats_ko['mean']:.2f}", f"{stats_ko['mean']-stats_ok['mean']:+.2f}")
    col_c.metric("📗 Écart-type (Sain)",  f"{stats_ok['std']:.2f}")
    col_d.metric("📕 Écart-type (Panne)", f"{stats_ko['std']:.2f}")

    st.markdown("---")
    st.subheader(f"📈 Histogramme – {SENSOR_VARS[var_target]}")
    fig_hist = px.histogram(
        df_filtered, x=var_target, color="Status", nbins=50, barmode="overlay",
        color_discrete_map={"Sain": COLOR_OK, "En Panne": COLOR_FAIL},
        template=TEMPLATE, histnorm="probability density",
    )
    fig_hist.update_traces(opacity=0.7)
    fig_hist.add_vline(x=stats_ok["mean"], line_dash="dash", line_color=COLOR_OK,
                       annotation_text=f"Moy. Sain : {stats_ok['mean']:.1f}", annotation_position="top left")
    fig_hist.add_vline(x=stats_ko["mean"], line_dash="dash", line_color=COLOR_FAIL,
                       annotation_text=f"Moy. Panne : {stats_ko['mean']:.1f}", annotation_position="top right")
    fig_hist.update_layout(height=420, margin=dict(t=40, b=20), legend=dict(orientation="h", y=1.12, title=""))
    st.plotly_chart(fig_hist, use_container_width=True)

    pooled_std = np.sqrt((stats_ok["std"] ** 2 + stats_ko["std"] ** 2) / 2)
    cohens_d   = abs(stats_ko["mean"] - stats_ok["mean"]) / pooled_std if pooled_std > 0 else 0
    if cohens_d > 0.8:   level, col = "FORTE",      COLOR_FAIL
    elif cohens_d > 0.5: level, col = "MODÉRÉE",    COLOR_WARNING
    elif cohens_d > 0.2: level, col = "FAIBLE",     COLOR_PRIMARY
    else:                level, col = "TRÈS FAIBLE", COLOR_OK
    st.markdown(
        f"<div class='insight-box'>🎯 Pouvoir discriminant de <b>{SENSOR_VARS[var_target]}</b> : "
        f"<span style='color:{col};font-weight:bold'>{level}</span> (d de Cohen = {cohens_d:.2f}).</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader(f"📉 Courbe de densité (KDE) – {SENSOR_VARS[var_target]}")
    fig_kde = go.Figure()
    x_range = np.linspace(df_filtered[var_target].min(), df_filtered[var_target].max(), 200)
    data_ok = df_filtered[df_filtered["Machine failure"] == 0][var_target].values
    data_ko = df_filtered[df_filtered["Machine failure"] == 1][var_target].values
    if len(data_ok) > 1:
        kde_ok = sp_stats.gaussian_kde(data_ok)
        fig_kde.add_trace(go.Scatter(x=x_range, y=kde_ok(x_range), fill="tozeroy", name="Sain",
                                     line=dict(color=COLOR_OK, width=2), fillcolor="rgba(0,204,150,0.3)"))
    if len(data_ko) > 1:
        kde_ko = sp_stats.gaussian_kde(data_ko)
        fig_kde.add_trace(go.Scatter(x=x_range, y=kde_ko(x_range), fill="tozeroy", name="En Panne",
                                     line=dict(color=COLOR_FAIL, width=2), fillcolor="rgba(239,85,59,0.4)"))
    fig_kde.update_layout(template=TEMPLATE, height=380, margin=dict(t=40, b=20),
                          xaxis_title=SENSOR_VARS[var_target], yaxis_title="Densité estimée",
                          legend=dict(orientation="h", y=1.12, title=""))
    st.plotly_chart(fig_kde, use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📦 Box Plot Sain vs Panne")
        fig_box = px.box(df_filtered, x="Status", y=var_target, color="Status", points="outliers",
                         color_discrete_map={"Sain": COLOR_OK, "En Panne": COLOR_FAIL}, template=TEMPLATE)
        fig_box.update_layout(showlegend=False, height=400, margin=dict(t=30, b=20))
        st.plotly_chart(fig_box, use_container_width=True)
    with c2:
        st.subheader("🎻 Violin par type de machine")
        fig_violin = px.violin(df_filtered, x="Type", y=var_target, color="Status", box=True, points=False,
                               color_discrete_map={"Sain": COLOR_OK, "En Panne": COLOR_FAIL},
                               template=TEMPLATE, category_orders={"Type": ["L","M","H"]})
        fig_violin.update_layout(height=400, margin=dict(t=30, b=20), legend=dict(orientation="h", y=1.12, title=""))
        st.plotly_chart(fig_violin, use_container_width=True)

# =====================================================================
# PAGE 3 : ANALYSE MULTIVARIÉE
# =====================================================================
elif page == "🔬 Analyse Multivariée":
    st.title("🔬 Analyse Multivariée & Corrélations")

    st.subheader("🌡️ Matrice de corrélation")
    numeric_cols = list(SENSOR_VARS.keys()) + ["Machine failure"]
    corr_matrix  = df_filtered[numeric_cols].corr()
    fig_heatmap  = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                             color_continuous_scale="RdBu_r", zmin=-1, zmax=1, template=TEMPLATE)
    fig_heatmap.update_layout(height=500, margin=dict(t=50, b=20))
    st.plotly_chart(fig_heatmap, use_container_width=True)

    fail_corr = corr_matrix["Machine failure"].drop("Machine failure").abs().sort_values(ascending=False)
    st.markdown(
        f"<div class='insight-box'>📌 Variable la plus corrélée à la panne : "
        f"<b>{SENSOR_VARS.get(fail_corr.index[0], fail_corr.index[0])}</b> (|r| = {fail_corr.iloc[0]:.3f}).</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("⚠️ Carte des zones de stress mécanique")
    df_sample = df_filtered.sample(n=min(3000, len(df_filtered)), random_state=42)
    fig_scatter = px.scatter(
        df_sample, x="Rotational speed [rpm]", y="Torque [Nm]",
        color="Status", size="Tool wear [min]", size_max=15, opacity=0.6,
        color_discrete_map={"Sain": COLOR_OK, "En Panne": COLOR_FAIL},
        template=TEMPLATE, hover_data=["Type","Air temperature [K]","Process temperature [K]"],
    )
    fig_scatter.update_layout(height=550, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🌡️ Différence Process – Air")
        fig_temp = px.histogram(df_filtered, x="Temp Diff", color="Status", barmode="overlay", nbins=50,
                                color_discrete_map={"Sain": COLOR_OK, "En Panne": COLOR_FAIL}, template=TEMPLATE)
        fig_temp.update_traces(opacity=0.7)
        fig_temp.update_layout(xaxis_title="ΔT = Process – Air (K)", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_temp, use_container_width=True)
        st.caption("💡 ΔT < 8.6 K → risque HDF.")
    with c2:
        st.subheader("⚡ Puissance mécanique (W)")
        fig_pow = px.histogram(df_filtered, x="Mechanical Power", color="Status", barmode="overlay", nbins=50,
                               color_discrete_map={"Sain": COLOR_OK, "En Panne": COLOR_FAIL}, template=TEMPLATE)
        fig_pow.update_traces(opacity=0.7)
        fig_pow.update_layout(xaxis_title="P = Couple × ω (Watts)", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_pow, use_container_width=True)
        st.caption("💡 Hors [3500 W ; 9000 W] → risque PWF.")

# =====================================================================
# PAGE 4 : MODES DE DÉFAILLANCE
# =====================================================================
elif page == "⚠️ Modes de Défaillance":
    st.title("⚠️ Analyse des Modes de Défaillance")

    summary = [
        {"Code": c, "Description": l,
         "Occurrences": int(df_filtered[c].sum()),
         "Taux (%)": round(df_filtered[c].sum() / len(df_filtered) * 100, 3)}
        for c, l in FAILURE_MODES.items() if c in df_filtered.columns
    ]
    st.subheader("📋 Tableau de synthèse")
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

    st.markdown("---")
    selected_mode = st.selectbox(
        "🔍 Analyser un mode en détail",
        list(FAILURE_MODES.keys()),
        format_func=lambda x: f"{x} – {FAILURE_MODES[x]}",
    )

    if selected_mode in df_filtered.columns:
        df_mode    = df_filtered[df_filtered[selected_mode] == 1]
        df_no_mode = df_filtered[df_filtered[selected_mode] == 0]

        col_a, col_b, col_c = st.columns(3)
        col_a.metric(f"Occurrences {selected_mode}", f"{len(df_mode)}")
        col_b.metric("Taux dans le parc", f"{len(df_mode)/len(df_filtered)*100:.3f}%")
        if len(df_mode) > 0:
            col_c.metric("Type le plus touché", df_mode["Type"].value_counts().idxmax())

        comparison = []
        for var in SENSOR_VARS:
            mean_ok = df_no_mode[var].mean()
            mean_ko = df_mode[var].mean() if len(df_mode) > 0 else 0
            ecart   = ((mean_ko - mean_ok) / mean_ok * 100) if mean_ok != 0 else 0
            comparison.append({"Variable": SENSOR_VARS[var], "Sans panne": mean_ok,
                                selected_mode: mean_ko, "Écart (%)": ecart})
        comp_df = pd.DataFrame(comparison)

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(x=comp_df["Variable"], y=comp_df["Sans panne"],
                                  name="Saines", marker_color=COLOR_OK))
        fig_comp.add_trace(go.Bar(x=comp_df["Variable"], y=comp_df[selected_mode],
                                  name=f"Mode {selected_mode}", marker_color=COLOR_FAIL))
        fig_comp.update_layout(barmode="group", template=TEMPLATE,
                               legend=dict(orientation="h", y=1.1), margin=dict(t=30))
        st.plotly_chart(fig_comp, use_container_width=True)

        def color_ecart(val):
            try:
                v = abs(float(val))
            except (ValueError, TypeError):
                return ""
            if v < 1:   return "background-color:rgba(0,204,150,0.25);color:white;font-weight:600"
            elif v < 5: return "background-color:rgba(255,165,0,0.30);color:white;font-weight:600"
            else:       return "background-color:rgba(239,85,59,0.40);color:white;font-weight:600"

        st.dataframe(
            comp_df.style
                .format({"Sans panne": "{:.2f}", selected_mode: "{:.2f}", "Écart (%)": "{:+.2f}%"})
                .map(color_ecart, subset=["Écart (%)"]),
            use_container_width=True, hide_index=True,
        )

        explanations = {
            "TWF": "🔧 **TWF** : outil > 200-240 min d'usure. → Remplacer avant 180 min.",
            "HDF": "🔥 **HDF** : ΔT < 8.6 K ET RPM < 1380. → Améliorer la ventilation.",
            "PWF": "⚡ **PWF** : puissance hors [3500 W ; 9000 W]. → Ajuster couple/vitesse.",
            "OSF": "🔩 **OSF** : Couple × Usure > seuil selon type. → Limiter couple sur outils âgés.",
            "RNF": "🎲 **RNF** : panne aléatoire (~0.1%). Inévitable mais minoritaire.",
        }
        if selected_mode in explanations:
            st.markdown(f"<div class='insight-box'>{explanations[selected_mode]}</div>", unsafe_allow_html=True)

# =====================================================================
# PAGE 5 : DIAGNOSTIC PRÉDICTIF
# =====================================================================
elif page == "🤖 Diagnostic Prédictif":
    st.title("🤖 Diagnostic Prédictif par Intelligence Artificielle")

    col_input, col_viz = st.columns([1, 1.2])

    with col_input:
        st.subheader("🎚️ Paramètres capteurs")
        with st.form("prediction_form"):
            m_type    = st.selectbox("Type de machine", ["L","M","H"],
                                     format_func=lambda x: {"L":"L – Low","M":"M – Medium","H":"H – High"}[x])
            air_temp  = st.slider("🌡️ Température Air (K)",     295.0, 305.0, 300.0, 0.1)
            proc_temp = st.slider("🔥 Température Process (K)", 305.0, 315.0, 310.0, 0.1)
            speed     = st.slider("⚙️ Vitesse rotation (RPM)",  1000,  3000,  1500,  10)
            torque    = st.slider("🔩 Couple (Nm)",              0.0,   80.0,  40.0,  0.5)
            wear      = st.slider("⏱️ Usure outil (min)",        0,     300,   100,   1)
            submit    = st.form_submit_button("🚀 LANCER LE DIAGNOSTIC", use_container_width=True)

        st.markdown("##### 📊 Comparaison avec le parc")
        live_df = pd.DataFrame({
            "Variable":     ["Air T (K)","Proc T (K)","RPM","Torque","Wear"],
            "Saisi":        [air_temp, proc_temp, speed, torque, wear],
            "Moyenne parc": [
                df["Air temperature [K]"].mean(),
                df["Process temperature [K]"].mean(),
                df["Rotational speed [rpm]"].mean(),
                df["Torque [Nm]"].mean(),
                df["Tool wear [min]"].mean(),
            ],
        })
        st.dataframe(live_df.style.format({"Saisi":"{:.1f}","Moyenne parc":"{:.1f}"}),
                     hide_index=True, use_container_width=True)

    with col_viz:
        if submit:
            type_encoded = encoder.transform([m_type])[0]
            input_data   = np.array([[type_encoded, air_temp, proc_temp, speed, torque, wear]])
            prediction   = model.predict(input_data)[0]
            try:
                prob = model.predict_proba(input_data)[0][1]
            except Exception:
                prob = float(prediction)

            st.subheader("📍 Verdict du diagnostic")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                number={"suffix":"%","font":{"size":42}},
                delta={"reference":50,"increasing":{"color":COLOR_FAIL},"decreasing":{"color":COLOR_OK}},
                title={"text":"Probabilité de défaillance","font":{"size":18,"color":"white"}},
                gauge={
                    "axis":    {"range":[0,100],"tickwidth":1,"tickcolor":"white"},
                    "bar":     {"color":"#FFFFFF","thickness":0.25},
                    "bgcolor": "rgba(0,0,0,0.3)",
                    "steps":   [
                        {"range":[0,30],  "color":COLOR_OK},
                        {"range":[30,60], "color":COLOR_WARNING},
                        {"range":[60,100],"color":COLOR_FAIL},
                    ],
                    "threshold":{"line":{"color":"yellow","width":4},"thickness":0.8,"value":50},
                },
            ))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color":"white"},
                                    height=320, margin=dict(t=50,b=20,l=20,r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            if prob >= 0.6:
                st.markdown(f"<div class='alert-danger'>🚨 <b>RISQUE ÉLEVÉ</b> – "
                            f"Probabilité : <b>{prob:.1%}</b> – Arrêt préventif recommandé.</div>",
                            unsafe_allow_html=True)
            elif prob >= 0.3:
                st.markdown(f"<div class='insight-box'>⚠️ <b>RISQUE MODÉRÉ</b> – "
                            f"Probabilité : <b>{prob:.1%}</b> – Surveillance renforcée.</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='alert-success'>✅ <b>MACHINE OPÉRATIONNELLE</b> – "
                            f"Probabilité de panne : <b>{prob:.1%}</b></div>",
                            unsafe_allow_html=True)

            st.markdown("##### 🔬 Analyse des facteurs de risque")
            delta_t = proc_temp - air_temp
            power   = torque * speed * 2 * np.pi / 60
            osf_thr = {"L":11000,"M":12000,"H":13000}
            if power < 3500:   pwf_risk = (3500 - power) / 3500 * 100
            elif power > 9000: pwf_risk = (power - 9000) / 9000 * 100
            else:              pwf_risk = 0

            risk_factors = [
                ("Usure outil (TWF)",        min(wear / 240 * 100, 100),           f"{wear} min / 240 min"),
                ("Dissipation chaleur (HDF)", max(0,(8.6-delta_t)/8.6*100) if speed < 1380 else 0, f"ΔT={delta_t:.1f}K, RPM={speed}"),
                ("Surcharge énergie (PWF)",  min(pwf_risk, 100),                   f"P={power:.0f}W (plage 3500-9000W)"),
                ("Sur-contrainte (OSF)",     min(torque*wear/osf_thr[m_type]*100,100), f"Couple×Usure={torque*wear:.0f}/{osf_thr[m_type]}"),
            ]
            risk_df = pd.DataFrame(risk_factors, columns=["Facteur","Risque (%)","Détail"])
            risk_df = risk_df.sort_values("Risque (%)", ascending=True)

            fig_risk = px.bar(
                risk_df, x="Risque (%)", y="Facteur", orientation="h",
                text="Risque (%)", color="Risque (%)",
                color_continuous_scale="RdYlGn_r", range_color=[0,100],
                hover_data=["Détail"], template=TEMPLATE,
            )
            fig_risk.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
            fig_risk.update_layout(xaxis_range=[0,115], coloraxis_showscale=False,
                                   height=300, margin=dict(t=20,b=20))
            st.plotly_chart(fig_risk, use_container_width=True)

            top_risk = risk_df.iloc[-1]
            if top_risk["Risque (%)"] > 70:
                st.markdown(
                    f"<div class='alert-danger'>🎯 Facteur critique : <b>{top_risk['Facteur']}</b> "
                    f"à <b>{top_risk['Risque (%)']:.0f}%</b><br>📌 {top_risk['Détail']}</div>",
                    unsafe_allow_html=True)
        else:
            st.info("👈 Saisissez les paramètres et cliquez sur **LANCER LE DIAGNOSTIC**.")
            st.markdown("""
            ##### 📐 Plages opérationnelles typiques
            - 🌡️ **Température Air** : 295 – 305 K
            - 🔥 **Température Process** : 305 – 315 K
            - ⚙️ **Vitesse rotation** : 1300 – 2900 RPM
            - 🔩 **Couple** : 3 – 77 Nm
            - ⏱️ **Usure outil** : remplacer avant **200 min**
            """)

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.caption("🏭 Industrial AI Insights – Maintenance Prédictive | Dataset AI4I 2020")
