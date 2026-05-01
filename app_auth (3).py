# =====================================================================
# 🏭 INDUSTRIAL AI INSIGHTS – Dashboard de Maintenance Prédictive
# Dataset : AI4I 2020 Predictive Maintenance Dataset
# Auteur  : Version améliorée par Claude
# =====================================================================

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# =====================================================================
# 1. CONFIGURATION DE LA PAGE & STYLE
# =====================================================================
st.set_page_config(
    page_title="Industrial AI Insights – Maintenance Prédictive",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Palette professionnelle cohérente
COLOR_OK       = "#00CC96"   # Vert : machine saine
COLOR_FAIL     = "#EF553B"   # Rouge : panne
COLOR_WARNING  = "#FFA500"   # Orange : alerte
COLOR_PRIMARY  = "#00B4D8"   # Bleu : primaire
COLOR_ACCENT   = "#7B2CBF"   # Violet : accent
TEMPLATE       = "plotly_dark"

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 28px; color: #00B4D8 !important; font-weight: 700;}
[data-testid="stMetricLabel"] { font-size: 14px; color: #CCCCCC !important;}
[data-testid="stMetricDelta"] { font-size: 13px; }
.stMetric {
    background-color: rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    padding: 18px;
    border: 1px solid #2A2A3E;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.stSelectbox label, .stNumberInput label, .stSlider label {
    color: #00B4D8 !important;
    font-weight: 600;
}
h1 { color: #FFFFFF; border-bottom: 3px solid #00B4D8; padding-bottom: 10px;}
h2 { color: #00B4D8; }
h3 { color: #FFFFFF; }
.insight-box {
    background: linear-gradient(135deg, rgba(0,180,216,0.1) 0%, rgba(123,44,191,0.1) 100%);
    border-left: 4px solid #00B4D8;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}
.alert-danger {
    background: linear-gradient(135deg, rgba(239,85,59,0.15) 0%, rgba(239,85,59,0.05) 100%);
    border-left: 4px solid #EF553B;
    padding: 15px;
    border-radius: 8px;
}
.alert-success {
    background: linear-gradient(135deg, rgba(0,204,150,0.15) 0%, rgba(0,204,150,0.05) 100%);
    border-left: 4px solid #00CC96;
    padding: 15px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. AUTHENTIFICATION
# =====================================================================

# Dictionnaire des utilisateurs : { "username": { "password": "...", "role": "..." } }
USERS = {
    "admin":     {"password": "admin123",    "role": "Administrateur"},
    "operateur": {"password": "machine456",  "role": "Opérateur"},
    "analyste":  {"password": "analyse789",  "role": "Analyste"},
}

def login_page():
    """Affiche la page de connexion et gère l'authentification."""
    st.markdown("""
    <div style="display:flex; justify-content:center; margin-top: 60px;">
        <div style="background: rgba(255,255,255,0.04); border: 1px solid #2A2A3E;
                    border-radius: 16px; padding: 40px 50px; width: 100%; max-width: 420px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.4);">
            <div style="text-align:center; margin-bottom: 30px;">
                <div style="font-size: 56px;">🏭</div>
                <h2 style="color:#FFFFFF; margin: 8px 0 4px 0; border:none;">Industrial AI Insights</h2>
                <p style="color:#888; font-size:13px; margin:0;">Maintenance Prédictive Industrielle</p>
            </div>
    """, unsafe_allow_html=True)

    col_l, col_mid, col_r = st.columns([1, 3, 1])
    with col_mid:
        username = st.text_input("👤 Nom d'utilisateur", placeholder="Entrez votre identifiant")
        password = st.text_input("🔒 Mot de passe", type="password", placeholder="Entrez votre mot de passe")
        login_btn = st.button("🔐  Se connecter", use_container_width=True)

        if login_btn:
            user = USERS.get(username)
            if user and user["password"] == password:
                st.session_state["logged_in"] = True
                st.session_state["username"]  = username
                st.session_state["role"]      = user["role"]
                st.rerun()
            else:
                st.error("❌ Identifiant ou mot de passe incorrect.")

    st.markdown("</div></div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#555; font-size:12px; margin-top:30px;'>"
        "Accès restreint – Personnel autorisé uniquement</p>",
        unsafe_allow_html=True
    )

# --- Vérification de la session ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login_page()
    st.stop()

# =====================================================================
# 3. CHARGEMENT DES RESSOURCES
# =====================================================================
@st.cache_resource
def load_resources():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(current_dir, 'modele_maintenance_predictive.pkl'))
    encoder = joblib.load(os.path.join(current_dir, 'label_encoder_type.pkl'))
    df = pd.read_csv(os.path.join(current_dir, 'ai4i2020.csv'))
    return model, encoder, df

try:
    model, encoder, df = load_resources()
except Exception as e:
    st.error(f"❌ Erreur de chargement des ressources : {e}")
    st.stop()

# Colonnes des modes de panne (dataset AI4I 2020)
FAILURE_MODES = {
    'TWF': 'Usure de l\'outil',
    'HDF': 'Dissipation de chaleur',
    'PWF': 'Surcharge énergétique',
    'OSF': 'Sur-contrainte (couple)',
    'RNF': 'Panne aléatoire'
}
SENSOR_VARS = {
    'Air temperature [K]': 'Température Air (K)',
    'Process temperature [K]': 'Température Process (K)',
    'Rotational speed [rpm]': 'Vitesse Rotation (RPM)',
    'Torque [Nm]': 'Couple (Nm)',
    'Tool wear [min]': 'Usure Outil (min)'
}

# Pré-calcul utile
df['Status'] = df['Machine failure'].map({0: 'Sain', 1: 'En Panne'})
if 'Process temperature [K]' in df.columns and 'Air temperature [K]' in df.columns:
    df['Temp Diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['Mechanical Power'] = (df['Torque [Nm]'] * df['Rotational speed [rpm]'] * 2 * np.pi / 60)  # Puissance en Watts

# =====================================================================
# 3. BARRE LATÉRALE – NAVIGATION & FILTRES
# =====================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1067/1067357.png", width=70)
    st.title("Fleet Manager AI")
    st.caption("Maintenance Prédictive Industrielle")
    st.markdown("---")

    # Infos utilisateur connecté + bouton déconnexion
    st.markdown(
        f"<div style='background:rgba(0,180,216,0.08); border:1px solid #2A2A3E; "
        f"border-radius:10px; padding:12px; margin-bottom:10px;'>"
        f"<span style='color:#888; font-size:12px;'>Connecté en tant que</span><br>"
        f"<b style='color:#00B4D8;'>👤 {st.session_state['username']}</b><br>"
        f"<span style='color:#AAA; font-size:12px;'>🏷️ {st.session_state['role']}</span>"
        f"</div>",
        unsafe_allow_html=True
    )
    if st.button("🚪 Se déconnecter", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.markdown("---")

    page = st.radio(
        "📌 **Navigation**",
        ["🏠 Vue d'Ensemble",
         "📊 Distribution & Comportement",
         "🔬 Analyse Multivariée",
         "⚠️ Modes de Défaillance",
         "🤖 Diagnostic Prédictif"],
        label_visibility="visible"
    )

    st.markdown("---")
    st.markdown("### 🎛️ Filtres globaux")
    selected_types = st.multiselect(
        "Type de machine",
        options=df['Type'].unique().tolist(),
        default=df['Type'].unique().tolist(),
        help="L = Low, M = Medium, H = High quality"
    )
    df_filtered = df[df['Type'].isin(selected_types)] if selected_types else df

    st.markdown("---")
    st.caption(f"📁 **Dataset :** {len(df):,} observations")
    st.caption(f"🔍 **Filtré :** {len(df_filtered):,} observations")
    st.caption(f"⚙️ **Modèle :** {type(model).__name__}")

# =====================================================================
# PAGE 1 : VUE D'ENSEMBLE (KPIs & SYNTHÈSE)
# =====================================================================
if page == "🏠 Vue d'Ensemble":
    st.title("🏭 Vue d'Ensemble du Parc Machines")
    st.markdown("**Objectif :** Synthèse des indicateurs critiques de santé du parc industriel.")

    # --- KPIs PRINCIPAUX ---
    total = len(df_filtered)
    fails = int(df_filtered['Machine failure'].sum())
    fail_rate = (fails / total * 100) if total > 0 else 0
    avg_wear = df_filtered['Tool wear [min]'].mean()
    avg_torque = df_filtered['Torque [Nm]'].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🏭 Parc total", f"{total:,}", "machines surveillées")
    col2.metric("🚨 Pannes", f"{fails:,}",
                f"{fail_rate:.2f} % du parc",
                delta_color="inverse")
    col3.metric("✅ Disponibilité", f"{100-fail_rate:.2f} %",
                "Taux opérationnel")
    col4.metric("🔧 Usure moy.", f"{avg_wear:.0f} min",
                "Tool wear moyen")
    col5.metric("⚙️ Couple moy.", f"{avg_torque:.1f} Nm",
                "Torque moyen")

    st.markdown("---")

    # --- LIGNE 1 : Répartition pannes & types ---
    c1, c2 = st.columns([1, 1.3])

    with c1:
        st.subheader("⚖️ Répartition Sain / Panne")
        status_counts = df_filtered['Status'].value_counts().reset_index()
        status_counts.columns = ['Statut', 'Nombre']
        fig_donut = px.pie(
            status_counts, values='Nombre', names='Statut',
            hole=0.55,
            color='Statut',
            color_discrete_map={'Sain': COLOR_OK, 'En Panne': COLOR_FAIL},
            template=TEMPLATE
        )
        fig_donut.update_traces(
            textposition='outside',
            textinfo='label+percent',
            marker=dict(line=dict(color='#000', width=2))
        )
        fig_donut.update_layout(
            showlegend=True,
            legend=dict(orientation='h', y=-0.1),
            margin=dict(t=10, b=10, l=10, r=10),
            annotations=[dict(
                text=f"{fail_rate:.1f}%<br><span style='font-size:12px'>panne</span>",
                x=0.5, y=0.5, font_size=24, showarrow=False, font_color='white'
            )]
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown(
            f"<div class='insight-box'>📌 Sur {total:,} machines, "
            f"<b>{fails}</b> sont en défaillance, soit <b>{fail_rate:.2f}%</b>. "
            f"Un taux > 5% signale un parc nécessitant une maintenance corrective globale.</div>",
            unsafe_allow_html=True
        )

    with c2:
        st.subheader("🎯 Taux de panne par type de machine")
        type_stats = df_filtered.groupby('Type').agg(
            Total=('Machine failure', 'size'),
            Pannes=('Machine failure', 'sum')
        ).reset_index()
        type_stats['Taux (%)'] = (type_stats['Pannes'] / type_stats['Total'] * 100).round(2)
        type_stats['Type Label'] = type_stats['Type'].map({'L': 'Low', 'M': 'Medium', 'H': 'High'})

        fig_type = go.Figure()
        fig_type.add_trace(go.Bar(
            x=type_stats['Type Label'], y=type_stats['Total'],
            name='Machines saines', marker_color=COLOR_OK,
            text=type_stats['Total']-type_stats['Pannes'], textposition='inside'
        ))
        fig_type.add_trace(go.Bar(
            x=type_stats['Type Label'], y=type_stats['Pannes'],
            name='Pannes', marker_color=COLOR_FAIL,
            text=type_stats['Pannes'], textposition='outside'
        ))
        fig_type.update_layout(
            barmode='group', template=TEMPLATE,
            xaxis_title='Qualité de la machine',
            yaxis_title='Nombre de machines',
            legend=dict(orientation='h', y=1.1),
            margin=dict(t=30, b=10)
        )
        st.plotly_chart(fig_type, use_container_width=True)

        worst = type_stats.loc[type_stats['Taux (%)'].idxmax()]
        st.markdown(
            f"<div class='insight-box'>📌 La catégorie <b>{worst['Type Label']}</b> "
            f"présente le taux de panne le plus élevé : <b>{worst['Taux (%)']}%</b>. "
            f"À surveiller en priorité.</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # --- LIGNE 2 : Modes de panne ---
    st.subheader("🔥 Décomposition des modes de défaillance")
    failure_data = []
    for code, label in FAILURE_MODES.items():
        if code in df_filtered.columns:
            failure_data.append({
                'Mode': f"{code} – {label}",
                'Code': code,
                'Occurrences': int(df_filtered[code].sum())
            })
    fail_df = pd.DataFrame(failure_data).sort_values('Occurrences', ascending=True)

    fig_modes = px.bar(
        fail_df, x='Occurrences', y='Mode', orientation='h',
        color='Occurrences', color_continuous_scale='Reds',
        text='Occurrences', template=TEMPLATE
    )
    fig_modes.update_traces(textposition='outside')
    fig_modes.update_layout(
        xaxis_title='Nombre d\'occurrences',
        yaxis_title='',
        coloraxis_showscale=False,
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_modes, use_container_width=True)

    if len(fail_df) > 0:
        top_mode = fail_df.iloc[-1]
        st.markdown(
            f"<div class='insight-box'>🔍 <b>Interprétation :</b> Le mode dominant est "
            f"<b>{top_mode['Mode']}</b> avec <b>{top_mode['Occurrences']}</b> occurrences. "
            f"Une action de maintenance ciblée sur ce mode aurait l'impact le plus fort sur la disponibilité du parc.</div>",
            unsafe_allow_html=True
        )

# =====================================================================
# PAGE 2 : DISTRIBUTION & COMPORTEMENT DES VARIABLES
# =====================================================================
elif page == "📊 Distribution & Comportement":
    st.title("📊 Distribution des Valeurs Capteurs")
    st.markdown("**Objectif :** Visualiser **la répartition statistique** de chaque variable capteur "
                "et comparer les profils Sain vs Panne. Plus les courbes sont décalées, plus la variable "
                "est un bon indicateur de défaillance.")

    var_target = st.selectbox(
        "🎯 Choisir une variable capteur à analyser",
        list(SENSOR_VARS.keys()),
        format_func=lambda x: SENSOR_VARS[x]
    )

    # Statistiques comparatives
    stats_ok = df_filtered[df_filtered['Machine failure']==0][var_target].describe()
    stats_ko = df_filtered[df_filtered['Machine failure']==1][var_target].describe()

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("📗 Moyenne (Sain)", f"{stats_ok['mean']:.2f}")
    col_b.metric("📕 Moyenne (Panne)", f"{stats_ko['mean']:.2f}",
                 f"{stats_ko['mean']-stats_ok['mean']:+.2f}")
    col_c.metric("📗 Écart-type (Sain)", f"{stats_ok['std']:.2f}")
    col_d.metric("📕 Écart-type (Panne)", f"{stats_ko['std']:.2f}")

    st.markdown("---")

    # === HISTOGRAMME COMPARATIF (pleine largeur) ===
    st.subheader(f"📈 Histogramme de distribution – {SENSOR_VARS[var_target]}")
    fig_hist = px.histogram(
        df_filtered, x=var_target, color="Status",
        nbins=50,
        barmode="overlay",
        color_discrete_map={'Sain': COLOR_OK, 'En Panne': COLOR_FAIL},
        template=TEMPLATE,
        histnorm='probability density'
    )
    fig_hist.update_traces(opacity=0.7)
    fig_hist.update_layout(
        xaxis_title=SENSOR_VARS[var_target],
        yaxis_title='Densité de probabilité',
        legend=dict(orientation='h', y=1.12, title=''),
        margin=dict(t=40, b=20),
        height=420
    )
    fig_hist.add_vline(x=stats_ok['mean'], line_dash='dash', line_color=COLOR_OK, line_width=2,
                       annotation_text=f"Moyenne Sain : {stats_ok['mean']:.1f}",
                       annotation_position="top left")
    fig_hist.add_vline(x=stats_ko['mean'], line_dash='dash', line_color=COLOR_FAIL, line_width=2,
                       annotation_text=f"Moyenne Panne : {stats_ko['mean']:.1f}",
                       annotation_position="top right")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Indice de discrimination (Cohen's d)
    pooled_std = np.sqrt((stats_ok['std']**2 + stats_ko['std']**2) / 2)
    cohens_d = abs(stats_ko['mean'] - stats_ok['mean']) / pooled_std if pooled_std > 0 else 0
    if cohens_d > 0.8:
        level, color = "FORTE", COLOR_FAIL
    elif cohens_d > 0.5:
        level, color = "MODÉRÉE", COLOR_WARNING
    elif cohens_d > 0.2:
        level, color = "FAIBLE", COLOR_PRIMARY
    else:
        level, color = "TRÈS FAIBLE", COLOR_OK

    st.markdown(
        f"<div class='insight-box'>🎯 <b>Pouvoir discriminant</b> de "
        f"<b>{SENSOR_VARS[var_target]}</b> : <span style='color:{color};font-weight:bold'>{level}</span> "
        f"(d de Cohen = {cohens_d:.2f}).<br>"
        f"➡️ Plus la valeur est élevée (>0.8), plus cette variable seule permet de détecter une panne.</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # === COURBE DE DENSITÉ (KDE) ===
    st.subheader(f"📉 Courbe de densité (KDE) – {SENSOR_VARS[var_target]}")
    st.caption("Vue lissée de la distribution : facilite l'identification des modes et des zones de risque.")

    from scipy import stats as sp_stats
    fig_kde = go.Figure()

    x_min, x_max = df_filtered[var_target].min(), df_filtered[var_target].max()
    x_range = np.linspace(x_min, x_max, 200)

    # KDE pour machines saines
    data_ok = df_filtered[df_filtered['Machine failure']==0][var_target].values
    if len(data_ok) > 1:
        kde_ok = sp_stats.gaussian_kde(data_ok)
        y_ok = kde_ok(x_range)
        fig_kde.add_trace(go.Scatter(
            x=x_range, y=y_ok, fill='tozeroy', name='Sain',
            line=dict(color=COLOR_OK, width=2),
            fillcolor='rgba(0, 204, 150, 0.3)'
        ))

        st.plotly_chart(fig_violin, use_container_width=True)
        st.caption("📌 Distribution détaillée pour chaque qualité de machine.")

# =====================================================================
# PAGE 3 : ANALYSE MULTIVARIÉE (CORRÉLATIONS)
# =====================================================================
elif page == "🔬 Analyse Multivariée":
    st.title("🔬 Analyse Multivariée & Corrélations")
    st.markdown("**Objectif :** Comprendre les **relations entre variables** et identifier "
                "les zones à risque où plusieurs facteurs combinés provoquent les pannes.")

    # --- HEATMAP DE CORRÉLATION ---
    st.subheader("🌡️ Matrice de corrélation")
    numeric_cols = list(SENSOR_VARS.keys()) + ['Machine failure']
    corr_matrix = df_filtered[numeric_cols].corr()

    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        template=TEMPLATE
    )
    fig_heatmap.update_layout(
        title="Coefficient de corrélation de Pearson",
        margin=dict(t=50, b=20),
        height=500
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Insight automatique sur les corrélations avec la panne
    fail_corr = corr_matrix['Machine failure'].drop('Machine failure').abs().sort_values(ascending=False)
    top_corr = fail_corr.index[0]
    st.markdown(
        f"<div class='insight-box'>📌 La variable la plus corrélée à la panne est "
        f"<b>{SENSOR_VARS.get(top_corr, top_corr)}</b> "
        f"(|r| = {fail_corr.iloc[0]:.3f}). "
        f"Une forte corrélation Process/Air Temp est normale (chaleur mécanique transmise).</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # --- SCATTER : COUPLE vs VITESSE (zone critique) ---
    st.subheader("⚠️ Carte des zones de stress mécanique")
    st.caption("Les pannes apparaissent souvent dans les **zones extrêmes** : très haute vitesse + bas couple, OU basse vitesse + haut couple.")

    sample_size = min(3000, len(df_filtered))
    df_sample = df_filtered.sample(n=sample_size, random_state=42)

    fig_scatter = px.scatter(
        df_sample, x='Rotational speed [rpm]', y='Torque [Nm]',
        color='Status',
        size='Tool wear [min]', size_max=15,
        opacity=0.6,
        color_discrete_map={'Sain': COLOR_OK, 'En Panne': COLOR_FAIL},
        template=TEMPLATE,
        hover_data=['Type', 'Air temperature [K]', 'Process temperature [K]']
    )
    fig_scatter.update_layout(
        xaxis_title='Vitesse de rotation (RPM)',
        yaxis_title='Couple (Nm)',
        legend=dict(orientation='h', y=1.1),
        height=550
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown(
        "<div class='insight-box'>🔍 <b>Lecture :</b> chaque point = une machine. "
        "La taille reflète l'usure de l'outil. Les <span style='color:#EF553B'>points rouges</span> "
        "groupés délimitent les <b>zones de défaillance</b> à éviter en exploitation.</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # --- DELTA TEMPÉRATURE & PUISSANCE ---
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🌡️ Différence Process – Air")
        fig_temp = px.histogram(
            df_filtered, x='Temp Diff', color='Status',
            barmode='overlay', nbins=50,
            color_discrete_map={'Sain': COLOR_OK, 'En Panne': COLOR_FAIL},
            template=TEMPLATE
        )
        fig_temp.update_traces(opacity=0.7)
        fig_temp.update_layout(
            xaxis_title='ΔT = Process – Air (K)',
            yaxis_title='Fréquence',
            legend=dict(orientation='h', y=1.1)
        )
        st.plotly_chart(fig_temp, use_container_width=True)
        st.caption("💡 Une faible ΔT (<8.6 K) déclenche le mode HDF (Heat Dissipation Failure).")

    with c2:
        st.subheader("⚡ Puissance mécanique (W)")
        fig_pow = px.histogram(
            df_filtered, x='Mechanical Power', color='Status',
            barmode='overlay', nbins=50,
            color_discrete_map={'Sain': COLOR_OK, 'En Panne': COLOR_FAIL},
            template=TEMPLATE
        )
        fig_pow.update_traces(opacity=0.7)
        fig_pow.update_layout(
            xaxis_title='P = Couple × ω (Watts)',
            yaxis_title='Fréquence',
            legend=dict(orientation='h', y=1.1)
        )
        st.plotly_chart(fig_pow, use_container_width=True)
        st.caption("💡 Hors plage [3500 W ; 9000 W] → mode PWF (Power Failure).")

# =====================================================================
# PAGE 4 : MODES DE DÉFAILLANCE
# =====================================================================
elif page == "⚠️ Modes de Défaillance":
    st.title("⚠️ Analyse des Modes de Défaillance")
    st.markdown("**Objectif :** Décortiquer **chaque type de panne** et identifier "
                "les variables à l'origine de chacun pour orienter la maintenance.")

    # Tableau récapitulatif
    summary = []
    for code, label in FAILURE_MODES.items():
        if code in df_filtered.columns:
            cnt = int(df_filtered[code].sum())
            pct = (cnt / len(df_filtered) * 100)
            summary.append({'Code': code, 'Description': label,
                            'Occurrences': cnt, 'Taux (%)': round(pct, 3)})
    sum_df = pd.DataFrame(summary)

    st.subheader("📋 Tableau de synthèse des modes")
    st.dataframe(sum_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Choix d'un mode pour analyse profonde
    st.subheader("🔍 Analyse approfondie d'un mode")
    selected_mode = st.selectbox(
        "Choisir un mode de défaillance",
        list(FAILURE_MODES.keys()),
        format_func=lambda x: f"{x} – {FAILURE_MODES[x]}"
    )

    if selected_mode in df_filtered.columns:
        df_mode = df_filtered[df_filtered[selected_mode] == 1]
        df_no_mode = df_filtered[df_filtered[selected_mode] == 0]

        col_a, col_b, col_c = st.columns(3)
        col_a.metric(f"Occurrences {selected_mode}", f"{len(df_mode)}")
        col_b.metric("Taux dans le parc", f"{len(df_mode)/len(df_filtered)*100:.3f}%")
        # Type le plus touché
        if len(df_mode) > 0:
            top_type = df_mode['Type'].value_counts().idxmax()
            col_c.metric("Type le plus touché", top_type)

        # Comparaison des moyennes
        st.markdown(f"#### 📊 Profil moyen des machines en mode **{selected_mode}**")
        comparison = []
        for var in SENSOR_VARS.keys():
            comparison.append({
                'Variable': SENSOR_VARS[var],
                'Sans panne': df_no_mode[var].mean(),
                f'{selected_mode}': df_mode[var].mean() if len(df_mode) > 0 else 0,
                'Écart (%)': ((df_mode[var].mean() - df_no_mode[var].mean()) / df_no_mode[var].mean() * 100) if len(df_mode) > 0 and df_no_mode[var].mean() != 0 else 0
            })
        comp_df = pd.DataFrame(comparison)

        # Bar chart comparatif
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=comp_df['Variable'], y=comp_df['Sans panne'],
            name='Machines saines', marker_color=COLOR_OK
        ))
        fig_comp.add_trace(go.Bar(
            x=comp_df['Variable'], y=comp_df[selected_mode],
            name=f'Mode {selected_mode}', marker_color=COLOR_FAIL
        ))
        fig_comp.update_layout(
            barmode='group', template=TEMPLATE,
            yaxis_title='Valeur moyenne (échelle variable)',
            legend=dict(orientation='h', y=1.1),
            margin=dict(t=30)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Tableau d'écart avec coloration native (sans matplotlib)
        def color_ecart(val):
            """Colore la cellule selon l'écart en % – sans matplotlib."""
            try:
                v = float(val)
            except (ValueError, TypeError):
                return ''
            abs_v = abs(v)
            if abs_v < 1:
                bg = 'rgba(0, 204, 150, 0.25)'      # vert
            elif abs_v < 5:
                bg = 'rgba(255, 165, 0, 0.30)'      # orange
            else:
                bg = 'rgba(239, 85, 59, 0.40)'      # rouge
            return f'background-color: {bg}; color: white; font-weight: 600;'

        st.dataframe(
            comp_df.style.format({
                'Sans panne': '{:.2f}',
                selected_mode: '{:.2f}',
                'Écart (%)': '{:+.2f}%'
            }).map(color_ecart, subset=['Écart (%)']),
            use_container_width=True, hide_index=True
        )

        # Explications métier
        explanations = {
            'TWF': "🔧 **TWF (Tool Wear Failure)** : déclenché lorsque l'outil dépasse 200-240 min d'usure. "
                   "Action : remplacer l'outil avant 180 min en préventif.",
            'HDF': "🔥 **HDF (Heat Dissipation Failure)** : si ΔT (Process-Air) < 8.6 K ET vitesse < 1380 RPM. "
                   "Action : améliorer la ventilation ou augmenter la vitesse de rotation.",
            'PWF': "⚡ **PWF (Power Failure)** : puissance mécanique hors de [3500 W ; 9000 W]. "
                   "Action : ajuster le couple/vitesse pour rester dans la plage nominale.",
            'OSF': "🔩 **OSF (Overstrain Failure)** : produit Couple × Usure outil > seuil (selon type L/M/H). "
                   "Action : limiter le couple sur outils âgés.",
            'RNF': "🎲 **RNF (Random Failure)** : panne aléatoire (~0.1% du parc). Inévitable mais minoritaire."
        }
        if selected_mode in explanations:
            st.markdown(f"<div class='insight-box'>{explanations[selected_mode]}</div>", unsafe_allow_html=True)

# =====================================================================
# PAGE 5 : DIAGNOSTIC PRÉDICTIF (IA)
# =====================================================================
elif page == "🤖 Diagnostic Prédictif":
    st.title("🤖 Diagnostic Prédictif par Intelligence Artificielle")
    st.markdown("**Objectif :** Saisir les paramètres capteurs d'une machine et obtenir "
                "instantanément une **probabilité de défaillance** avec recommandations.")

    col_input, col_viz = st.columns([1, 1.2])

    with col_input:
        st.subheader("🎚️ Paramètres capteurs")
        with st.form("prediction_form"):
            m_type = st.selectbox(
                "Type de machine",
                ["L", "M", "H"],
                format_func=lambda x: {'L':'L – Low quality', 'M':'M – Medium', 'H':'H – High quality'}[x]
            )
            air_temp = st.slider("🌡️ Température Air (K)", 295.0, 305.0, 300.0, 0.1)
            proc_temp = st.slider("🔥 Température Process (K)", 305.0, 315.0, 310.0, 0.1)
            speed = st.slider("⚙️ Vitesse rotation (RPM)", 1000, 3000, 1500, 10)
            torque = st.slider("🔩 Couple (Nm)", 0.0, 80.0, 40.0, 0.5)
            wear = st.slider("⏱️ Usure outil (min)", 0, 300, 100, 1)
            submit = st.form_submit_button("🚀 LANCER LE DIAGNOSTIC", use_container_width=True)

        # Comparaison live avec la moyenne du parc
        st.markdown("##### 📊 Comparaison avec le parc")
        live_df = pd.DataFrame({
            'Variable': ['Air T (K)', 'Proc T (K)', 'RPM', 'Torque', 'Wear'],
            'Saisi': [air_temp, proc_temp, speed, torque, wear],
            'Moyenne parc': [
                df['Air temperature [K]'].mean(),
                df['Process temperature [K]'].mean(),
                df['Rotational speed [rpm]'].mean(),
                df['Torque [Nm]'].mean(),
                df['Tool wear [min]'].mean()
            ]
        })
        st.dataframe(live_df.style.format({'Saisi':'{:.1f}','Moyenne parc':'{:.1f}'}),
                     hide_index=True, use_container_width=True)

    with col_viz:
        if submit:
            type_encoded = encoder.transform([m_type])[0]
            input_data = np.array([[type_encoded, air_temp, proc_temp, speed, torque, wear]])
            prediction = model.predict(input_data)[0]
            try:
                prob = model.predict_proba(input_data)[0][1]
            except Exception:
                prob = float(prediction)

            st.subheader("📍 Verdict du diagnostic")

            # JAUGE
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                number={'suffix': "%", 'font': {'size': 42}},
                delta={'reference': 50, 'increasing': {'color': COLOR_FAIL},
                       'decreasing': {'color': COLOR_OK}},
                title={'text': "Probabilité de défaillance",
                       'font': {'size': 18, 'color': 'white'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#FFFFFF", 'thickness': 0.25},
                    'bgcolor': "rgba(0,0,0,0.3)",
                    'borderwidth': 2,
                    'bordercolor': "#444",
                    'steps': [
                        {'range': [0, 30],   'color': COLOR_OK},
                        {'range': [30, 60],  'color': COLOR_WARNING},
                        {'range': [60, 100], 'color': COLOR_FAIL}
                    ],
                    'threshold': {
                        'line': {'color': "yellow", 'width': 4},
                        'thickness': 0.8, 'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': "white"},
                height=320,
                margin=dict(t=50, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Verdict
            if prob >= 0.6:
                st.markdown(
                    f"<div class='alert-danger'>🚨 <b>RISQUE ÉLEVÉ – Arrêt préventif recommandé</b><br>"
                    f"Probabilité : <b>{prob:.1%}</b><br>"
                    f"Action : programmer immédiatement une intervention de maintenance.</div>",
                    unsafe_allow_html=True
                )
            elif prob >= 0.3:
                st.markdown(
                    f"<div class='insight-box'>⚠️ <b>RISQUE MODÉRÉ – Surveillance renforcée</b><br>"
                    f"Probabilité : <b>{prob:.1%}</b><br>"
                    f"Action : augmenter la fréquence de monitoring, planifier une inspection.</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='alert-success'>✅ <b>MACHINE OPÉRATIONNELLE</b><br>"
                    f"Probabilité de panne : <b>{prob:.1%}</b><br>"
                    f"Action : maintenir le rythme de surveillance standard.</div>",
                    unsafe_allow_html=True
                )

            # ANALYSE DES MODES DE PANNE PROBABLES
            st.markdown("##### 🔬 Analyse des facteurs de risque")
            risk_factors = []

            # TWF
            wear_risk = min(wear / 240 * 100, 100)
            risk_factors.append(('Usure outil (TWF)', wear_risk,
                                 f"{wear} min / 240 min seuil"))
            # HDF
            delta_t = proc_temp - air_temp
            hdf_risk = max(0, (8.6 - delta_t) / 8.6 * 100) if speed < 1380 else 0
            risk_factors.append(('Dissipation chaleur (HDF)', hdf_risk,
                                 f"ΔT={delta_t:.1f}K, RPM={speed}"))
            # PWF
            power = torque * speed * 2 * np.pi / 60
            if power < 3500:
                pwf_risk = (3500 - power) / 3500 * 100
            elif power > 9000:
                pwf_risk = (power - 9000) / 9000 * 100
            else:
                pwf_risk = 0
            pwf_risk = min(pwf_risk, 100)
            risk_factors.append(('Surcharge énergie (PWF)', pwf_risk,
                                 f"P={power:.0f}W (plage: 3500-9000W)"))
            # OSF
            osf_thresholds = {'L': 11000, 'M': 12000, 'H': 13000}
            osf_value = torque * wear
            osf_risk = min(osf_value / osf_thresholds[m_type] * 100, 100)
            risk_factors.append(('Sur-contrainte (OSF)', osf_risk,
                                 f"Couple×Usure={osf_value:.0f}/{osf_thresholds[m_type]}"))

            risk_df = pd.DataFrame(risk_factors, columns=['Facteur', 'Risque (%)', 'Détail'])
            risk_df = risk_df.sort_values('Risque (%)', ascending=True)

            fig_risk = px.bar(
                risk_df, x='Risque (%)', y='Facteur',
                orientation='h', text='Risque (%)',
                color='Risque (%)', color_continuous_scale='RdYlGn_r',
                range_color=[0, 100],
                hover_data=['Détail'],
                template=TEMPLATE
            )
            fig_risk.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
            fig_risk.update_layout(
                xaxis_range=[0, 115],
                coloraxis_showscale=False,
                height=300,
                margin=dict(t=20, b=20)
            )
            st.plotly_chart(fig_risk, use_container_width=True)

            # Recommandations
            top_risk = risk_df.iloc[-1]
            if top_risk['Risque (%)'] > 70:
                st.markdown(
                    f"<div class='alert-danger'>🎯 <b>Facteur critique détecté :</b> "
                    f"{top_risk['Facteur']} à <b>{top_risk['Risque (%)']:.0f}%</b><br>"
                    f"📌 {top_risk['Détail']}</div>",
                    unsafe_allow_html=True
                )

        else:
            st.info("👈 Saisissez les paramètres capteurs et cliquez sur **LANCER LE DIAGNOSTIC** "
                    "pour obtenir l'analyse de risque IA.")

            # Aperçu des plages typiques
            st.markdown("##### 📐 Plages opérationnelles typiques")
            st.markdown("""
            - 🌡️ **Température Air** : 295 – 305 K
            - 🔥 **Température Process** : 305 – 315 K (toujours > Air)
            - ⚙️ **Vitesse rotation** : 1300 – 2900 RPM
            - 🔩 **Couple** : 3 – 77 Nm
            - ⏱️ **Usure outil** : remplacer avant **200 min**
            """)

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.caption("🏭 Industrial AI Insights – Maintenance Prédictive | Dataset AI4I 2020 | "
           "Modèle ML chargé depuis `modele_maintenance_predictive.pkl`")
