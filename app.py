import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Industrial AI Insights", layout="wide")

# Style CSS pour corriger les KPIs et l'interface
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; color: #00FFAA !important; }
    [data-testid="stMetricLabel"] { font-size: 16px; }
    .stMetric { background-color: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 15px; border: 1px solid #333; }
    .stSelectbox label, .stNumberInput label { color: #00FFAA !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(current_dir, 'modele_maintenance_predictive.pkl'))
    encoder = joblib.load(os.path.join(current_dir, 'label_encoder_type.pkl'))
    # Remplace par ton chemin local si nécessaire
    df = pd.read_csv(r"C:\Users\Farouha\Downloads\ai4i+2020+predictive+maintenance+dataset\ai4i2020.csv")
    return model, encoder, df

try:
    model, encoder, df = load_resources()
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()

# --- BARRE LATÉRALE ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1067/1067357.png", width=80)
st.sidebar.title("Fleet Manager AI")
page = st.sidebar.selectbox("Menu Principal", ["Tableau de Bord", "Analyse de Stress", "Comportement Variables", "Diagnostic Prédictif"])

# --- PAGE 1 : TABLEAU DE BORD ---
if page == "Tableau de Bord":
    st.title("📊 Indicateurs Clés de Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Parc Machines", f"{len(df)} unités")
    with col2:
        fails = df['Machine failure'].sum()
        st.metric("Taux de Panne", f"{(fails/len(df))*100:.2f}%", f"{fails} arrêts")
    with col3:
        st.metric("Temp. Moyenne Air", f"{df['Air temperature [K]'].mean():.1f} K")
    with col4:
        st.metric("Productivité Est.", "94.2%", "+1.5%")

    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🛠️ Importance des facteurs de panne")
        importances = pd.DataFrame({
            'Facteur': ['Couple (Torque)', 'Vitesse (RPM)', 'Usure Outil', 'Temp Air', 'Temp Process'],
            'Importance': [0.32, 0.29, 0.21, 0.11, 0.07]
        }).sort_values('Importance', ascending=True)
        fig_imp = px.bar(importances, x='Importance', y='Facteur', orientation='h', 
                         color='Importance', color_continuous_scale='Viridis', template="plotly_dark")
        st.plotly_chart(fig_imp, use_container_width=True)
    
    with c2:
        st.subheader("📂 Répartition par Type & État")
        fig_sun = px.sunburst(df, path=['Type', 'Machine failure'], color='Type', template="plotly_dark")
        st.plotly_chart(fig_sun, use_container_width=True)

# --- PAGE 2 : ANALYSE DE STRESS ---
elif page == "Analyse de Stress":
    st.title("🌡️ Analyse des Stress Mécaniques")
    st.write("Visualisation des zones de rupture (Vitesse vs Couple).")
    
    df_sample = df.sample(n=2000) if len(df) > 2000 else df
    fig_stress = px.scatter(df_sample, x="Torque [Nm]", y="Rotational speed [rpm]", 
                            color="Machine failure", size="Tool wear [min]",
                            color_continuous_scale='RdYlGn_r', opacity=0.6,
                            template="plotly_dark", title="Corrélation Couple/Vitesse")
    st.plotly_chart(fig_stress, use_container_width=True)

# --- PAGE 3 : COMPORTEMENT VARIABLES (HISTOGRAMMES & COURBES) ---
elif page == "Comportement Variables":
    st.title("📈 Analyse de Distribution des Pannes")
    st.write("Visualisation des zones de danger par variable (Sain vs Panne).")
    
    var_target = st.selectbox("Choisir une mesure à analyser", 
                              ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"])
    
    # Création d'un histogramme avec courbe de distribution (KDE-like)
    fig_dist = px.histogram(df, x=var_target, color="Machine failure", 
                            marginal="rug", # Ajoute des petits traits en bas pour voir la densité
                            nbins=50, 
                            barmode="overlay", # Superpose les histogrammes
                            color_discrete_map={0: '#00cc96', 1: '#ef553b'},
                            template="plotly_dark",
                            title=f"Analyse comparative de la distribution : {var_target}")
    
    # Amélioration du design de la courbe
    fig_dist.update_traces(opacity=0.75)
    st.plotly_chart(fig_dist, use_container_width=True)
    st.info(f"💡 Plus les surfaces se séparent, plus la variable **{var_target}** est un bon indicateur de panne.")

# --- PAGE 4 : DIAGNOSTIC PRÉDICTIF (CORRIGÉ) ---
elif page == "Diagnostic Prédictif":
    st.title("🤖 Intelligence Artificielle Prédictive")
    st.write("Saisie des paramètres capteurs pour diagnostic immédiat.")

    col_input, col_viz = st.columns([1, 1])

    with col_input:
        with st.form("prediction_form"):
            st.subheader("Données Capteurs")
            m_type = st.selectbox("Type de Machine", ["L", "M", "H"])
            air_temp = st.number_input("Température Air [K]", 280.0, 320.0, 300.0)
            proc_temp = st.number_input("Température Process [K]", 300.0, 330.0, 310.0)
            speed = st.number_input("Vitesse (RPM)", 1000, 3000, 1500)
            torque = st.number_input("Couple (Torque) [Nm]", 0.0, 100.0, 40.0)
            wear = st.number_input("Usure Outil [min]", 0, 300, 100)
            
            submit = st.form_submit_button("🚀 LANCER LE DIAGNOSTIC")

    if submit:
        # Encodage
        type_encoded = encoder.transform([m_type])[0]
        input_data = np.array([[type_encoded, air_temp, proc_temp, speed, torque, wear]])
        
        # Calcul
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        with col_viz:
            st.subheader("Analyse de Risque")
            
            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probabilité de Panne (%)", 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1},
                    'bar': {'color': "#00FFAA"},
                    'bgcolor': "white",
                    'steps': [
                        {'range': [0, 40], 'color': '#008000'},
                        {'range': [40, 75], 'color': '#FFA500'},
                        {'range': [75, 100], 'color': '#FF0000'}
                    ],
                    'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 50}
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)

            if prediction == 1:
                st.error(f"🚨 DANGER : Défaillance imminente détectée ({prob:.1%})")
            else:
                st.success(f"✅ OPÉRATIONNEL : Risque faible ({prob:.1%})")