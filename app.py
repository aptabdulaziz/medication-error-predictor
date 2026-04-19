import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Medication Error Predictor", page_icon="🏥", layout="wide")

st.title("🏥 Sistem Prediksi Medication Error")
st.markdown("**Prototype Penelitian Tesis S2 Farmasi Klinis**")
st.markdown("---")

# Sidebar
menu = st.sidebar.radio("Menu:", ["🎯 Prediksi", "ℹ️ Info"])

# Load demo model
@st.cache_resource
def get_model():
    np.random.seed(42)
    X = np.random.rand(1000, 5) * 10
    y = (X[:, 0] + X[:, 1] * 0.5 > 8).astype(int)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

model = get_model()

if menu == "🎯 Prediksi":
    st.header("Input Data Pasien")
    
    col1, col2 = st.columns(2)
    with col1:
        usia = st.slider("Usia (tahun)", 18, 95, 55)
        jumlah_obat = st.slider("Jumlah obat", 1, 20, 6)
        lama_rawat = st.slider("Lama rawat (hari)", 1, 30, 8)
    with col2:
        komorbiditas = st.slider("Komorbiditas", 0, 5, 2)
        beban_kerja = st.slider("Beban kerja (1-10)", 1, 10, 6)
    
    if st.button("🔮 Prediksi Risiko", type="primary"):
        input_data = np.array([[usia/10, jumlah_obat, lama_rawat, komorbiditas, beban_kerja]])
        risk_score = model.predict_proba(input_data)[0][1]
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Risk Score", f"{risk_score:.3f}")
        with col_b:
            st.metric("Persentase", f"{risk_score*100:.1f}%")
        
        if risk_score > 0.7:
            st.error("🔴 HIGH RISK - Review apoteker segera")
        elif risk_score > 0.4:
            st.warning("🟡 MEDIUM RISK - Monitor ketat")
        else:
            st.success("🟢 LOW RISK - Pengawasan standar")

else:
    st.header("ℹ️ Tentang Sistem")
    st.markdown("""
    ### Medication Error Prediction System
    
    **Model:** Random Forest Classifier
    **Fitur:** usia, jumlah_obat, lama_rawat, komorbiditas, beban_kerja
    
    **Disclaimer:** Prototype untuk penelitian. Data simulasi.
    
    © 2025 Penelitian Tesis S2 Farmasi Klinis
    """)

st.markdown("---")
st.caption("Medication Error AI System • Prototype")

# Di app.py, tambahkan di bagian bawah
st.markdown("""
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
""", unsafe_allow_html=True)
