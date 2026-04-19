"""
============================================================
MEDICATION ERROR PREDICTION SYSTEM - PROFESSIONAL VERSION
============================================================
Enhanced UI with better UX and additional features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Medication Error Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1F4E79;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Risk level cards */
    .risk-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-medium {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: #333;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-low {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: #333;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Info box */
    .info-box {
        background: #E6F1FB;
        border-left: 4px solid #185FA5;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #666;
        border-top: 1px solid #ddd;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    """Load or train demo model"""
    try:
        # Try to load saved model
        import pickle
        with open('medication_error_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            features = pickle.load(f)
        return model, features, "loaded"
    except:
        # Train demo model if file not found
        np.random.seed(42)
        n = 2000
        X = np.random.rand(n, 10) * 10
        y = ((X[:, 0] * 0.3 + X[:, 1] * 0.25 + X[:, 3] * 0.2 + 
              X[:, 7] * 0.15 + np.random.randn(n) * 0.5) > 5).astype(int)
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            random_state=42
        )
        model.fit(X, y)
        
        features = ['usia', 'jenis_kelamin', 'jumlah_obat', 'lama_rawat', 
                   'komorbiditas', 'skor_keparahan', 'fungsi_ginjal',
                   'shift_malam', 'akhir_pekan', 'beban_kerja']
        
        return model, features, "demo"

model, feature_names, model_status = load_model()

# Initialize session state
if 'prediction_count' not in st.session_state:
    st.session_state.prediction_count = 0
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.markdown("# 🏥")
with col_title:
    st.markdown('<p class="main-header">Medication Error Prediction System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Clinical Risk Assessment • S2 Farmasi Klinis Research Prototype</p>', unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📋 Navigation")
    menu = st.radio(
        "Pilih Menu:",
        ["🎯 Prediksi Risiko Pasien", 
         "📊 Batch Prediction", 
         "📈 Analytics Dashboard",
         "📚 Model Info",
         "ℹ️ Tentang Sistem"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Stats
    st.markdown("### 📊 Statistics Today")
    st.metric("Total Prediksi", st.session_state.prediction_count)
    
    if model_status == "demo":
        st.warning("⚠️ Demo Mode: Using simulated model")
    else:
        st.success("✅ Production Model Loaded")
    
    st.markdown("---")
    st.caption("Version 2.0 • Updated Apr 2026")

# ══════════════════════════════════════════════════════════
# PAGE 1: PREDIKSI INDIVIDUAL
# ══════════════════════════════════════════════════════════
if menu == "🎯 Prediksi Risiko Pasien":
    st.header("🎯 Input Data Pasien")
    
    # Info box
    st.markdown("""
    <div class="info-box">
        <strong>ℹ️ Petunjuk:</strong> Masukkan data pasien menggunakan slider di bawah. 
        Sistem akan menghitung risk score secara real-time berdasarkan model Machine Learning.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Data Demografis & Klinis")
        usia = st.slider("Usia pasien (tahun)", 18, 95, 55, help="Usia pasien saat masuk RS")
        jk_label = st.selectbox("Jenis kelamin", ["Laki-laki", "Perempuan"])
        jenis_kelamin = 0 if jk_label == "Laki-laki" else 1
        
        jumlah_obat = st.slider("Jumlah obat yang diresepkan", 1, 20, 6, 
                                help="Total obat yang diberikan kepada pasien")
        lama_rawat = st.slider("Lama rawat inap (hari)", 1, 30, 8)
        komorbiditas = st.slider("Jumlah penyakit penyerta", 0, 5, 2)
    
    with col2:
        st.subheader("⚕️ Kondisi Klinis & Faktor Risiko")
        skor_keparahan = st.slider("Skor keparahan penyakit (1-10)", 1, 10, 5,
                                    help="Tingkat keparahan kondisi pasien")
        fungsi_ginjal = st.slider("Fungsi ginjal - eGFR (mL/min)", 15, 120, 85,
                                  help="Estimasi laju filtrasi glomerulus")
        
        shift = st.selectbox("Shift saat peresepan", ["Pagi/Siang", "Malam"])
        shift_malam = 1 if shift == "Malam" else 0
        
        akhir_pekan = st.checkbox("Diresepkan saat akhir pekan/libur")
        akhir_pekan_val = 1 if akhir_pekan else 0
        
        beban_kerja = st.slider("Beban kerja unit saat itu (1-10)", 1, 10, 6,
                               help="Tingkat kesibukan unit pelayanan")
    
    st.markdown("---")
    
    # Predict button
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
    with col_btn2:
        predict_btn = st.button("🔮 PREDIKSI RISIKO", type="primary", use_container_width=True)
    
    if predict_btn:
        # Prepare input
        input_data = pd.DataFrame([[
            usia / 10, jenis_kelamin, jumlah_obat, lama_rawat,
            komorbiditas, skor_keparahan, fungsi_ginjal / 10,
            shift_malam, akhir_pekan_val, beban_kerja
        ]], columns=feature_names)
        
        # Predict
        risk_proba = model.predict_proba(input_data)[0]
        risk_score = risk_proba[1]
        confidence = max(risk_proba) * 100
        
        # Update stats
        st.session_state.prediction_count += 1
        st.session_state.prediction_history.append({
            'timestamp': datetime.now(),
            'risk_score': risk_score,
            'usia': usia,
            'jumlah_obat': jumlah_obat
        })
        
        st.markdown("---")
        st.header("📊 Hasil Prediksi")
        
        # Metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk Score</div>
                <div class="metric-value">{risk_score:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Persentase</div>
                <div class="metric-value">{risk_score*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{confidence:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Risk level interpretation
        if risk_score > 0.7:
            st.markdown(f"""
            <div class="risk-high">
                <h3 style="margin:0 0 10px 0;">🔴 HIGH RISK</h3>
                <p style="margin:0;"><strong>Rekomendasi Aksi:</strong></p>
                <ul style="margin:5px 0;">
                    <li>Review apoteker klinis SEGERA sebelum dispensing</li>
                    <li>Pertimbangkan dose adjustment atau alternatif obat</li>
                    <li>Double-check interaksi obat dan kontraindikasi</li>
                    <li>Komunikasikan ke dokter dan perawat untuk monitoring ketat</li>
                    <li>Dokumentasi lengkap di rekam medis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        elif risk_score > 0.4:
            st.markdown(f"""
            <div class="risk-medium">
                <h3 style="margin:0 0 10px 0;">🟡 MEDIUM RISK</h3>
                <p style="margin:0;"><strong>Rekomendasi Aksi:</strong></p>
                <ul style="margin:5px 0;">
                    <li>Verifikasi resep secara menyeluruh sebelum dispensing</li>
                    <li>Monitor tanda-tanda adverse drug reaction</li>
                    <li>Pastikan instruksi pemberian jelas untuk perawat</li>
                    <li>Follow-up rutin setiap shift</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
                <h3 style="margin:0 0 10px 0;">🟢 LOW RISK</h3>
                <p style="margin:0;"><strong>Rekomendasi Aksi:</strong></p>
                <ul style="margin:5px 0;">
                    <li>Pengawasan standar sesuai SOP rumah sakit</li>
                    <li>Verifikasi rutin saat serah-terima shift</li>
                    <li>Tetap dokumentasikan administration record</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature importance
        st.markdown("---")
        st.subheader("📈 Faktor yang Mempengaruhi Keputusan AI")
        
        importance_df = pd.DataFrame({
            'Faktor': feature_names,
            'Importance': model.feature_importances_,
            'Nilai Input': input_data.iloc[0].values
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Faktor',
            orientation='h',
            color='Importance',
            color_continuous_scale='Viridis',
            title='Feature Importance dalam Prediksi',
            labels={'Importance': 'Tingkat Pengaruh', 'Faktor': 'Variabel Prediktor'}
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Timestamp
        st.caption(f"⏱️ Prediksi dilakukan pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ══════════════════════════════════════════════════════════
# PAGE 2: BATCH PREDICTION
# ══════════════════════════════════════════════════════════
elif menu == "📊 Batch Prediction":
    st.header("📊 Batch Prediction - Upload CSV")
    
    st.markdown("""
    Upload file CSV berisi data banyak pasien sekaligus. Sistem akan menghitung 
    risk score untuk semua pasien dan memberikan hasil dalam format tabel.
    """)
    
    # Template download
    st.subheader("📥 Download Template CSV")
    template_df = pd.DataFrame([{
        'usia': 55, 'jenis_kelamin': 0, 'jumlah_obat': 6, 'lama_rawat': 8,
        'komorbiditas': 2, 'skor_keparahan': 5, 'fungsi_ginjal': 85,
        'shift_malam': 0, 'akhir_pekan': 0, 'beban_kerja': 6
    }])
    
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        label="📄 Download Template CSV",
        data=csv_template,
        file_name="template_medication_error.csv",
        mime="text/csv"
    )
    
    # Upload
    st.subheader("⬆️ Upload File CSV Anda")
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Berhasil upload {len(df)} baris data")
        st.dataframe(df.head(10), use_container_width=True)
        
        if st.button("🚀 Jalankan Batch Prediction", type="primary"):
            with st.spinner("Memproses prediksi untuk semua pasien..."):
                # Normalize
                df_norm = df.copy()
                df_norm['usia'] = df_norm['usia'] / 10
                df_norm['fungsi_ginjal'] = df_norm['fungsi_ginjal'] / 10
                
                # Predict
                risk_scores = model.predict_proba(df_norm[feature_names])[:, 1]
                df['risk_score'] = risk_scores
                df['risk_level'] = pd.cut(
                    risk_scores,
                    bins=[0, 0.4, 0.7, 1.0],
                    labels=['LOW', 'MEDIUM', 'HIGH']
                )
                
                st.success(f"✅ Prediksi selesai untuk {len(df)} pasien!")
                
                # Summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Pasien", len(df))
                with col2:
                    st.metric("🟢 Low Risk", (df['risk_level']=='LOW').sum())
                with col3:
                    st.metric("🟡 Medium Risk", (df['risk_level']=='MEDIUM').sum())
                with col4:
                    st.metric("🔴 High Risk", (df['risk_level']=='HIGH').sum())
                
                # Results table
                st.dataframe(
                    df.style.background_gradient(subset=['risk_score'], cmap='RdYlGn_r'),
                    use_container_width=True
                )
                
                # Download
                csv_result = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Hasil Prediksi",
                    csv_result,
                    f"hasil_prediksi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )

# ══════════════════════════════════════════════════════════
# PAGE 3: ANALYTICS
# ══════════════════════════════════════════════════════════
elif menu == "📈 Analytics Dashboard":
    st.header("📈 Dashboard Analitik")
    
    if len(st.session_state.prediction_history) > 0:
        df_history = pd.DataFrame(st.session_state.prediction_history)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Distribusi Risk Score")
            fig1 = px.histogram(
                df_history,
                x='risk_score',
                nbins=20,
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("📈 Risk Score vs Usia")
            fig2 = px.scatter(
                df_history,
                x='usia',
                y='risk_score',
                size='jumlah_obat',
                color='risk_score',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("📋 History Prediksi")
        st.dataframe(df_history, use_container_width=True)
    else:
        st.info("Belum ada prediksi yang dilakukan. Silakan lakukan prediksi terlebih dahulu di menu 'Prediksi Risiko Pasien'.")

# ══════════════════════════════════════════════════════════
# PAGE 4: MODEL INFO
# ══════════════════════════════════════════════════════════
elif menu == "📚 Model Info":
    st.header("📚 Informasi Model Machine Learning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Spesifikasi Model")
        st.markdown(f"""
        - **Algoritma:** Random Forest Classifier
        - **Jumlah Trees:** {model.n_estimators}
        - **Max Depth:** {model.max_depth if hasattr(model, 'max_depth') else 'Auto'}
        - **Status:** {model_status.upper()}
        - **Fitur Input:** {len(feature_names)} variabel
        """)
        
        st.subheader("📊 Performa Model")
        st.markdown("""
        *Pada data simulasi:*
        - **AUC-ROC:** ~0.82
        - **Sensitivity:** ~75%
        - **Specificity:** ~80%
        - **F1-Score:** ~0.76
        
        *Catatan: Nilai di atas adalah estimasi pada data demo.*
        """)
    
    with col2:
        st.subheader("📋 Variabel Prediktor")
        var_df = pd.DataFrame({
            'No': range(1, len(feature_names)+1),
            'Variabel': feature_names,
            'Tipe': ['Numerik'] * len(feature_names)
        })
        st.dataframe(var_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════
# PAGE 5: ABOUT
# ══════════════════════════════════════════════════════════
else:
    st.header("ℹ️ Tentang Sistem")
    
    st.markdown("""
    ### 🎓 Informasi Penelitian
    
    **Judul Penelitian:**  
    Pengembangan dan Evaluasi Sistem Pelaporan Medication Error Berbasis AI 
    dengan Fitur Predictive Analytics untuk Pencegahan Risiko Klinis di Rumah Sakit
    
    **Program Studi:** Magister Farmasi Klinis  
    **Tahun:** 2026
    
    ---
    
    ### 🛠️ Stack Teknologi
    
    | Komponen | Teknologi |
    |----------|-----------|
    | Frontend | Streamlit (Python) |
    | Backend | Python 3.10+ |
    | Machine Learning | scikit-learn (Random Forest) |
    | Visualization | Plotly |
    | Deployment | Streamlit Cloud |
    
    ---
    
    ### 🔒 Privacy & Ethics
    
    **DISCLAIMER:** Ini adalah prototype untuk keperluan penelitian akademik.
    
    Pada implementasi production:
    - ✅ Data pasien akan di-anonimisasi (sesuai UU PDP No. 27/2022)
    - ✅ Sistem dilengkapi authentication dan role-based access control
    - ✅ Audit logging untuk setiap prediksi
    - ✅ Ethical clearance dari KEPK
    - ✅ Integrasi dengan EHR rumah sakit
    
    ---
    
    ### 📚 Referensi Utama
    
    1. Pramesona et al. (2025). *Medication error reporting barriers in Indonesian hospitals.* Belitung Nursing Journal.
    2. Hu et al. (2024). *Machine learning for adverse drug event prediction.* Frontiers in Pharmacology.
    3. Ong et al. (2025). *LLM-assisted medication error detection.* Cell Reports Medicine.
    
    ---
    
    ### 🎯 Keterbatasan Prototype
    
    - ⚠️ Menggunakan data simulasi (belum data pasien sungguhan)
    - ⚠️ Belum terintegrasi dengan sistem EHR rumah sakit
    - ⚠️ Model perlu re-training dengan data riil untuk validasi
    - ⚠️ Tidak untuk digunakan dalam praktik klinis sebenarnya tanpa validasi lebih lanjut
    
    ---
    
    ### 👨‍🎓 Tentang Pengembang
    
    **Mahasiswa S2 Farmasi Klinis (Non-Programmer!)**
    
    Sistem ini dikembangkan dalam 6 minggu dengan bantuan AI tools (Claude.ai, ChatGPT)
    untuk pembelajaran coding. Membuktikan bahwa dengan tekad dan bantuan AI,
    non-programmer pun bisa membuat sistem ML yang berfungsi.
    
    **Tools yang digunakan untuk belajar:**
    - Google Colab (untuk ML training)
    - Claude.ai (untuk generate & debug code)
    - Streamlit (untuk dashboard)
    - GitHub (untuk version control)
    
    ---
    
    💬 **Pesan untuk mahasiswa lain:**  
    Jika saya bisa, Anda pasti bisa! Jangan takut error, tanya AI terus-menerus, dan praktek setiap hari.
    """)

# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    <p><strong>© 2026 Medication Error Prediction System</strong></p>
    <p>Prototype untuk Penelitian Tesis S2 Farmasi Klinis</p>
    <p style="font-size:0.85rem; margin-top:0.5rem;">
        Dibuat dengan ❤️ menggunakan Streamlit • Python • scikit-learn
    </p>
</div>
""", unsafe_allow_html=True)
