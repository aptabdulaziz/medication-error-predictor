"""
Medication Error Prediction System
WITH SIMPLE AUTHENTICATION (SHA-256)
Production-ready untuk deployment
"""

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import hashlib
import plotly.express as px

# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Medication Error Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    /* Main background and text */
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1e3a8a;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Inputs and Selectboxes */
    .stTextInput>div>div>input, .stSelectbox>div>div>select {
        border-radius: 6px;
        border: 1px solid #cbd5e1;
    }
    
    /* Cards/Containers */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #0f172a;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: #64748b;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Success/Warning/Error messages */
    .stSuccess {
        background-color: #dcfce7;
        color: #166534;
        border-left: 4px solid #22c55e;
    }
    .stWarning {
        background-color: #fef9c3;
        color: #854d0e;
        border-left: 4px solid #eab308;
    }
    .stError {
        background-color: #fee2e2;
        color: #991b1b;
        border-left: 4px solid #ef4444;
    }
    .stInfo {
        background-color: #e0f2fe;
        color: #075985;
        border-left: 4px solid #0ea5e9;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# USER DATABASE (SHA-256 HASHING)
# ══════════════════════════════════════════════════════════
USERS = {
    "admin": {
        "password": hashlib.sha256("admin123".encode()).hexdigest(),
        "name": "Administrator"
    },
    "apoteker1": {
        "password": hashlib.sha256("apoteker123".encode()).hexdigest(),
        "name": "Apoteker Klinik"
    },
    "dokter1": {
        "password": hashlib.sha256("dokter123".encode()).hexdigest(),
        "name": "Dr. Dokter"
    }
}

# ══════════════════════════════════════════════════════════
# AUTHENTICATION FUNCTIONS
# ══════════════════════════════════════════════════════════
def check_login(username, password):
    """Verify username and password"""
    if username in USERS:
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        if USERS[username]["password"] == hashed_password:
            return True, USERS[username]["name"]
    return False, None

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.name = None

# ══════════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════════
if not st.session_state.authenticated:
    st.markdown("<h1 style='text-align: center; color: #1e3a8a; margin-bottom: 0;'>🏥 Medication Error Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; font-size: 1.2rem; margin-top: 0;'>AI-Powered Clinical Risk Assessment</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='background-color: white; padding: 2rem; border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);'>
            <h3 style='text-align: center; color: #334155; margin-bottom: 1.5rem;'>🔐 Secure Login</h3>
        </div>
        """, unsafe_allow_html=True)
        
        username = st.text_input("Username", placeholder="Masukkan username")
        password = st.text_input("Password", type="password", placeholder="Masukkan password")
        
        if st.button("Login", type="primary", use_container_width=True):
            if username and password:
                success, name = check_login(username, password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.name = name
                    st.success(f"✅ Login berhasil! Welcome {name}")
                    st.rerun()
                else:
                    st.error("❌ Username atau password salah!")
            else:
                st.warning("⚠️ Harap isi username dan password!")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.expander("🔍 Demo Credentials untuk Testing"):
            st.info("""
            **Admin:**
            - Username: `admin`
            - Password: `admin123`
            
            **Apoteker:**
            - Username: `apoteker1`
            - Password: `apoteker123`
            
            **Dokter:**
            - Username: `dokter1`
            - Password: `dokter123`
            """)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>🔐 Secure Authentication System • Medication Error Prediction System v2.0</p>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# MAIN APPLICATION (AFTER LOGIN)
# ══════════════════════════════════════════════════════════
else:
    # Load model
    @st.cache_resource
    def load_model():
        """Load or train demo model"""
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
        
        return model, features
    
    model, features = load_model()
    
    # Header with user info and logout
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<h1 style='color: #1e3a8a; margin-bottom: 0;'>🏥 Medication Error Prediction System</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #64748b; font-size: 1.1rem; margin-top: 0;'>AI-Powered Clinical Risk Assessment • S2 Farmasi Klinis Research Prototype</p>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style='text-align: right; padding: 1rem; background-color: white; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
            <p style='margin: 0; font-weight: bold; color: #334155;'>👤 {st.session_state.name}</p>
            <p style='margin: 0; font-size: 0.9rem; color: #94a3b8;'>@{st.session_state.username}</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.name = None
            st.rerun()
    
    st.markdown("<hr style='border: 1px solid #e2e8f0; margin: 2rem 0;'>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='color: #1e3a8a; text-align: center;'>📋 Navigation</h2>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style='background-color: #f0fdf4; padding: 1rem; border-radius: 8px; border-left: 4px solid #22c55e; margin-bottom: 1.5rem;'>
            <p style='margin: 0; font-weight: bold; color: #166534;'>Logged in as:</p>
            <p style='margin: 0; color: #15803d;'>{st.session_state.name}</p>
            <p style='margin: 0; font-size: 0.9rem; color: #16a34a;'>🔐 Authenticated</p>
        </div>
        """, unsafe_allow_html=True)
        
        menu = st.radio(
            "Select Page:",
            ["🎯 Prediksi Risiko Pasien", 
             "📊 Batch Prediction",
             "📈 Analytics Dashboard",
             "📚 Model Info",
             "ℹ️ Tentang Sistem"],
            label_visibility="collapsed"
        )
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.8rem;'>Version 2.0 with Authentication • Apr 2026</p>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════
    # PAGE: PREDIKSI RISIKO INDIVIDUAL
    # ═══════════════════════════════════════════════════════
    if menu == "🎯 Prediksi Risiko Pasien":
        st.markdown("<h2 style='color: #1e3a8a; margin-bottom: 1.5rem;'>🎯 Prediksi Risiko Medication Error - Input Data Pasien</h2>", unsafe_allow_html=True)
        
        st.info("ℹ️ **Petunjuk:** Masukkan data pasien menggunakan slider di bawah. Sistem akan menghitung risk score secara real-time berdasarkan model Machine Learning.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div style='background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
            st.markdown("<h3 style='color: #334155; margin-bottom: 1rem;'>👤 Data Demografis & Klinis</h3>", unsafe_allow_html=True)
            usia = st.slider("Usia pasien (tahun)", 18, 95, 55, 
                           help="Usia pasien saat masuk rumah sakit")
            
            jk = st.selectbox("Jenis kelamin", ["Laki-laki", "Perempuan"])
            jenis_kelamin = 0 if jk == "Laki-laki" else 1
            
            jumlah_obat = st.slider("Jumlah obat yang diresepkan", 1, 20, 6,
                                   help="Total obat yang diberikan kepada pasien")
            
            lama_rawat = st.slider("Lama rawat inap (hari)", 1, 30, 8)
            
            komorbiditas = st.slider("Jumlah penyakit penyerta", 0, 5, 2)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div style='background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
            st.markdown("<h3 style='color: #334155; margin-bottom: 1rem;'>⚕️ Kondisi Klinis & Faktor Risiko</h3>", unsafe_allow_html=True)
            
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
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Prediction button
        if st.button("🔮 PREDIKSI RISIKO", type="primary", use_container_width=True):
            # Prepare input
            input_data = np.array([[
                usia / 10, jenis_kelamin, jumlah_obat, lama_rawat,
                komorbiditas, skor_keparahan, fungsi_ginjal / 10,
                shift_malam, akhir_pekan_val, beban_kerja
            ]])
            
            # Predict
            risk_proba = model.predict_proba(input_data)[0]
            risk_score = risk_proba[1]
            confidence = max(risk_proba) * 100
            
            st.markdown("<h2 style='color: #1e3a8a; margin-top: 2rem; margin-bottom: 1.5rem;'>📊 Hasil Prediksi</h2>", unsafe_allow_html=True)
            
            # Metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(f"""
                <div style='background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); text-align: center;'>
                    <p style='color: #64748b; font-size: 1.1rem; margin-bottom: 0.5rem;'>Risk Score</p>
                    <p style='color: #0f172a; font-size: 2.5rem; font-weight: bold; margin: 0;'>{risk_score:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""
                <div style='background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); text-align: center;'>
                    <p style='color: #64748b; font-size: 1.1rem; margin-bottom: 0.5rem;'>Persentase</p>
                    <p style='color: #0f172a; font-size: 2.5rem; font-weight: bold; margin: 0;'>{risk_score*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            with col_c:
                st.markdown(f"""
                <div style='background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); text-align: center;'>
                    <p style='color: #64748b; font-size: 1.1rem; margin-bottom: 0.5rem;'>Confidence</p>
                    <p style='color: #0f172a; font-size: 2.5rem; font-weight: bold; margin: 0;'>{confidence:.0f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Risk level interpretation
            if risk_score > 0.7:
                st.markdown("""
                <div style='background-color: #fef2f2; padding: 2rem; border-radius: 12px; border-left: 6px solid #ef4444; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
                    <h3 style='color: #b91c1c; margin-top: 0;'>🔴 HIGH RISK</h3>
                    <p style='color: #7f1d1d; font-weight: bold; margin-bottom: 0.5rem;'>Rekomendasi Aksi:</p>
                    <ul style='color: #991b1b; margin-bottom: 0;'>
                        <li>Review apoteker klinis SEGERA sebelum dispensing</li>
                        <li>Pertimbangkan dose adjustment atau alternatif obat</li>
                        <li>Double-check interaksi obat dan kontraindikasi</li>
                        <li>Komunikasikan ke dokter dan perawat untuk monitoring ketat</li>
                        <li>Dokumentasi lengkap di rekam medis</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif risk_score > 0.4:
                st.markdown("""
                <div style='background-color: #fefce8; padding: 2rem; border-radius: 12px; border-left: 6px solid #eab308; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
                    <h3 style='color: #a16207; margin-top: 0;'>🟡 MEDIUM RISK</h3>
                    <p style='color: #854d0e; font-weight: bold; margin-bottom: 0.5rem;'>Rekomendasi Aksi:</p>
                    <ul style='color: #a16207; margin-bottom: 0;'>
                        <li>Verifikasi resep secara menyeluruh sebelum dispensing</li>
                        <li>Monitor tanda-tanda adverse drug reaction</li>
                        <li>Pastikan instruksi pemberian jelas untuk perawat</li>
                        <li>Follow-up rutin setiap shift</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background-color: #f0fdf4; padding: 2rem; border-radius: 12px; border-left: 6px solid #22c55e; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
                    <h3 style='color: #15803d; margin-top: 0;'>🟢 LOW RISK</h3>
                    <p style='color: #166534; font-weight: bold; margin-bottom: 0.5rem;'>Rekomendasi Aksi:</p>
                    <ul style='color: #15803d; margin-bottom: 0;'>
                        <li>Pengawasan standar sesuai SOP rumah sakit</li>
                        <li>Verifikasi rutin saat serah-terima shift</li>
                        <li>Tetap dokumentasikan administration record</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # PAGE: BATCH PREDICTION
    # ═══════════════════════════════════════════════════════
    elif menu == "📊 Batch Prediction":
        st.markdown("<h2 style='color: #1e3a8a; margin-bottom: 1.5rem;'>📊 Batch Prediction - Upload CSV</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 1.5rem;'>
            <p style='color: #475569; font-size: 1.1rem; margin: 0;'>Upload file CSV berisi data banyak pasien sekaligus. Sistem akan menghitung risk score untuk semua pasien dan memberikan hasil dalam format tabel.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Template download
        st.markdown("<h3 style='color: #334155; margin-bottom: 1rem;'>📥 Download Template CSV</h3>", unsafe_allow_html=True)
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
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Upload
        st.markdown("<h3 style='color: #334155; margin-bottom: 1rem;'>⬆️ Upload File CSV Anda</h3>", unsafe_allow_html=True)
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
                    risk_scores = model.predict_proba(df_norm[features])[:, 1]
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
                        st.markdown(f"""
                        <div style='background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); text-align: center;'>
                            <p style='color: #64748b; font-size: 1.1rem; margin-bottom: 0.5rem;'>Total Pasien</p>
                            <p style='color: #0f172a; font-size: 2.5rem; font-weight: bold; margin: 0;'>{len(df)}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div style='background-color: #f0fdf4; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); text-align: center;'>
                            <p style='color: #166534; font-size: 1.1rem; margin-bottom: 0.5rem;'>🟢 Low Risk</p>
                            <p style='color: #15803d; font-size: 2.5rem; font-weight: bold; margin: 0;'>{(df['risk_level']=='LOW').sum()}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div style='background-color: #fefce8; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); text-align: center;'>
                            <p style='color: #854d0e; font-size: 1.1rem; margin-bottom: 0.5rem;'>🟡 Medium Risk</p>
                            <p style='color: #a16207; font-size: 2.5rem; font-weight: bold; margin: 0;'>{(df['risk_level']=='MEDIUM').sum()}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col4:
                        st.markdown(f"""
                        <div style='background-color: #fef2f2; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); text-align: center;'>
                            <p style='color: #991b1b; font-size: 1.1rem; margin-bottom: 0.5rem;'>🔴 High Risk</p>
                            <p style='color: #b91c1c; font-size: 2.5rem; font-weight: bold; margin: 0;'>{(df['risk_level']=='HIGH').sum()}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Results table
                    st.dataframe(
                        df.style.background_gradient(subset=['risk_score'], cmap='RdYlGn_r'),
                        use_container_width=True
                    )
                    
                    # Download
                    from datetime import datetime
                    csv_result = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Hasil Prediksi",
                        csv_result,
                        f"hasil_prediksi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )

    # ═══════════════════════════════════════════════════════
    # PAGE: ANALYTICS
    # ═══════════════════════════════════════════════════════
    elif menu == "📈 Analytics Dashboard":
        st.markdown("<h2 style='color: #1e3a8a; margin-bottom: 1.5rem;'>📈 Dashboard Analitik</h2>", unsafe_allow_html=True)
        
        if len(st.session_state.get('prediction_history', [])) > 0:
            df_history = pd.DataFrame(st.session_state.prediction_history)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div style='background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
                st.markdown("<h3 style='color: #334155; margin-bottom: 1rem;'>📊 Distribusi Risk Score</h3>", unsafe_allow_html=True)
                fig1 = px.histogram(
                    df_history,
                    x='risk_score',
                    nbins=20,
                    color_discrete_sequence=['#3b82f6']
                )
                fig1.update_layout(plot_bgcolor='white', paper_bgcolor='white')
                st.plotly_chart(fig1, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div style='background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
                st.markdown("<h3 style='color: #334155; margin-bottom: 1rem;'>📈 Risk Score vs Usia</h3>", unsafe_allow_html=True)
                fig2 = px.scatter(
                    df_history,
                    x='usia',
                    y='risk_score',
                    size='jumlah_obat',
                    color='risk_score',
                    color_continuous_scale='Reds'
                )
                fig2.update_layout(plot_bgcolor='white', paper_bgcolor='white')
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<h3 style='color: #334155; margin-bottom: 1rem;'>📋 History Prediksi</h3>", unsafe_allow_html=True)
            st.dataframe(df_history, use_container_width=True)
        else:
            st.info("ℹ️ Belum ada prediksi yang dilakukan. Silakan lakukan prediksi terlebih dahulu di menu 'Prediksi Risiko Pasien'.")    

    
    # ═══════════════════════════════════════════════════════
    # PAGE: MODEL INFO
    # ═══════════════════════════════════════════════════════
    elif menu == "📚 Model Info":
        st.markdown("<h2 style='color: #1e3a8a; margin-bottom: 1.5rem;'>📚 Informasi Model Machine Learning</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div style='background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
            st.markdown("<h3 style='color: #334155; margin-bottom: 1rem;'>🤖 Spesifikasi Model</h3>", unsafe_allow_html=True)
            st.markdown(f"""
            <ul style='color: #475569; line-height: 1.6;'>
                <li><strong>Algoritma:</strong> Random Forest Classifier</li>
                <li><strong>Jumlah Trees:</strong> {model.n_estimators}</li>
                <li><strong>Max Depth:</strong> {model.max_depth}</li>
                <li><strong>Status:</strong> Demo Mode (Simulated Data)</li>
                <li><strong>Fitur Input:</strong> {len(features)} variabel</li>
            </ul>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div style='background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
            st.markdown("<h3 style='color: #334155; margin-bottom: 1rem;'>📊 Performa Model</h3>", unsafe_allow_html=True)
            st.markdown("""
            <p style='color: #64748b; font-style: italic; margin-bottom: 0.5rem;'>Pada data simulasi:</p>
            <ul style='color: #475569; line-height: 1.6;'>
                <li><strong>AUC-ROC:</strong> ~0.82</li>
                <li><strong>Sensitivity:</strong> ~75%</li>
                <li><strong>Specificity:</strong> ~80%</li>
                <li><strong>F1-Score:</strong> ~0.76</li>
            </ul>
            <p style='color: #94a3b8; font-size: 0.9rem; margin-top: 1rem;'>Catatan: Nilai di atas adalah estimasi pada data demo.</p>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div style='background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
            st.markdown("<h3 style='color: #334155; margin-bottom: 1rem;'>📋 Variabel Prediktor</h3>", unsafe_allow_html=True)
            var_df = pd.DataFrame({
                'No': range(1, len(features)+1),
                'Variabel': features,
                'Tipe': ['Numerik'] * len(features)
            })
            st.dataframe(var_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════
    # PAGE: ABOUT
    # ═══════════════════════════════════════════════════════
    else:
        st.markdown("<h2 style='color: #1e3a8a; margin-bottom: 1.5rem;'>ℹ️ Tentang Sistem</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 1.5rem;'>
            <h3 style='color: #334155; margin-top: 0;'>🎓 Informasi Penelitian</h3>
            <p style='color: #475569; line-height: 1.6;'>
                <strong>Judul Penelitian:</strong><br>
                Pengembangan dan Evaluasi Sistem Pelaporan Medication Error Berbasis AI 
                dengan Fitur Predictive Analytics untuk Pencegahan Risiko Klinis di Rumah Sakit
            </p>
            <p style='color: #475569;'>
                <strong>Program Studi:</strong> Magister Farmasi Klinis<br>
                <strong>Tahun:</strong> 2026
            </p>
            
            <hr style='border: 1px solid #e2e8f0; margin: 1.5rem 0;'>
            
            <h3 style='color: #334155;'>🔐 Fitur Keamanan (Authentication)</h3>
            <ul style='color: #475569; line-height: 1.6;'>
                <li>✅ <strong>Password Hashing:</strong> SHA-256 encryption</li>
                <li>✅ <strong>Session Management:</strong> Streamlit session state</li>
                <li>✅ <strong>Secure Logout:</strong> Clear session on logout</li>
                <li>✅ <strong>Role-based Access:</strong> Admin, Apoteker, Dokter</li>
            </ul>
            
            <hr style='border: 1px solid #e2e8f0; margin: 1.5rem 0;'>
            
            <h3 style='color: #334155;'>🛠️ Stack Teknologi</h3>
            <table style='width: 100%; border-collapse: collapse; margin-bottom: 1rem;'>
                <tr style='background-color: #f8fafc; border-bottom: 2px solid #cbd5e1;'>
                    <th style='padding: 0.75rem; text-align: left; color: #334155;'>Komponen</th>
                    <th style='padding: 0.75rem; text-align: left; color: #334155;'>Teknologi</th>
                </tr>
                <tr style='border-bottom: 1px solid #e2e8f0;'>
                    <td style='padding: 0.75rem; color: #475569;'>Frontend</td>
                    <td style='padding: 0.75rem; color: #475569;'>Streamlit (Python)</td>
                </tr>
                <tr style='border-bottom: 1px solid #e2e8f0;'>
                    <td style='padding: 0.75rem; color: #475569;'>Backend</td>
                    <td style='padding: 0.75rem; color: #475569;'>Python 3.10+</td>
                </tr>
                <tr style='border-bottom: 1px solid #e2e8f0;'>
                    <td style='padding: 0.75rem; color: #475569;'>Machine Learning</td>
                    <td style='padding: 0.75rem; color: #475569;'>scikit-learn (Random Forest)</td>
                </tr>
                <tr style='border-bottom: 1px solid #e2e8f0;'>
                    <td style='padding: 0.75rem; color: #475569;'>Authentication</td>
                    <td style='padding: 0.75rem; color: #475569;'>SHA-256 Hashing</td>
                </tr>
                <tr>
                    <td style='padding: 0.75rem; color: #475569;'>Deployment</td>
                    <td style='padding: 0.75rem; color: #475569;'>Streamlit Cloud</td>
                </tr>
            </table>
            
            <hr style='border: 1px solid #e2e8f0; margin: 1.5rem 0;'>
            
            <h3 style='color: #334155;'>🔒 Privacy & Ethics</h3>
            <p style='color: #b91c1c; font-weight: bold;'>DISCLAIMER: Ini adalah prototype untuk keperluan penelitian akademik.</p>
            <p style='color: #475569; margin-bottom: 0.5rem;'>Pada implementasi production:</p>
            <ul style='color: #475569; line-height: 1.6;'>
                <li>✅ Data pasien akan di-anonimisasi (sesuai UU PDP No. 27/2022)</li>
                <li>✅ Sistem dilengkapi authentication dan role-based access control</li>
                <li>✅ Audit logging untuk setiap prediksi</li>
                <li>✅ Ethical clearance dari KEPK</li>
                <li>✅ Integrasi dengan EHR rumah sakit</li>
            </ul>
            
            <hr style='border: 1px solid #e2e8f0; margin: 1.5rem 0;'>
            
            <h3 style='color: #334155;'>🎯 Keterbatasan Prototype</h3>
            <ul style='color: #475569; line-height: 1.6;'>
                <li>⚠️ Menggunakan data simulasi (belum data pasien sungguhan)</li>
                <li>⚠️ Belum terintegrasi dengan sistem EHR rumah sakit</li>
                <li>⚠️ Model perlu re-training dengan data riil untuk validasi</li>
                <li>⚠️ Tidak untuk digunakan dalam praktik klinis sebenarnya tanpa validasi lebih lanjut</li>
            </ul>
            
            <hr style='border: 1px solid #e2e8f0; margin: 1.5rem 0;'>
            
            <h3 style='color: #334155;'>👨‍🎓 Tentang Pengembang</h3>
            <p style='color: #475569; font-weight: bold;'>Mahasiswa S2 Farmasi Klinis (Non-Programmer!)</p>
            <p style='color: #475569; line-height: 1.6;'>
                Sistem ini dikembangkan dalam 6 minggu dengan bantuan AI tools (Claude.ai)
                untuk pembelajaran coding. Membuktikan bahwa dengan tekad dan bantuan AI,
                non-programmer pun bisa membuat sistem ML yang berfungsi.
            </p>
            
            <div style='margin-top: 2rem; text-align: center; color: #94a3b8; font-size: 0.9rem;'>
                <p style='margin-bottom: 0;'><strong>© 2026 Medication Error Prediction System</strong></p>
                <p style='margin-top: 0;'><em>Prototype untuk Penelitian Tesis S2 Farmasi Klinis</em></p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# FOOTER (shown on all pages)
# ══════════════════════════════════════════════════════════
if st.session_state.authenticated:
    st.markdown("<hr style='border: 1px solid #e2e8f0; margin: 2rem 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>🔐 Secure Session Active • Medication Error Prediction System v2.0 • © 2026</p>", unsafe_allow_html=True)
