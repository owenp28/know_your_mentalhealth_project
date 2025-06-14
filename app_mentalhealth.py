import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Konfigurasi halaman dengan tema yang konsisten
st.set_page_config(
    page_title="Know Your Mental Health - Analisis Kesehatan Mental Mahasiswa",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk memperbaiki font dan warna (menggunakan 3-color rule: primary, secondary, accent)
st.markdown("""
<style>
    /* Warna utama: #4B0082 (Indigo) */
    /* Warna sekunder: #6A5ACD (Slate Blue) */
    /* Warna aksen: #9370DB (Medium Purple) */
    
    /* Font yang mudah dibaca */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    
    h1, h2, h3 {
        color: #4B0082;
        font-weight: 600;
    }
    
    .stButton>button {
        background-color: #4B0082;
        color: white;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        background-color: #6A5ACD;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Memperbaiki simetri input */
    div[data-testid="stNumberInput"] {
        width: 100%;
    }
    
    /* Memperbaiki tampilan label */
    label {
        font-weight: 500;
        color: #4B0082;
    }
    
    /* Memperbaiki tampilan progress bar */
    div.stProgress > div > div {
        background-color: #9370DB;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
        with open("best_model_xgb.pkl", "rb") as f:
            xgb, accuracy_xgb, prec_xgb, rec_xgb, f1_xgb = pickle.load(f)
            return xgb, accuracy_xgb, "XGBoost", prec_xgb, rec_xgb, f1_xgb
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None, None, None, None, None, None

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("Student Lifestyle and Mental Health.csv")
        return df
    except FileNotFoundError:
        st.error("File dataset tidak ditemukan.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat dataset: {e}")
        return None

model, model_accuracy, model_name, precision, recall, f1 = load_model()
dataset = load_dataset()

def preprocess_and_predict(study_hours, extracurricular_hours, sleep_hours, social_hours, 
                          physical_activity_hours, gender_val):
    if model is None:
        return None, None

    # Sesuaikan fitur dengan model yang mengharapkan 11 fitur
    features = [
        float(study_hours),
        float(extracurricular_hours),
        float(sleep_hours),
        float(social_hours),
        float(physical_activity_hours),
        float(gender_val),
        0.0,  
        0.0,  
        0.0,  
        0.0,  
        0.0   
    ]
    
    try:
        input_array = np.array(features).reshape(1, -1)
        prediction = model.predict(input_array)
        probability = model.predict_proba(input_array)

        # Rule-based logic untuk override prediksi jika diperlukan
        stress_label = None
        if (0 <= study_hours <= 5.3 and 2 <= extracurricular_hours <= 4 and 
            7.2 <= sleep_hours <= 10 and 2 <= social_hours <= 6 and 
            1.8 <= physical_activity_hours <= 7.6):
            stress_label = "Low"
        elif (5 <= study_hours <= 7.9 and 1.1 <= extracurricular_hours <= 3.9 and 
              5.1 <= sleep_hours <= 9.9 and 0.3 <= social_hours <= 5.8 and 
              0.2 <= physical_activity_hours <= 8.4):
            stress_label = "Moderate"
        elif (6 <= study_hours <= 10 and 0 <= extracurricular_hours <= 1.8 and 
              5.1 <= sleep_hours <= 9.9 and 0 <= social_hours <= 6 and 
              0 <= physical_activity_hours <= 8.4):
            stress_label = "High"

        if stress_label:
            return stress_label, probability

        return prediction[0], probability
    except Exception as e:
        print(f"Terjadi kesalahan saat prediksi: {e}")
        return None, None

def create_visualizations():
    if dataset is None:
        return

    st.subheader("📊 Visualisasi Data Kesehatan Mental Mahasiswa")

    # Menggunakan warna yang konsisten dengan tema UI
    palettes = ["Blues", "Purples", "RdPu"]
    
    visual_list = [
        ("Study Hours", "Grades", "Stress Level", palettes[0], "Nilai Akademik vs Jam Belajar per Tingkat Stres",
         "Semakin banyak jam belajar, nilai akademik cenderung meningkat, namun tingkat stres juga bisa meningkat jika jam belajar berlebihan."),
        ("Sleep Hours", "Grades", "Stress Level", palettes[1], "Nilai Akademik vs Jam Tidur per Tingkat Stres",
         "Jam tidur yang cukup berkorelasi dengan nilai akademik yang baik dan tingkat stres yang lebih rendah."),
        ("Physical Activity Hours", "Grades", "Stress Level", palettes[2], "Nilai Akademik vs Aktivitas Fisik per Tingkat Stres",
         "Aktivitas fisik yang rutin dapat membantu menurunkan tingkat stres dan menjaga performa akademik."),
        ("Social Hours", "Grades", "Stress Level", "mako", "Nilai Akademik vs Jam Sosialisasi per Tingkat Stres",
         "Sosialisasi yang seimbang penting untuk kesehatan mental, namun terlalu banyak dapat mengganggu waktu belajar."),
        ("Extracurricular Hours", "Grades", "Stress Level", "icefire", "Nilai Akademik vs Ekstrakurikuler per Tingkat Stres",
         "Kegiatan ekstrakurikuler dapat membantu mengurangi stres jika tidak berlebihan."),
        ("Study Hours", "Sleep Hours", "Stress Level", "Spectral", "Jam Tidur vs Jam Belajar per Tingkat Stres",
         "Keseimbangan antara jam belajar dan jam tidur penting untuk menjaga tingkat stres tetap rendah.")
    ]

    # Tampilkan visualisasi dalam 3 kolom untuk simetri
    cols = st.columns(3)
    
    for i, (x, y, hue, palette, title, analysis) in enumerate(visual_list):
        with cols[i % 3]:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=dataset, x=x, y=y, hue=hue, palette=palette, ax=ax)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel(x, fontsize=10)
            ax.set_ylabel(y, fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown(f"<div style='text-align:left; font-size:14px;'><b>Analisis:</b> {analysis}</div>", unsafe_allow_html=True)

# Header aplikasi dengan warna tema
st.markdown("""
<h1 style='text-align: center; color: #FF7F50;'>🧠 Know Your Mental Health 🧠</h1>
<p style='text-align: center; font-size: 18px; color: #D2691E;'>Aplikasi berbasis AI yang memberikan rekomendasi aktivitas positif untuk menjaga kesehatan mental.</p>
<p style='text-align: center; font-size: 16px;'>Silakan isi semua field pada sidebar di samping dengan input angka yang sesuai.</p>
""", unsafe_allow_html=True)

if model_accuracy:
    st.info(f"Model {model_name} digunakan dengan akurasi {model_accuracy:.2f}%, presisi {precision:.2f}%, recall {recall:.2f}%, dan F1-score {f1:.2f}%")

if not model and not model_accuracy:
    st.sidebar.error("Model tidak dapat dimuat. Fungsi prediksi tidak akan berjalan.")

# Sidebar untuk input dengan layout yang simetris
with st.sidebar:
    st.markdown("<h3 style='color: #FFFFFF;'>👤 Data Diri & Gaya Hidup</h3>", unsafe_allow_html=True)
    
    # Gunakan container untuk memastikan simetri
    with st.container():
        study_hours = st.number_input("Jam Belajar per Hari", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        extracurricular_hours = st.number_input("Jam Ekstrakurikuler per Hari", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
        sleep_hours = st.number_input("Jam Tidur per Hari", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        social_hours = st.number_input("Jam Sosialisasi per Hari", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
        physical_activity_hours = st.number_input("Jam Aktivitas Fisik per Hari", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
        
        st.markdown("<br>", unsafe_allow_html=True)
        gender_options = {"Laki-laki": 1, "Perempuan": 0}
        gender = st.selectbox("Jenis Kelamin", options=list(gender_options.keys()), index=0)
        gender_val = gender_options[gender]
    
    # Tombol prediksi dengan warna tema
    predict_button = st.button("🧠 Prediksi Tingkat Stres Saya!", use_container_width=True)

# Proses prediksi
if predict_button and model:
    st.markdown("---")
    
    with st.spinner('Menganalisis data Anda...'):
        prediction_result, prediction_proba = preprocess_and_predict(
            study_hours, extracurricular_hours, sleep_hours, social_hours, physical_activity_hours, gender_val
        )

    # Mapping hasil prediksi ke label teks
    stress_levels = ["Low", "Moderate", "High"]
    
    hasil_teks = None
    if prediction_result is not None:
        # Jika prediction_result sudah berupa string (dari rule-based logic)
        if isinstance(prediction_result, str):
            hasil_teks = prediction_result
        else:
            # Jika prediction_result berupa angka/indeks
            try:
                idx = int(prediction_result)
                hasil_teks = stress_levels[idx]
            except (ValueError, IndexError):
                hasil_teks = str(prediction_result)
    
    if hasil_teks:
        # Container untuk hasil prediksi
        result_container = st.container()
        with result_container:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # Tampilkan hasil dengan warna yang sesuai
                if hasil_teks == "High":
                    st.error("### Tingkat Stres Anda: HIGH")
                    progress_val = 0.9
                    color = "#FF5252"
                elif hasil_teks == "Moderate":
                    st.warning("### Tingkat Stres Anda: MODERATE")
                    progress_val = 0.5
                    color = "#FFC107"
                else:
                    st.success("### Tingkat Stres Anda: LOW")
                    progress_val = 0.2
                    color = "#4CAF50"
                
                st.progress(progress_val)
                
                # Tampilkan probabilitas jika tersedia
                if prediction_proba is not None:
                    if isinstance(prediction_result, str):
                        # Jika hasil dari rule-based, tampilkan probabilitas tertinggi
                        prob_val = max(prediction_proba[0])
                    else:
                        # Jika hasil dari model, tampilkan probabilitas kelas yang diprediksi
                        try:
                            idx = int(prediction_result)
                            prob_val = prediction_proba[0][idx]
                        except (ValueError, IndexError):
                            prob_val = 0.0
                    
                    st.markdown(f"<p style='text-align: center; font-size: 16px;'>Tingkat kepercayaan: <strong>{prob_val:.2f}</strong></p>", unsafe_allow_html=True)

        st.markdown("---")
        
        # Data pengguna dalam layout yang simetris
        st.subheader("📝 Data Anda")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown(f"""
            - **Jam Belajar:** {study_hours} jam/hari  
            - **Jam Ekstrakurikuler:** {extracurricular_hours} jam/hari  
            - **Jam Tidur:** {sleep_hours} jam/hari  
            """)

        with col_right:
            st.markdown(f"""
            - **Jam Sosialisasi:** {social_hours} jam/hari  
            - **Jam Aktivitas Fisik:** {physical_activity_hours} jam/hari  
            - **Jenis Kelamin:** {gender}
            """)

        st.markdown("---")
        
        # Saran dan analisis dengan styling yang konsisten
        st.subheader("💡 Saran & Analisis untuk Anda")

        penjelasan = []

        if study_hours > 10:
            penjelasan.append(f"📚 Anda belajar {study_hours:.1f} jam/hari. Terlalu banyak belajar dapat meningkatkan stres.")
        elif study_hours < 5:
            penjelasan.append(f"📚 Anda hanya belajar {study_hours:.1f} jam/hari. Meningkatkan waktu belajar bisa membantu.")

        if sleep_hours < 7:
            penjelasan.append("🌙 Anda tidur kurang dari 7 jam per hari. Kurang tidur meningkatkan stres.")
        elif sleep_hours > 10:
            penjelasan.append("🛌 Tidur lebih dari 10 jam bisa jadi tanda stres atau depresi.")

        if physical_activity_hours < 1:
            penjelasan.append("🏃‍♂️ Aktivitas fisik Anda rendah. Olahraga bisa membantu menurunkan stres.")

        if social_hours < 1:
            penjelasan.append("👥 Waktu sosialisasi Anda sangat rendah. Interaksi sosial penting untuk kesehatan mental.")
        elif social_hours > 6:
            penjelasan.append("👥 Sosialisasi Anda tinggi. Pastikan tidak mengganggu waktu belajar atau tidur.")

        total_hours = study_hours + extracurricular_hours + sleep_hours + social_hours + physical_activity_hours
        if total_hours > 24:
            penjelasan.append("⏰ Total aktivitas Anda melebihi 24 jam. Prioritaskan aktivitas harian.")

        # Tampilkan saran dalam card untuk tampilan yang lebih baik
        if penjelasan:
            for saran in penjelasan:
                st.markdown(f"""
                <div style="padding:10px; border-radius:5px; margin-bottom:10px; border-left:4px solid #6A5ACD;">
                    {saran}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("🎉 Pola hidup Anda tampaknya seimbang. Pertahankan keseimbangan ini!")

# Tampilkan visualisasi data
create_visualizations()

# Footer dengan styling yang konsisten
st.markdown("""
---
<div style='text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 5px;'>
    <p style='color: #4B0082; font-family: "AMASIS MT PRO BLACK"; text-transform: uppercase; font-weight: bold;'>EVERY SMALL STEP TOWARDS MENTAL WELL-BEING IS A BIG VICTORY. REMEMBER, YOU ARE NOT ALONE ON THIS JOURNEY.</p>
</div>
""", unsafe_allow_html=True)
