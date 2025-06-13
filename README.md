# 🧠 Know Your Mental Health 🧠

Aplikasi analisis kesehatan mental mahasiswa berbasis Machine Learning dengan Streamlit.

---

## 🚀 Cara Menjalankan Aplikasi (Offline)

### ⚙️ 1. Melakukan Install Python
Jika device Anda **belum memiliki Python**, ikuti langkah berikut:

1. Download Python dari situs Python secara resmi:  
   🔗 [https://www.python.org/downloads/](https://www.python.org/downloads/)

2. Jalankan file installer (`python-3.xx.x.exe`), lalu:
   - ✅ **Centang** opsi `Add Python to PATH`
   - Klik tombol **Install Now**

3. Setelah selesai, buka Command Prompt (CMD) dan jalankan:
   ```bash
   python --version
   pip --version
   ```
Jika keduanya menampilkan versi Python, maka Python sudah siap digunakan ✅

### 💻 2. Menjalankan Aplikasi
1. **Download file ZIP** dari repository ini  
2. **Extract ZIP** dan buka folder hasil ekstrak  
3. **Install library yang dibutuhkan**:
   - Buka Command Prompt di dalam folder
   - Jalankan perintah berikut:
     ```bash
     pip install -r requirements.txt
     ```
4. **Jalankan aplikasi**:
   ```bash
   streamlit run streamlit.py
   ```
   Aplikasi akan otomatis terbuka di browser local pada device Anda.

> ⚠️ Jika aplikasi tidak bisa dijalankan, pastikan telah menginstall semua library yang dibutuhkan (streamlit, pandas, matplotlib, seaborn, scikit-learn).

---

## 📊 Fitur Aplikasi

- Prediksi tingkat stres berdasarkan gaya hidup mahasiswa
- Visualisasi data dengan scatterplot untuk analisis mendalam
- Rekomendasi personal berdasarkan hasil prediksi
- Analisis hubungan antara jam belajar, tidur, aktivitas fisik, dan tingkat stres

---

## 📋 Dataset

Aplikasi ini menggunakan dataset "Student Lifestyle and Mental Health" yang berisi informasi tentang:
- Jam belajar
- Jam kegiatan ekstrakurikuler
- Jam tidur
- Jam sosialisasi
- Jam aktivitas fisik
- Jenis kelamin
- Tingkat stres
- Nilai akademik

---

## 🛠️ Teknologi yang Digunakan

- Python
- Streamlit
- Pandas
- Matplotlib & Seaborn
- Scikit-learn
- XGBoost/Random Forest/KNN