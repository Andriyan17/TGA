import streamlit as st

def new_line(n=1):
    for i in range(n):
        st.write("\n")

# Fungsi untuk halaman "App Overview"
def app_overview():
   # Dataframe selection
    st.markdown("<h1 align='center'> <b>Analisa Metode Support Vector Machnine dan Linear Regression dalam Memprediksi Jumlah Kunjungan WIsatawan Mancanegara di Sumatera Utara</b></h1>", unsafe_allow_html=True)
    new_line(1)
    st.markdown("Selamat datang! Aplikasi ini dirancang untuk menampilkan secara langsung bagaimana kinerja 2 algoritma yang berbeda dalam memprediksi jumlah kunjungan wisatawan mancanegara di sumatera utara", unsafe_allow_html=True)
    
    st.divider()
    
    # Overview
    st.markdown("""
    Dalam proses membangun model prediksi, ada serangkaian langkah yang harus diikuti. Berikut ini adalah langkah-langkah utama dalam proses Machine Learning:
    
    - **📦 Pengumpulan Data**: Proses pengumpulan data dilakukan dengan cara scrapping melalui salah satu akun admin media fajarsumbar.com, disimpan dalam bentuk table excel.<br> <br>
    - **⚙️ Pra-pemrosesan Data**: Proses mengolah data ke dalam bentuk yang sesuai untuk analisis. Ini termasuk melakukan uji stasioneritas, mengidentifikasi dan menangani outlier dll.<br> <br>
    - **✂️ Pembagian Data**: Proses membagi data menjadi set pelatihan dan pengujian. Set pelatihan digunakan untuk melatih model dan set pengujian digunakan untuk mengevaluasi model.<br> <br>
    - **🧠 Membangun Model Pembelajaran Mesin**: Model yang digunakan pada aplikasi ini adalah Autoregressive Integrated Moving Average (ARIMA).<br> <br>
    - **⚖️ Evaluasi Model Pembelajaran Mesin**: Proses mengevaluasi model prediksi dengan menggunakan metrik seperti Mean Absolute Percentage Error (MAPE) dan Root Mean Squared Error (RMSE).<br> <br>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Pada bagian membangun model, pengguna memasukkan nilai masing-masing hiperparameter ARIMA. Hiperparameter adalah variabel yang secara signifikan mempengaruhi proses pelatihan model:
  
    - **Autoregressive Integrated Moving Average (ARIMA)**:
        - **p (Auto-regressive order)**: Jumlah lag observasi yang dimasukkan ke dalam model (nilai integer).<br> <br>
        - **d (Differencing order)**: Jumlah kali perbedaan yang diperlukan untuk menjadikan seri waktu stasioner (nilai integer).<br> <br>
        - **q (Moving average order)**: Ukuran jendela rata-rata bergerak untuk model (nilai integer).<br> <br>
    """, unsafe_allow_html=True)
    new_line()
    
    # Source Code
    new_line()
    st.header("📂 Source Code")
    st.markdown("Untuk pengembangan aplikasi ini, source code tersedia di [**GitHub**](https://github.com/scotch01/forecasting_ARIMA). Jangan ragu untuk berkontribusi, memberikan feedback, atau menyesuaikan aplikasi agar sesuai dengan kebutuhan Anda.", unsafe_allow_html=True)
    new_line()
    
    # Contributors
    st.header("👤 Kontributor")
    st.markdown("Aplikasi dibuat untuk kebutuhan tugas akhir/skripsi, **Ansharulhaq Aminsyah** (200170191) .", unsafe_allow_html=True)
    new_line()
    
    st.markdown("""Jika Anda memiliki pertanyaan atau saran, jangan ragu untuk menghubungi **ansharulhaqaminsyah08@gmail.com**. Kami siap membantu!

**Connect with us on social media:** 

<a href="https://www.linkedin.com/in/ansharulhaq-aminsyah/" target="_blank">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQe0adDoUGWVD3jGzfT8grK5Uhw0dLXSk3OWJwZaXI-t95suRZQ-wPF7-Az6KurXDVktV4&usqp=CAU" alt="LinkedIn" width="80" height="80" style="border-radius: 25%;">
</a>              
<a href="https://github.com/scotch01" target="_blank">
  <img src="https://seeklogo.com/images/G/github-logo-5F384D0265-seeklogo.com.png" alt="GitHub" width="80" height="80" style="border-radius: 25%;">
</a>

<br>
<br>

Kami menantikan kabar dari Anda dan mendukung perjalanan Anda dalam pembelajaran mesin!
    """, unsafe_allow_html=True)

# Fungsi utama aplikasi
def main():
    app_overview()

if __name__ == "__main__":
    main()
