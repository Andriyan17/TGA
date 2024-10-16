import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import calendar
from data_cleaning import clean_data
from sklearn.metrics import mean_absolute_error

# Fungsi untuk mendapatkan nama bulan dan tahun berdasarkan bulan terakhir dan jumlah bulan prediksi
def get_month_year_labels(start_month, start_year, months_ahead):
    """Menghasilkan label bulan dan tahun berdasarkan input bulan dan jumlah bulan ke depan."""
    labels = []
    current_year = start_year
    current_month = start_month

    for _ in range(months_ahead):
        labels.append(f"{calendar.month_name[current_month]} {current_year}")
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1

    return labels

# Fungsi untuk menghitung MAPE
def calculate_mape(y_true, y_pred):
    """Menghitung Mean Absolute Percentage Error (MAPE)."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Fungsi utama untuk halaman Linear Regression
def linear_regression_page():
    st.title("Prediksi Kunjungan Wisatawan Menggunakan Linear Regression")

    # Deskripsi aplikasi
    st.write(
        """
        Aplikasi ini menggunakan **Linear Regression** untuk memprediksi kunjungan wisatawan
        berdasarkan data historis. Anda dapat mengunggah data dalam format Excel, mengatur
        jumlah bulan yang ingin diprediksi, dan mendapatkan hasil prediksi serta visualisasinya.
        """
    )

    # Upload file Excel
    st.header("Unggah Data Excel")
    uploaded_file = st.file_uploader("Pilih file Excel", type="xlsx")

    # Input berapa bulan ke depan yang akan diprediksi
    st.subheader("Pengaturan Prediksi")
    col1, col2 = st.columns(2)
    with col1:
        months_ahead = st.number_input("Jangka Waktu Prediksi (bulan)", min_value=1, max_value=60, value=12)
    with col2:
        st.write("")  # Spasi untuk keseimbangan layout

    if uploaded_file is not None:
        # Membersihkan dan memproses data
        excel_data = pd.read_excel(uploaded_file)
        cleaned_data = clean_data(excel_data)
        
        st.write("### Data Setelah Dibersihkan")
        st.dataframe(cleaned_data, height=300)

        # Linear Regression Model
        X = cleaned_data[['MonthNumber']]
        y = cleaned_data['Average'].str.replace(".", "").astype(int)  # Mengonversi ke integer

        # Variabel untuk prediksi
        last_month = 0  # Januari
        last_year = 2024
        
        # Model Linear Regression
        model = LinearRegression()
        model.fit(X, y)
        prediksi_historis = model.predict(X).round().astype(int)

        cleaned_data['Prediction'] = prediksi_historis

        # Menghitung MAPE untuk prediksi historis
        mape_historical = calculate_mape(y, prediksi_historis)
        st.success(f"Mean Absolute Percentage Error (MAPE) untuk Data Historis (Linear Regression): {mape_historical:.2f}%")

        # Visualisasi hasil prediksi Linear Regression
        st.subheader("Visualisasi Prediksi Linear Regression untuk Data Historis")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=cleaned_data['MonthNumber'], y=y, color='blue', label='Data Asli', ax=ax)
        sns.lineplot(x=cleaned_data['MonthNumber'], y=cleaned_data['Prediction'], color='red', label='Prediksi Linear Regression', ax=ax)
        plt.xticks(ticks=cleaned_data['MonthNumber'], labels=cleaned_data['Bulan'], rotation=45)
        plt.xlabel('Bulan')
        plt.ylabel('Rata-rata Kunjungan')
        plt.title('Prediksi Linear Regression Kunjungan Wisatawan')
        plt.grid(True)
        st.pyplot(fig)

        # Prediksi bulan ke depan sesuai input user
        future_months = np.array([last_month + i for i in range(1, months_ahead + 1)]).reshape(-1, 1)
        prediksi_future = model.predict(future_months).round().astype(int)

        # Dapatkan label bulan dan tahun yang sesuai
        future_labels = get_month_year_labels(last_month + 1, last_year, months_ahead)

        future_data = pd.DataFrame({
            'Prediksi': prediksi_future,
            'Bulan-Tahun': future_labels
        })

        st.subheader(f"Prediksi Kunjungan Wisatawan untuk {months_ahead} Bulan ke Depan")
        st.table(future_data[['Bulan-Tahun', 'Prediksi']].reset_index(drop=True))

        # Visualisasi prediksi Linear Regression untuk bulan ke depan
        st.subheader("Visualisasi Prediksi Linear Regression untuk Bulan Mendatang")
        fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
        ax_pred.plot(future_data['Bulan-Tahun'], future_data['Prediksi'], marker='o', color='red', label="Prediksi")
        plt.xticks(rotation=45)
        ax_pred.set_xlabel("Bulan-Tahun")
        ax_pred.set_ylabel("Prediksi Kunjungan")
        ax_pred.set_title(f"Prediksi Kunjungan {months_ahead} Bulan Mendatang")
        ax_pred.grid(True)
        st.pyplot(fig_pred)

if __name__ == "__main__":
    linear_regression_page()
