import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import calendar
from data_cleaning import clean_data

# Fungsi untuk mendapatkan nama bulan dan tahun
def generate_month_year_labels(start_month, start_year, months_ahead):
    """Menghasilkan label bulan dan tahun berdasarkan input start month dan jumlah bulan ke depan."""
    month_year_labels = []
    current_year = start_year
    current_month = start_month

    for _ in range(months_ahead):
        month_year_labels.append(f"{calendar.month_name[current_month]} {current_year}")
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1

    return month_year_labels

# Fungsi untuk menghitung MAPE
def calculate_mape(true_values, predicted_values):
    """Menghitung Mean Absolute Percentage Error (MAPE)."""
    return np.mean(np.abs((true_values - predicted_values) / true_values)) * 100

# Fungsi untuk visualisasi hasil prediksi SVR
def plot_predictions(actual_values, predicted_values, month_labels, title, xlabel, ylabel):
    """Visualisasi prediksi menggunakan scatter plot dan line plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=month_labels, y=actual_values, color='blue', label='Data Aktual', ax=ax)
    sns.lineplot(x=month_labels, y=predicted_values, color='orange', label='Prediksi SVR', ax=ax)
    plt.xticks(rotation=45)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    return fig

# Fungsi utama halaman Support Vector Regression
def svr_page():
    # Header dengan gambar dan judul yang lebih estetis
    st.title("Prediksi Kunjungan Wisatawan Menggunakan Support Vector Regression (SVR)")
    
    # Info ringkas tentang aplikasi
    st.write(
        """
        Aplikasi ini menggunakan **Support Vector Regression (SVR)** untuk memprediksi kunjungan wisatawan 
        ke lokasi tertentu berdasarkan data historis. Anda dapat mengunggah data dalam format Excel,
        mengatur parameter model SVR, dan mendapatkan prediksi hingga beberapa bulan ke depan.
        """
    )

    # Upload file Excel
    st.header("Unggah Data Excel")
    uploaded_file = st.file_uploader("Pilih file Excel", type="xlsx")

    # Input berapa bulan ke depan yang akan diprediksi
    st.subheader("Pengaturan Prediksi")
    col1, col2 = st.columns(2)
    with col1:
        forecast_horizon = st.number_input("Jangka Waktu Prediksi (bulan)", min_value=1, max_value=60, value=12)
    with col2:
        st.write("")  # Spasi untuk keseimbangan layout
    
    # Input parameter SVR dengan penjelasan tooltips
    st.subheader("Parameter SVR")
    col1, col2, col3 = st.columns(3)
    with col1:
        param_c = st.text_input("Parameter C", "100", help="Parameter regularisasi model SVR")
    with col2:
        param_gamma = st.text_input("Parameter Gamma", "0.1", help="Koefisien kernel untuk model SVR")
    with col3:
        param_epsilon = st.text_input("Parameter Epsilon", "0.1", help="Parameter epsilon yang menentukan margin dalam SVR")

    # Konversi input ke float
    try:
        param_c = float(param_c)
        param_gamma = float(param_gamma)
        param_epsilon = float(param_epsilon)
    except ValueError:
        st.error("Pastikan parameter C, gamma, dan epsilon diisi dengan angka yang valid!")
        return

    if uploaded_file:
        # Membersihkan dan memproses data
        raw_data = pd.read_excel(uploaded_file)
        cleaned_data = clean_data(raw_data)

        st.write("### Data Setelah Dibersihkan")
        st.dataframe(cleaned_data, height=300)

        # Ekstraksi fitur dan target dari data
        features = cleaned_data[['MonthNumber']]
        target = cleaned_data['Average'].str.replace(".", "").astype(int)

        last_month = 0
        last_year = 2024

        # Standarisasi fitur dan target
        scaler_features = StandardScaler()
        scaler_target = StandardScaler()
        features_scaled = scaler_features.fit_transform(features)
        target_scaled = scaler_target.fit_transform(target.values.reshape(-1, 1)).flatten()

        # Model SVR dengan parameter yang diinputkan user
        svr_model = SVR(kernel='rbf', C=param_c, gamma=param_gamma, epsilon=param_epsilon)
        svr_model.fit(features_scaled, target_scaled)

        # Prediksi untuk data historis
        historical_predictions = scaler_target.inverse_transform(svr_model.predict(features_scaled).reshape(-1, 1)).flatten().round().astype(int)
        cleaned_data['Prediksi_SVR'] = historical_predictions

        # Menghitung MAPE untuk prediksi historis
        mape_historical = calculate_mape(target, historical_predictions)
        st.success(f"Mean Absolute Percentage Error (MAPE) untuk Data Historis (SVR): {mape_historical:.2f}%")

        # Visualisasi prediksi historis
        st.subheader("Visualisasi Prediksi SVR untuk Data Historis")
        fig_svr = plot_predictions(target, historical_predictions, cleaned_data['Bulan'], 
                                        'Prediksi SVR Kunjungan Wisatawan', 'Bulan', 'Rata-rata Kunjungan')
        st.pyplot(fig_svr)

        # Prediksi untuk bulan ke depan
        future_month_numbers = np.array([last_month + i for i in range(1, forecast_horizon + 1)]).reshape(-1, 1)
        future_month_numbers_scaled = scaler_features.transform(future_month_numbers)
        future_predictions = scaler_target.inverse_transform(svr_model.predict(future_month_numbers_scaled).reshape(-1, 1)).flatten().round().astype(int)

        future_labels = generate_month_year_labels(last_month + 1, last_year, forecast_horizon)

        # Tabel prediksi bulan ke depan
        future_predictions_df = pd.DataFrame({
            'Prediksi SVR': future_predictions,
            'Bulan-Tahun': future_labels
        })

        st.subheader(f"Prediksi Kunjungan Wisatawan untuk {forecast_horizon} Bulan ke Depan (SVR)")
        st.table(future_predictions_df[['Bulan-Tahun', 'Prediksi SVR']].reset_index(drop=True))

        # Visualisasi prediksi bulan ke depan
        st.subheader("Visualisasi Prediksi (SVR) untuk Bulan Mendatang")
        fig_future_svr, ax_future_svr = plt.subplots(figsize=(10, 6))
        ax_future_svr.plot(future_predictions_df['Bulan-Tahun'], future_predictions_df['Prediksi SVR'], marker='o', color='orange', label="Prediksi SVR")
        plt.xticks(rotation=45)
        ax_future_svr.set_xlabel("Bulan-Tahun")
        ax_future_svr.set_ylabel("Prediksi Kunjungan")
        ax_future_svr.set_title(f"Prediksi Kunjungan {forecast_horizon} Bulan Mendatang (SVR)")
        ax_future_svr.grid(True)
        st.pyplot(fig_future_svr)

if __name__ == "__main__":
    svr_page()
