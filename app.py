import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import calendar

# Import fungsi clean_data dari file data_cleaning.py
from data_cleaning import clean_data

# Fungsi untuk mendapatkan nama bulan dan tahun berdasarkan bulan terakhir dan jumlah bulan prediksi
def get_month_year_labels(start_month, start_year, months_ahead):
    labels = []
    current_year = start_year
    current_month = start_month

    for i in range(months_ahead):
        # Tambahkan bulan dan tahun ke label
        labels.append(f"{calendar.month_name[current_month]} {current_year}")
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1

    return labels

# Fungsi untuk menghitung MAPE
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Main function for Streamlit app
def main():
    st.title("Prediksi Kunjungan Wisatawan di Sumatera Utara")
    
    # Sidebar untuk input data
    st.sidebar.header("Upload Data Excel")
    uploaded_file = st.sidebar.file_uploader("Pilih file Excel", type="xlsx")

    # Dropdown untuk memilih model (Linear Regression atau SVR)
    model_choice = st.sidebar.selectbox("Pilih Metode Prediksi", ["Linear Regression", "Support Vector Regression (SVR)"])

    # Input untuk memilih berapa bulan ke depan yang ingin diprediksi
    months_ahead = st.sidebar.number_input("Pilih berapa bulan ke depan yang ingin diprediksi", min_value=1, max_value=60, value=12)

    if uploaded_file is not None:
        # Memanggil fungsi clean_data untuk membersihkan dan memproses data
        excel_data = pd.read_excel(uploaded_file)
        cleaned_data = clean_data(excel_data)
        
        st.write("Data Setelah Dibersihkan:")
        st.table(cleaned_data)

        # Linear Regression Model
        X = cleaned_data[['MonthNumber']]
        y = cleaned_data['Average'].str.replace(".", "").astype(int)  # Mengonversi kembali ke integer untuk model prediksi

        # Ubah last_month menjadi 0 untuk memulai dari Januari
        last_month = 0  # Mengatur Januari sebagai bulan pertama untuk prediksi
        last_year = 2024  # Mengatur tahun mulai prediksi
        
        if model_choice == "Linear Regression":
            # Model Linear Regression
            model = LinearRegression()
            model.fit(X, y)
            prediksi_historis = model.predict(X).round().astype(int)  # Prediksi untuk data historis (2019-2023)

            cleaned_data['Prediction'] = prediksi_historis

            # Menghitung MAPE untuk prediksi historis
            mape_historical = calculate_mape(y, prediksi_historis)
            st.write(f"Mean Absolute Percentage Error (MAPE) untuk Data Historis (Linear Regression): {mape_historical:.2f}%")

            # Visualisasi hasil prediksi Linear Regression
            st.write("Prediksi Menggunakan Linear Regression:")
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
            future_labels = get_month_year_labels(last_month + 1, last_year, months_ahead)  # Mulai dari Januari

            future_data = pd.DataFrame({
                'Prediksi': prediksi_future,
                'Bulan-Tahun': future_labels
            })

            # Tampilkan prediksi untuk bulan-bulan ke depan tanpa menampilkan indeks
            st.write(f"Prediksi Kunjungan Wisatawan untuk {months_ahead} bulan ke depan:")
            st.table(future_data[['Bulan-Tahun', 'Prediksi']].reset_index(drop=True))  # Menghapus tampilan indeks
            
            # Menampilkan prediksi dalam bentuk line chart
            st.write("Visualisasi Prediksi:")
            fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
            ax_pred.plot(future_data['Bulan-Tahun'], future_data['Prediksi'], marker='o', color='red', label="Prediksi")
            plt.xticks(rotation=45)
            ax_pred.set_xlabel("Bulan-Tahun")
            ax_pred.set_ylabel("Prediksi Kunjungan")
            ax_pred.set_title(f"Prediksi Kunjungan {months_ahead} Bulan Mendatang")
            ax_pred.grid(True)
            st.pyplot(fig_pred)

        elif model_choice == "Support Vector Regression (SVR)":
            # Model SVR
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

            svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
            svr_model.fit(X_scaled, y_scaled)
            prediksi_historis = scaler_y.inverse_transform(svr_model.predict(X_scaled).reshape(-1, 1)).flatten().round().astype(int)

            cleaned_data['Prediction_SVR'] = prediksi_historis

            # Menghitung MAPE untuk prediksi historis SVR
            mape_historical_svr = calculate_mape(y, prediksi_historis)
            st.write(f"Mean Absolute Percentage Error (MAPE) untuk Data Historis (SVR): {mape_historical_svr:.2f}%")

            # Visualisasi hasil prediksi SVR
            st.write("Prediksi Menggunakan Support Vector Regression (SVR):")
            fig_svr, ax_svr = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=cleaned_data['MonthNumber'], y=y, color='blue', label='Data Asli', ax=ax_svr)
            sns.lineplot(x=cleaned_data['MonthNumber'], y=cleaned_data['Prediction_SVR'], color='orange', label='Prediksi SVR', ax=ax_svr)
            plt.xticks(ticks=cleaned_data['MonthNumber'], labels=cleaned_data['Bulan'], rotation=45)
            plt.xlabel('Bulan')
            plt.ylabel('Rata-rata Kunjungan')
            plt.title('Prediksi SVR Kunjungan Wisatawan')
            plt.grid(True)
            st.pyplot(fig_svr)
            
            # Prediksi bulan ke depan sesuai input user menggunakan SVR
            future_months = np.array([last_month + i for i in range(1, months_ahead + 1)]).reshape(-1, 1)
            future_months_scaled = scaler_X.transform(future_months)
            prediksi_future_svr = scaler_y.inverse_transform(svr_model.predict(future_months_scaled).reshape(-1, 1)).flatten().round().astype(int)

            # Dapatkan label bulan dan tahun yang sesuai
            future_labels = get_month_year_labels(last_month + 1, last_year, months_ahead)

            future_data_svr = pd.DataFrame({
                'Prediksi SVR': prediksi_future_svr,
                'Bulan-Tahun': future_labels
            })

            # Tampilkan prediksi untuk bulan-bulan ke depan tanpa menampilkan indeks
            st.write(f"Prediksi Kunjungan Wisatawan untuk {months_ahead} bulan ke depan (SVR):")
            st.table(future_data_svr[['Bulan-Tahun', 'Prediksi SVR']].reset_index(drop=True))  # Menghapus tampilan indeks

                    # Menampilkan prediksi dalam bentuk line chart untuk SVR
            st.write("Visualisasi Prediksi (SVR):")
            fig_pred_svr, ax_pred_svr = plt.subplots(figsize=(10, 6))
            ax_pred_svr.plot(future_data_svr['Bulan-Tahun'], future_data_svr['Prediksi SVR'], marker='o', color='orange', label="Prediksi SVR")
            plt.xticks(rotation=45)
            ax_pred_svr.set_xlabel("Bulan-Tahun")
            ax_pred_svr.set_ylabel("Prediksi Kunjungan")
            ax_pred_svr.set_title(f"Prediksi Kunjungan {months_ahead} Bulan Mendatang (SVR)")  # Perbaikan disini
            ax_pred_svr.grid(True)
            st.pyplot(fig_pred_svr)

if __name__ == "__main__":
    main()
