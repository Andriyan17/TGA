import pandas as pd
import numpy as np

def clean_data(excel_data):
    # Mengambil data sesuai dengan baris dan kolom yang dibutuhkan
    cleaned_data = excel_data.iloc[2:14, 1:6].copy()
    
    # Menambahkan nama kolom
    cleaned_data.columns = ['2019', '2020', '2021', '2022', '2023']
    
    # Menambahkan kolom bulan
    cleaned_data['Bulan'] = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli',
                             'Agustus', 'September', 'Oktober', 'November', 'Desember']
    
    # Menyusun ulang kolom agar bulan di posisi pertama
    cleaned_data = cleaned_data[['Bulan', '2019', '2020', '2021', '2022', '2023']]
    
    # Mengonversi data menjadi numerik
    cleaned_data = cleaned_data.apply(pd.to_numeric, errors='ignore')
    
    # Menambahkan kolom 'MonthNumber' (untuk bulan dalam angka 1-12)
    cleaned_data['MonthNumber'] = np.arange(1, 13)
    
    # Menghitung rata-rata dari tahun 2019 hingga 2023 dan menyimpannya dalam kolom 'Average'
    cleaned_data['Average'] = cleaned_data[['2019', '2020', '2021', '2022', '2023']].mean(axis=1).round().astype(int)
    
    # Format data dengan pemisah ribuan menggunakan titik
    cleaned_data[['2019', '2020', '2021', '2022', '2023', 'Average']] = cleaned_data[['2019', '2020', '2021', '2022', '2023', 'Average']].applymap(lambda x: "{:,.0f}".format(x).replace(",", "."))
    
    return cleaned_data
