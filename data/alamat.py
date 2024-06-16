import pandas as pd

# Membaca file CSV
df = pd.read_csv('./datab.csv')

# Menampilkan nama kolom
print("Nama Kolom Sebelum Dihapus:")
print(df.columns)

# Menghapus kolom 'alamat' jika ada
if 'alamat ' in df.columns:
    df.drop(columns=['alamat '], inplace=True)

# Menyimpan dataframe yang sudah diubah
df.to_csv('data.csv', index=False)