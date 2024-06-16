from flask import Flask, render_template, request, redirect, flash, jsonify
import csv
import threading
import matplotlib.pyplot as plt
from data_process import generate_scatter_plot, generate_kmeans3d_plot, generate_pie_plot, generate_trend_plot, get_summary,tabel_data_rfm, top_10
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'secret_key'

# Path file CSV
csv_file = 'data/data.csv'

plt.switch_backend('Agg')

# Dekorator untuk menjalankan fungsi Matplotlib di utas utama
def generate_plot_async(func):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
    return wrapper

# Fungsi untuk mengambil nilai counter terakhir dari file CSV
def get_last_id():
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        if not df.empty:
            last_id = df['id'].max()
            if pd.notna(last_id):
                return int(last_id)
    return 0

def get_last_id_manual():
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        data = list(reader)
        if data:
            last_id = int(data[-1]['id'])
        else:
            last_id = 0
    return last_id

counter = get_last_id()
counter2 = get_last_id_manual()

@app.route('/')
def index():
    # Baca data dari file CSV
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        data = list(reader)
    return render_template('tables.html', data=data)

@app.route('/dashboard')
def dashboard(): 
    # Menjalankan fungsi Matplotlib di utas utama
    scatter_data = generate_scatter_plot()
    pie_data = generate_pie_plot()
    kmeans3d_data = generate_kmeans3d_plot()
    trend_data = generate_trend_plot()
    total_barang_dibeli, total_belanja, total_transaksi = get_summary()
    data = tabel_data_rfm().to_dict(orient='records')
    top10 = top_10().to_dict(orient='records')

    return render_template('dashboard.html', data=data, top10=top10, scatter_data=scatter_data, pie_data=pie_data, 
                           kmeans3d_data=kmeans3d_data, trend_data=trend_data, total_barang_dibeli=total_barang_dibeli,
                           total_belanja=total_belanja, total_transaksi=total_transaksi)


# # Fungsi untuk membaca DataFrame RFM dari file CSV atau sumber data lainnya
# def read_rfm_data():
#     # Misalnya, baca data dari file CSV
#     rfm_data = pd.read_csv('data/data_rfm.csv')  # Ubah sesuai dengan nama file Anda
#     return rfm_data

# @app.route('/get_data')
# def get_data():
#     # Membaca DataFrame RFM
#     rfm_df = read_rfm_data()
#     # Mengirimkan data JSON sebagai respons
#     return rfm_df[['recency', 'monetary']].to_json(orient='records')

@app.route('/add_data', methods=['POST'])
def add_data():
    global counter
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect('/')
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect('/')
    
    if file and file.filename.endswith('.csv'):
        file.save(file.filename)  # Save the uploaded file to the current directory
        
        # Load uploaded CSV into a pandas DataFrame
        df = pd.read_csv(file.filename)
        
        # Assign new IDs based on the last ID in the existing CSV file
        df['id'] = range(counter + 1, counter + 1 + len(df))
        
        # Append DataFrame to existing CSV file
        df.to_csv(csv_file, mode='a', header=False, index=False)
        
        counter += len(df)  # Update the counter
        
        os.remove(file.filename)  # Remove the uploaded file after processing
        
        flash('Data berhasil ditambahkan', 'success')
    else:
        flash('Invalid file type. Please upload a CSV file.', 'danger')
    
    return redirect('/')

@app.route('/add_data_manual', methods=['POST'])
def add_data_manual():
    global counter2
    # Ambil data dari formulir
    name = request.form['name']
    product = request.form['product']
    total_barang = request.form['total_barang']
    total_belanja = request.form['total_belanja']
    frekuensi_pembelian = request.form['frekuensi_pembelian']
    tanggal_terakhir_pembelian = request.form['tanggal_terakhir_pembelian']

    # Tulis data baru ke dalam file CSV dengan ID auto-increment
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        counter2 += 1  # Increment counter
        writer.writerow([counter2, name, product, total_barang, total_belanja, frekuensi_pembelian, tanggal_terakhir_pembelian])
    
    # Set pesan flash untuk memberi tanda bahwa data berhasil dimasukkan
    flash('Data berhasil ditambahkan', 'success')
    
    # Redirect kembali ke halaman utama setelah menambahkan data
    return redirect('/')

@app.route('/delete_data', methods=['POST'])
def delete_data():
    # Ambil ID data yang akan dihapus
    id_to_delete = request.form['id']

    # Baca data dari file CSV
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    # Hapus data berdasarkan ID
    new_data = [row for row in data if row['id'] != id_to_delete]

    # Tulis kembali data yang telah dihapus ke dalam file CSV
    with open(csv_file, mode='w', newline='') as file:
        fieldnames = ['id', 'nama', 'product', 'total_barang_yang_dibeli', 'total_belanja', 'frekuensi_pembelian', 'tanggal_terakhir_pembelian']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in new_data:
            writer.writerow(row)

    # Set pesan flash untuk memberi tanda bahwa data berhasil dihapus
    flash('Data berhasil dihapus', 'danger')

    # Redirect kembali ke halaman utama setelah menghapus data
    return redirect('/')  


@app.route('/update_data', methods=['POST'])
def update_data():
    # Ambil data dari formulir pembaruan data
    id = request.form['id']
    nama = request.form['nama']
    product = request.form['product']
    total_barang = request.form['total_barang']
    total_belanja = request.form['total_belanja']
    frekuensi_pembelian = request.form['frekuensi_pembelian']
    tanggal_terakhir_pembelian = request.form['tanggal_terakhir_pembelian']

    # Baca data dari file CSV
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    # Perbarui data yang sesuai
    for row in data:
        if row['id'] == id:
            row['nama'] = nama
            row['product'] = product
            row['total_barang_yang_dibeli'] = total_barang
            row['total_belanja'] = total_belanja
            row['frekuensi_pembelian'] = frekuensi_pembelian
            row['tanggal_terakhir_pembelian'] = tanggal_terakhir_pembelian
            break

    # Tulis kembali data yang telah diperbarui ke dalam file CSV
    with open(csv_file, mode='w', newline='') as file:
        fieldnames = ['id', 'nama', 'product', 'total_barang_yang_dibeli', 'total_belanja', 'frekuensi_pembelian', 'tanggal_terakhir_pembelian']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    # Set pesan flash untuk memberi tanda bahwa data berhasil diperbarui
    flash('Data berhasil diperbarui', 'info')

    # Redirect kembali ke halaman utama setelah memperbarui data
    return redirect('/')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
