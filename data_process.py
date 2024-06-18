import pandas as pd
import numpy as np
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns


def tabel_data_rfm():
    df = pd.read_csv('data/data.csv')

    df['tanggal_terakhir_pembelian'] = pd.to_datetime(df['tanggal_terakhir_pembelian'])

    """**RFM**"""

    # Setting today's date
    today_date = dt.datetime(2024, 3, 14)

    # Recency Metric
    recency_df = (today_date - df.groupby("nama")["tanggal_terakhir_pembelian"].max()).dt.days.rename("Recency")

    # Frequency Metric
    freq_df = df.dropna(subset=["tanggal_terakhir_pembelian"]).groupby("nama").size().rename("Frequency")

    # Monetary Metric
    monetary_df = df.groupby("nama")["total_belanja"].sum().rename("Monetary")

    # Concatenating all metrics into RFM DataFrame
    df = pd.concat([recency_df, freq_df, monetary_df], axis=1)
    df.reset_index(inplace=True)

    # Assigning RFM DataFrame back to 'df'
    df = df.dropna(subset=['Recency'])

    df

    rfm = df
    rfm["RecencyScore"] = pd.qcut(rfm['Recency'], 5, labels = [5, 4, 3, 2, 1])
    rfm["FrequencyScore"] = pd.qcut(rfm['Frequency'].rank(method = "first"), 5, labels = [1,2,3,4,5])
    rfm["MonetaryScore"] = pd.qcut(rfm['Monetary'], 5, labels = [1,2,3,4,5])
    rfm["RFM_SCORE"] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str) + rfm['MonetaryScore'].astype(str)
    rfm

    seg_map = {
        r'[1-2][1-2][1-2]': 'Pendatang',
        r'[1-2][1-2][3-5]': 'Pendatang baru',
        r'[1-2][3-5][1-2]': 'Berlangganan',
        r'[1-2][3-5][3-5]': 'Berlangganan',
        r'[3-5][1-2][1-2]': 'Loyal',
        r'[3-5][1-2][3-5]': 'Loyal',
        r'[3-5][3-5][1-2]': 'Super Loyal',
        r'[3-5][3-5][3-5]': 'Super Loyal'
    }

    rfm['Segment'] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str) + rfm['MonetaryScore'].astype(str)
    rfm['Segment'] = rfm['Segment'].replace(seg_map, regex=True)
    return rfm

def top_10():
    data = tabel_data_rfm()
    top10 = data.nsmallest(10, 'Monetary')

    return top10


rfm = tabel_data_rfm()
scaler = StandardScaler()
rfm_normalized = rfm[['Monetary','Frequency','Recency']]

rfm_normalized = scaler.fit_transform(rfm_normalized)
rfm_normalized = pd.DataFrame(rfm_normalized)

rfm_normalized = rfm_normalized.rename(columns={0: 'monetary', 1: 'frequency', 2: 'recency'})
rfm_normalized.dtypes
rfm_normalized.to_csv('data/data_rfm.csv', index=False)

kmeans = KMeans (n_clusters=4, max_iter=50)
kmeans. fit (rfm_normalized)

inertia = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
  kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
  kmeans. fit(rfm_normalized)
  inertia.append(kmeans.inertia_)

def data_inertia():
   inersia = inertia
   return inersia

def data_clusters():
   klaster = range_n_clusters
   return klaster

def data_rfm():
   data_rfm = rfm_normalized
   return data_rfm

# plt.plot(range_n_clusters, inertia, marker='o')
# plt.xlabel('Number of clusters (K)')
# plt.ylabel('Inertia')
# plt.title('Elbow Method for Optimal K')
# # plt.show()

# K-Means
from sklearn.metrics import silhouette_score

for nuim_clustrers in range_n_clusters:
  #intialise kmeans
  kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
  kmeans.fit(rfm_normalized)

  cluster_labels = kmeans.labels_
  # silhouette score
  silhouette_avg = silhouette_score(rfm_normalized, cluster_labels)
  print("For n_clusters={0}, the silhouette score is {1}". format(num_clusters, silhouette_avg))

kmeans = KMeans(n_clusters=3, max_iter=50)
kmeans.fit(rfm_normalized)
KMeans (max_iter=50, n_clusters=3)
rfm_normalized. loc[:, 'nama'] = rfm ['nama']
rfm_normalized

rfm_normalized['cluster'] = kmeans.labels_
# rfm_normalized

# sns.boxplot(x='cluster', y='Monetary', data=rfm_normalized)

# sns.boxplot(x='cluster', y='Frequency', data=rfm_normalized)

# sns.boxplot(x='cluster', y='Recency', data=rfm_normalized)

# rfm_normalized

def generate_scatter_plot():
   # Memvisualisasikan hasil klasterisasi
    plt.figure(figsize=(14, 8))  
    plt.scatter(rfm_normalized['recency'], rfm_normalized['monetary'],c = rfm_normalized['cluster'], s=50, cmap='viridis')

    # Menampilkan pusat klaster
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)

    plt.xlabel('Recency')
    plt.ylabel('Monetary')
    plt.title('KMeans Clustering of Recency and Monetary')

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode the plot image to base64
    scatter_data = base64.b64encode(buffer.read()).decode('utf-8')

    return scatter_data

def generate_pie_plot():
    segment_counts = rfm['Segment'].value_counts()

    # Persiapkan data untuk pie chart
    segments = segment_counts.index
    segment_frequencies = segment_counts.values

    # Buat pie chart
    plt.figure(figsize=(6,4))  # atur ukuran gambar
    plt.pie(segment_frequencies, labels=segments, autopct='%1.1f%%', startangle=140)
    # plt.title('Customer Distribution by Segment')
    plt.axis('equal')  # membuat lingkaran
    
    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode the plot image to base64
    pie_data = base64.b64encode(buffer.read()).decode('utf-8')

    return pie_data

def generate_kmeans3d_plot():
   # Ukuran plot
    plt.rcParams["figure.figsize"] = (25, 25)

    # Inisialisasi figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    scatter = ax.scatter(rfm_normalized['frequency'],
                        rfm_normalized['recency'],
                        rfm_normalized['monetary'],
                        c=rfm_normalized['cluster'],
                        cmap='spring',
                        s=200,
                        alpha=0.5,
                        edgecolor='darkgrey')

    # Label sumbu
    ax.set_xlabel('Frequency', fontsize=16)
    ax.set_ylabel('Recency', fontsize=16)
    ax.set_zlabel('Monetary', fontsize=16)

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode the plot image to base64
    kmeans3d_data = base64.b64encode(buffer.read()).decode('utf-8')

    return kmeans3d_data

def generate_trend_plot():
    # Baca data dari file CSV
    data = pd.read_csv('data/data.csv')

    # Ubah kolom tanggal_terakhir_pembelian menjadi tipe data datetime
    data['tanggal_terakhir_pembelian'] = pd.to_datetime(data['tanggal_terakhir_pembelian'])

    # Konversi kolom tanggal_terakhir_pembelian menjadi bulan dan tahun
    data['bulan'] = data['tanggal_terakhir_pembelian'].dt.to_period('Y')

    # Kelompokkan data berdasarkan bulan dan tahun, dan hitung total penjualan
    sales_data = data.groupby('bulan')['total_belanja'].sum()

    # Buat plot garis
    plt.figure(figsize=(10, 6))
    plt.plot(sales_data.index.astype(str), sales_data.values, marker='o', linestyle='-')
    # plt.title('Trend Penjualan')
    plt.xlabel('bulan')
    plt.ylabel('Total Penjualan')
    # plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode the plot image to base64
    trend_data = base64.b64encode(buffer.read()).decode('utf-8')

    return trend_data


def get_summary():
    # Mendapatkan total barang yang dibeli
    data = pd.read_csv('data/data.csv')

    total_barang_dibeli = data['total_barang_yang_dibeli'].sum()
    
    # Mendapatkan total belanja
    total_belanja = data['total_belanja'].sum()
    total_belanja = "{:,.0f}".format(total_belanja)
    
    # Mendapatkan total transaksi
    total_transaksi = data.shape[0]  # Jumlah baris dalam DataFrame
    
    return total_barang_dibeli, total_belanja, total_transaksi



