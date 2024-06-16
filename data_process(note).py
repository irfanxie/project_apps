import pandas as pd
import numpy as np
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/data.csv')
df.head()

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
    r'[1-2][1-2][1-2]': 'At Risk - Low Value',
    r'[1-2][1-2][3-5]': 'At Risk - Mid Value',
    r'[1-2][3-5][1-2]': 'Potential Loyalist - Low Value',
    r'[1-2][3-5][3-5]': 'Potential Loyalist - Mid Value',
    r'[3-5][1-2][1-2]': 'Loyal - Low Value',
    r'[3-5][1-2][3-5]': 'Loyal - Mid Value',
    r'[3-5][3-5][1-2]': 'Champions - Low Value',
    r'[3-5][3-5][3-5]': 'Champions - Mid Value'
}

rfm['Segment'] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str) + rfm['MonetaryScore'].astype(str)
rfm['Segment'] = rfm['Segment'].replace(seg_map, regex=True)
rfm

"""#K-Means"""

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# Dataframe
df = pd.concat([recency_df, freq_df, monetary_df], axis=1)

# Membuat boxplot
plt.figure(figsize=(10, 6))

plt.subplot(1, 3, 1)
sns.boxplot(y=df['Recency'])
plt.title('Recency')

plt.subplot(1, 3, 2)
sns.boxplot(y=df['Frequency'])
plt.title('Frequency')

plt.subplot(1, 3, 3)
sns.boxplot(y=df['Monetary'])
plt.title('Monetary')

plt.tight_layout()
plt.show()

scaler = StandardScaler()
rfm_normalized = rfm[['Monetary','Frequency','Recency']]
rfm_normalized

rfm_normalized = scaler.fit_transform(rfm_normalized)
rfm_normalized = pd.DataFrame(rfm_normalized)
rfm_normalized

rfm_normalized = rfm_normalized.rename(columns={0: 'Monetary', 1: 'Frequency', 2: 'Recency'})
rfm_normalized.dtypes

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Membuat boxplot
plt.figure(figsize=(10, 6))

plt.subplot(1, 3, 1)
sns.boxplot(y=rfm_normalized['Monetary'])
plt.title('Recency')

plt.subplot(1, 3, 2)
sns.boxplot(y=rfm_normalized['Frequency'])
plt.title('Frequency')

plt.subplot(1, 3, 3)
sns.boxplot(y=rfm_normalized['Recency'])
plt.title('Monetary')

plt.tight_layout()
plt.show()

kmeans = KMeans (n_clusters=4, max_iter=50)
kmeans. fit (rfm_normalized)

inertia = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
  kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
  kmeans. fit(rfm_normalized)
  inertia.append(kmeans.inertia_)

plt.plot(range_n_clusters, inertia, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

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
rfm_normalized. loc[:, 'Nama'] = rfm ['Nama']
rfm_normalized

rfm_normalized['cluster'] = kmeans.labels_
rfm_normalized

sns.boxplot(x='cluster', y='Monetary', data=rfm_normalized)

sns.boxplot(x='cluster', y='Frequency', data=rfm_normalized)

sns.boxplot(x='cluster', y='Recency', data=rfm_normalized)

rfm_normalized

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams["figure.figsize"] = (25, 25)
fig = plt.figure(1)
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
ax.scatter(rfm_normalized['Frequency'], rfm_normalized['Recency'], rfm_normalized['Monetary'],
           c=rfm_normalized['cluster'],
           s=200,
           cmap='spring',
           alpha=0.5,
           edgecolor='darkgrey')

ax.set_xlabel('Frequency', fontsize=16)
ax.set_ylabel('Recency', fontsize=16)
ax.set_zlabel('Monetary', fontsize=16)

plt.show()

# Memvisualisasikan hasil klasterisasi
plt.scatter(rfm_normalized['Recency'], rfm_normalized['Monetary'],c = rfm_normalized['cluster'], s=200, cmap='viridis')

# Menampilkan pusat klaster
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)

plt.xlabel('Recency')
plt.ylabel('Monetary')
plt.title('KMeans Clustering of Recency and Monetary')
plt.show()

# Memvisualisasikan hasil klasterisasi
plt.scatter(rfm_normalized['Frequency'], rfm_normalized['Monetary'],c = rfm_normalized['cluster'], s=200, cmap='viridis')

# Menampilkan pusat klaster
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)

plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.title('KMeans Clustering of Recency and Monetary')
plt.show()

# Memvisualisasikan hasil klasterisasi
plt.scatter(rfm_normalized['Frequency'], rfm_normalized['Recency'],c = rfm_normalized['cluster'], s=200, cmap='viridis')

# Menampilkan pusat klaster
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)

plt.xlabel('Frequency')
plt.ylabel('Recency')
plt.title('KMeans Clustering of Recency and Monetary')
plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Ukuran plot
plt.rcParams["figure.figsize"] = (25, 25)

# Inisialisasi figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
scatter = ax.scatter(rfm_normalized['Frequency'],
                     rfm_normalized['Recency'],
                     rfm_normalized['Monetary'],
                     c=rfm_normalized['cluster'],
                     cmap='spring',
                     s=200,
                     alpha=0.5,
                     edgecolor='darkgrey')

# Label sumbu
ax.set_xlabel('Frequency', fontsize=16)
ax.set_ylabel('Recency', fontsize=16)
ax.set_zlabel('Monetary', fontsize=16)

# Tambahkan colorbar
colorbar = fig.colorbar(scatter, ax=ax)
colorbar.set_label('Cluster', fontsize=16)

plt.show()