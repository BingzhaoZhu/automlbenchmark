import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import autogluon.text

# def get_even_clusters(self, idx):
#     print(len(idx))
#     cluster_size = self.batch_size
#     num_clusters = int(np.ceil(len(idx) / cluster_size))
#     res = []
#     X = self.data_source.iloc[idx]
#     if len(idx) > cluster_size:
#         kmeans = KMeans(num_clusters).fit(X)
#         for i in range(num_clusters):
#             idx_ = idx[np.where(kmeans.labels_ == i)]
#             res += self.get_even_clusters(idx_)
#     else:
#         res.append(idx)
#     print(len(res))
#     return res

def get_even_clusters(X, cluster_size):
    n_clusters = int(np.ceil(len(X) / cluster_size))
    kmeans = KMeans(n_clusters)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    centers = centers.reshape(-1, 1, X.shape[-1]).repeat(cluster_size, 1).reshape(-1, X.shape[-1])
    distance_matrix = cdist(X, centers)
    clusters = linear_sum_assignment(distance_matrix)[1] // cluster_size
    return clusters

if __name__ == "__main__":
    x = np.random.randn(40000, 100)
    cluster = get_even_clusters(x, 128)
    print(len(np.where(cluster == 0)[0]))
    print(len(np.where(cluster == 1)[0]))
    print(len(np.where(cluster == 2)[0]))
    print(len(np.where(cluster == 3)[0]))
    print(cluster)
