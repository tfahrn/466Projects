import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def get_data():
    data = np.loadtxt(open('data/encoded_data.csv', "rb"), delimiter=",")

    return data

def main():
    
    data = get_data()
    sse = {}
    
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k).fit(data)
        labels = kmeans.labels_
        sse[k] = kmeans.inertia_
        sil_coeff = silhouette_score(data, labels, metric='euclidean')
        print("Silhouette Coefficient with {} clusters: {}".format(k, sil_coeff))

    """
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.title("k-means Clustering on Turkey Political Opinions")
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
    plt.show()
    """
def main_p(k):
    data = get_data()
    kmeans = KMeans(n_clusters=k).fit(data)
    clusters = kmeans.labels_
    return clusters
    
#if __name__ == '__main__':
   # main()
