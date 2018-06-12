import numpy as np
from sklearn.cluster import AgglomerativeClustering 
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def get_data():
    data = np.loadtxt(open('data/encoded_data.csv', "rb"), delimiter=",")

    return data


def main():
    
    data = get_data()
    sse = {}
     
    for k in range(2, 10):
        agg = AgglomerativeClustering(n_clusters=k, linkage='average').fit(data)
        labels = agg.labels_
        #sse[k] = agg.inertia_
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
#main function used or analysis    
def main_p(k):
    data = get_data()
    agg = AgglomerativeClustering(n_clusters=k,linkage='average').fit(data)
    labels = agg.labels_
    return labels

if __name__ == '__main__':
    main()
