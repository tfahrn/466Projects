import numpy as np
from sklearn.cluster import DBSCAN 
from sklearn import metrics


def get_data():
    data = np.loadtxt(open('data/encoded_data.csv', "rb"), delimiter=",")

    return data


def main():
    data = get_data()
    sse = {}

    for k in range(1, 7):
        db = DBSCAN(min_samples=k).fit(data)
        labels = db.labels_

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        sil_coeff = metrics.silhouette_score(data,  labels, metric='euclidean')
        print("Minimum number of samples in neighborhood to be consider a core point: {}".format(k))
        print("Number of clusters: {}".format(num_clusters))
        print("Silhouette Coefficient: {}\n".format(sil_coeff))




    

if __name__ == '__main__':
    main()
