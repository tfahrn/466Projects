import argparse 
import re
import pandas as pd
import random
import sys
import numpy as np
import itertools
from matplotlib import pyplot as plt

class Datum:

    def __init__(self, position, cluster):
        self.position = position
        self.cluster = cluster

class Cluster:

    def __init__(self, number, position):
        self.number = number
        self.position = position
        self.min_dist = sys.float_info.max
        self.max_dist = 0
        self.tot_dist = 0
        self.num_data = 0

def get_distance(datum, cluster):
    datum_position = np.asarray(datum.position)
    cluster_position = np.asarray(cluster.position)
    dist = np.linalg.norm(datum_position - cluster_position)

    return dist

def get_restrictions_vector(line):
    one_hot = re.split(',|\n', line)
    by_index = [i for i, value in enumerate(one_hot) if value == '1']

    return by_index

def get_data(file_name):
    with open(file_name) as f:
        restrictions = get_restrictions_vector(f.readline())
        df = pd.read_csv(f, header=None, skiprows=0, usecols=restrictions)
        df = (df - df.mean())/(df.std()) # normalize
        
        return df

def get_args():
    parser = argparse.ArgumentParser(description='k-means clustering')
    parser.add_argument('-f', '--filename', help='.csv data file', required=True)
    parser.add_argument('-k', '--k', help='number of clusters', required=True)

    return vars(parser.parse_args())

def init_clusters(k, data):
    clusters = []
    positions = []

    for i in range(k):
        # randomly select init position among data w/o replacement
        position = data[random.randint(0, len(data)-1)].position
        while position in positions:
            position = data[random.randint(0, len(data)-1)].position
        positions.append(position)
        clusters.append(Cluster(i, position))

    return clusters

def init_data(df):
    data = []

    for i, row in df.iterrows():
        position = [coord for coord in row]
        data.append(Datum(position, -1))

    return data 

def kmeans(data, clusters, stop_ratio):
    ratio_moved = 1

    while(ratio_moved > stop_ratio):
        num_changed = 0
        cluster_pos_totals = [[] for c in clusters] # LoL, sum of positions of data points per cluster 
        cluster_count_totals = [0 for c in clusters] # list of num data points per cluster

        for datum in data:
            min_dist = sys.float_info.max
            nearest = datum.cluster 
            for cluster in clusters:
                dist = get_distance(datum, cluster)
                if dist < min_dist:
                    min_dist = dist
                    nearest = cluster.number
                    
            # nearest cluster has changed
            if datum.cluster != nearest:
                num_changed += 1
                datum.cluster = nearest
            
            """
            if len(cluster_pos_totals) < datum.cluster + 1:
                cluster_pos_totals[datum.cluster] = None
            """
            cluster_pos_totals[datum.cluster] = \
                [sum(pos) for pos in
                itertools.zip_longest(datum.position, cluster_pos_totals[datum.cluster], fillvalue=0)]
            
            cluster_count_totals[datum.cluster] += 1

        for i, cluster in enumerate(clusters):
            mean_position = [coord/cluster_count_totals[i]
                for coord in cluster_pos_totals[i]]
            cluster.position = mean_position

        ratio_moved = num_changed/len(data)

    return data, clusters

def statistics(data, clusters):
    sse = 0

    for cluster in clusters:
        for datum in data:
            if datum.cluster == cluster.number:
                dist = get_distance(datum, cluster)
                if dist > cluster.max_dist:
                    cluster.max_dist = dist
                if dist < cluster.min_dist:
                    cluster.min_dist = dist
                cluster.tot_dist += dist
                cluster.num_data += 1
                sse += dist*dist

    return sse

def main():
    args = get_args()
    file_name = args['filename']
    k = int(args['k'])
    df = get_data(file_name)
    data = init_data(df)
    clusters = init_clusters(k, data)

    """
    ks = [k for k in range(1, min(20, len(data)-1))]
    sse_per_k = []
    max_sse = 0
    for k in ks:
        clusters = init_clusters(k, data)
        stop_ratio = 1/len(data)
        data, clusters = kmeans(data, clusters, stop_ratio)
        sse = statistics(data, clusters)
        sse_per_k.append(sse)
        if sse > max_sse:
            max_sse = sse

    plt.plot(ks, sse_per_k, 'bo')
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
    plt.axis([0, 20, 0, max_sse*(7/5)])
    plt.show()

    """
    stop_ratio = 1/len(data)
    data, clusters = kmeans(data, clusters, stop_ratio)

    sse = statistics(data, clusters)

    print("SSE:", sse)

    for cluster in clusters:
        print("\nCluster", cluster.number)
        print("Center:", [round(p, 2) for p in cluster.position])
        print("Max Dist. to Center:", round(cluster.max_dist, 2))
        print("Min Dist. to Center:", round(cluster.min_dist, 2))
        print("Avg Dist. to Center:", round(cluster.tot_dist/cluster.num_data, 2))
        print("Number of data in cluster:", cluster.num_data)
        print(min(4, cluster.num_data), "Points:")

        count = 0
        for datum in data:
            if datum.cluster == cluster.number:
                print([round(p, 2) for p in datum.position])
                count += 1
                if count > 3:
                    break;

main()
