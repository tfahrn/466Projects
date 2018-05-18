import argparse 
import re
import pandas as pd
import random
import sys
import numpy as np
import itertools
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Datum:
    def __init__(self, position, label):
        self.position = position
        self.label = None

class Cluster:
    def __init__(self, number, position):
        self.number = number
        self.position = position
        self.min_dist = sys.float_info.max
        self.max_dist = 0
        self.tot_dist = 0
        self.num_data = 0

def get_distance(p, q):
    p_position = np.asarray(p.position)
    q_position = np.asarray(q.position)
    dist = np.linalg.norm(p_position - q_position)

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
    parser.add_argument('-e', '--epsilon', help='the radius within which DBSCAN searches', required=True)
    parser.add_argument('-n', '--num', help='minimum number of points to build cluster', required=True)

    return vars(parser.parse_args())

def init_data(df):
    data = []

    for i, row in df.iterrows():
        position = [coord for coord in row]
        data.append(Datum(position, None))

    return data 

def dbscan(data, eps, min_points):
    cluster_num = 0

    for datum in data:
        if datum.label == None:
            if expand(data, datum, cluster_num, eps, min_points):
                cluster_num += 1

    return data, cluster_num

def expand(data, point, cluster_num, eps, min_points):
    seeds = region_query(data, point, eps)

    if len(seeds) < min_points:
        point.label = 'Noise'
        return False
    else:
        point.label = cluster_num
        for seed in seeds:
            seed.label = cluster_num

        while len(seeds) > 0:
            seed = seeds[0]
            deep_seeds = region_query(data, seed, eps)

            if len(deep_seeds) >= min_points:
                for deep_seed in deep_seeds:
                    if deep_seed.label == None or deep_seed.label == 'Noise':
                        if deep_seed.label == None:
                            seeds.append(deep_seed)
                        deep_seed.label = cluster_num
            
            seeds = seeds[1:]

    return True

def region_query(data, point, eps):
    seeds = []
    for datum in data:
        if get_distance(point, datum) < eps:
            seeds.append(datum)

    return seeds

def statistics(data, clusters):
    sse = 0

    for cluster in clusters:
        for datum in data:
            if datum.label == cluster.number:
                dist = get_distance(datum, cluster)
                if dist > cluster.max_dist:
                    cluster.max_dist = dist
                if dist < cluster.min_dist:
                    cluster.min_dist = dist
                cluster.tot_dist += dist
                cluster.num_data += 1
                sse += dist*dist

    return sse

def init_clusters(data, num_clusters):
    clusters = [Cluster(i, None) for i in range(num_clusters)]

    cluster_pos_totals = [[] for c in clusters] # LoL, sum of positions of data points per cluster 
    cluster_count_totals = [0 for c in clusters] # list of num data points per cluster

    for datum in data:
        if datum.label == None or datum.label == 'Noise':
            continue
        cluster_pos_totals[datum.label] = \
            [sum(pos) for pos in
            itertools.zip_longest(datum.position, cluster_pos_totals[datum.label], fillvalue=0)]
        cluster_count_totals[datum.label] += 1

    for i, cluster in enumerate(clusters):
        mean_position = [coord/cluster_count_totals[i]
            for coord in cluster_pos_totals[i]]
        cluster.position = mean_position

    return clusters

def print_clusters(data, clusters):
    for cluster in clusters:
        if cluster.position == []:
            continue
        print("\nCluster", cluster.number)
        print("Center:", [round(p, 2) for p in cluster.position])
        print("Max Dist. to Center:", round(cluster.max_dist, 2))
        print("Min Dist. to Center:", round(cluster.min_dist, 2))
        print("Avg Dist. to Center:", round(cluster.tot_dist/cluster.num_data, 2))
        print("Number of data in cluster:", cluster.num_data)
        print(min(4, cluster.num_data), "Points:")

        count = 0
        for datum in data:
            if datum.label == cluster.number:
                print([round(p, 2) for p in datum.position])
                count += 1
                if count > 3:
                    break;

def print_outliers(data, clusters):
    num_outliers = 0
    count = 0

    for datum in data:
        if datum.label == 'Noise':
            num_outliers += 1

    print("\nNumber of outliers:", num_outliers)
    print(min(4, num_outliers), "Outliers:")
    for datum in data:
        if datum.label == 'Noise':
            print([round(p, 2) for p in datum.position])
            count += 1
            if count > 3:
                break;

def main():
    args = get_args()
    file_name = args['filename']
    eps = float(args['epsilon'])
    min_points = int(args['num'])

    df = get_data(file_name)
    data = init_data(df)
    data, num_clusters = dbscan(data, eps, min_points)
    clusters = init_clusters(data, num_clusters)
    sse = statistics(data, clusters)

    print("SSE:", sse)
    print("Number of clusters:", num_clusters)

    print_clusters(data, clusters)
    print_outliers(data, clusters)

def get_hypers():
    args = get_args()
    file_name = args['filename']
    eps = float(args['epsilon'])
    min_points = int(args['num'])

    df = get_data(file_name)
    data = init_data(df)


    es = [i/1000 for i in range(1, 100)]
    ns = [i for i in range(1, len(data)-1)]
    ns = [4]
    x = []
    y = []
    z = []
    min_sse = sys.float_info.max
    best_e = 0
    best_n = 0

    for e in es:
        print(e)
        for n in ns:
            data = dbscan(data, e, n)
            clusters = init_clusters(data)
            sse = statistics(data, clusters)
            clusters = [c for c in clusters if c.num_data > 0]
            if sse < min_sse and len(clusters) < 10:
                min_sse = sse
                best_e = e
                best_n = n
            x.append(e)
            y.append(n)
            z.append(sse)

    figure = plt.figure().gca(projection='3d')
    figure.scatter(x, y, z)
    plt.xlabel("epsilon")
    plt.ylabel("min_points")
    #plt.show()

    print("sse:", min_sse)
    print("e:", best_e)
    print("n:", best_n)

if __name__ == '__main__':    
    main()
