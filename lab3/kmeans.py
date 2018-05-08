import argparse 
import re
import pandas as pd
import random

class Datum:

    def __init__(self, position):
        self.position = position

    def set_cluster(self, cluster):
        self.cluster = cluster

class Cluster:

    def __init__(self, number, position):
        self.number = number
        self.position = position

    def set_position(self, position):
        self.position = position

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
        data.append(Datum(position))

    return data 

def main():
    args = get_args()
    file_name = args['filename']
    k = int(args['k'])
    df = get_data(file_name)
    data = init_data(df)
    clusters = init_clusters(k, data)

    for cluster in clusters:
        print(cluster.number)
        print(cluster.position)

    for datum in data:
        print(datum.position)

main()
