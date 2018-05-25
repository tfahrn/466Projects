import argparse
import json
import itertools
from heapq import heappush, heappop
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='knn to classify authors')
    parser.add_argument('-v', '--vectors', help='file of vectorized document representations', required=True)
    parser.add_argument('-g', '--ground', help='file of ground truth', required=True)
    parser.add_argument('-k', '--k', help='integer; neighbors to use for classification', required=True)

    return vars(parser.parse_args())


# returns list of np arrays (each vector in vector_file)
def get_vectors(vector_file):
    vectors = [np.asarray(json.loads(line)) for line in open(vector_file).readlines()]

    return vectors


def get_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def main():
    args = get_args()
    vector_file = args['vectors']
    ground_file = args['ground']
    k = int(args['k'])

    vectors = get_vectors(vector_file)
    # list of author_name
    predictions = ["Unknown Author" for v in vectors] 
    # list of (filename, author_name)
    filenames = [line.split(',')[0].rstrip() for line in open(ground_file).readlines()]
    classifications = [line.split(',')[1].rstrip() for line in open(ground_file).readlines()]

    for i in range(len(vectors)):
        vector = vectors[i]
        author = classifications[i]
        heap = []
        
        for j in range(len(vectors)):
            if i != j:
                other_vector = vectors[j]
                other_author = classifications[j]
                similarity = get_similarity(vector, other_vector)
                # invert similarity to simulate max-heap
                heappush(heap, (-similarity, other_author))

        neighbor_authors = [heappop(heap)[1] for neighbor in range(k)]
        plurality_author = max(neighbor_authors, key=neighbor_authors.count) 
        predictions[i] = plurality_author
        print('{},{}'.format(filenames[i], predictions[i]))



if __name__ == '__main__':
    main()
