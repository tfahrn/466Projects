import argparse
import csv
import numpy as np
import time


class Graph:

    def __init__(self):
        self.N = 0 
        self.nodes = {} 
        self.edges = []
        self.matrix = None
        self.last_rank = None
        self.next_rank = None


    def construct_small(self, file):
        with open(file) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in reader:
                # remove whitespace, quotations
                row = [s.replace(' ', '').replace('"', '') for s in row]

                node1 = row[0]
                node2 = row[2]

                if node1 not in self.nodes:
                    self.N += 1
                    self.nodes[node1] = self.N - 1
                if node2 not in self.nodes:
                    self.N += 1
                    self.nodes[node2] = self.N - 1

                self.edges.append(row)
        
        self.matrix = np.zeros((self.N, self.N))

        for edge in self.edges:
            node1 = edge[0]
            value1 = int(edge[1])
            node2 = edge[2]
            value2 = int(edge[3])

            if "football" in file:
                losing_team = node1 if value1 < value2 else node2
                winning_team = node1 if losing_team == node2 else node2
                # directed edge from losing team to winning team
                self.matrix[self.nodes[losing_team], self.nodes[winning_team]] = 1
            elif "lesmis" in file:
                self.matrix[self.nodes[node1], self.nodes[node2]] = value1
                self.matrix[self.nodes[node2], self.nodes[node1]] = value1
            else:
                self.matrix[self.nodes[node1], self.nodes[node2]] = 1 
                self.matrix[self.nodes[node2], self.nodes[node1]] = 1


    def construct_snap(self, file):
        with open(file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                else:
                    edge = line.split('\t')
                    node1 = edge[0]
                    node2 = edge[1]

                    if node1 not in self.nodes:
                        self.N += 1
                        self.nodes[node1] = self.N - 1
                    if node2 not in self.nodes:
                        self.N += 1
                        self.nodes[node2] = self.N - 1

                    self.edges.append(edge)

        self.matrix = np.zeros((self.N, self.N))

        for edge in self.edges:
            node1 = edge[0]
            node2 = edge[1]
            self.matrix[self.nodes[node1], self.nodes[node2]] = 1 


    def page_rank(self, d, epsilon):
        self.old_rank = np.zeros(self.N)
        self.new_rank = np.zeros(self.N) 
        for i in range(self.N):
            self.old_rank[i] = 1/self.N

        num_iterations = 0
        while(num_iterations < epsilon):
        # while(abs(np.sum(self.new_rank - self.old_rank)) > epsilon):
            num_iterations += 1
            for i in range(self.N):
                random_term = (1-d)*(1/self.N)
                incoming_sum = 0
                for k in range(self.N):
                    if i != k and self.matrix[k, i] != 0:
                        num_k_out = np.count_nonzero(self.matrix[k])
                        page_rank_k = self.old_rank[k]
                        incoming_sum += (1/num_k_out)*page_rank_k
                incoming_term = d*incoming_sum

                self.new_rank[i] = random_term + incoming_term
        
        id_to_node = {id : node for node, id in self.nodes.items()}
        node_to_rank = {}

        for i in range(self.N):
            node = id_to_node[i]
            node_to_rank[node] = self.new_rank[i]

        return node_to_rank, num_iterations


def get_args():
    parser = argparse.ArgumentParser(description='PageRank')
    parser.add_argument('-f', '--filename', help='path to input file', required=True)

    return vars(parser.parse_args())


def main():
    start_time = time.time()
    args = get_args()
    filename = args['filename']
    is_snap = filename.endswith('.txt')

    graph = Graph() 
    if is_snap:
        graph.construct_snap(filename)
    else:
        graph.construct_small(filename)

    read_time = time.time() - start_time
    start_time = time.time()

    node_to_rank, num_iterations = graph.page_rank(0.5, 100)

    processing_time = time.time() - start_time

    print("Read time: {}. Processing time: {}. Number of iterations: {}".format(round(read_time, 2), round(processing_time, 2), num_iterations))

    results = [(node, rank) for node, rank in node_to_rank.items()]
    results.sort(key=lambda t:t[1], reverse=True)

    print(results[:5])


if __name__ == '__main__':
    main()
