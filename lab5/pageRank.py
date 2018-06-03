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
            print("Iteration:", num_iterations)
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


class Node:
    def __init__(self, name):
        self.name = name
        self.num_out = 0
        self.in_node_names = set() 


def construct_snap(file):
    name_to_node = {}

    with open(file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            else:
                edge = line.split('\t')
                from_name = edge[0]
                to_name = edge[1]
                from_node = Node(from_name)
                to_node = Node(to_name)

                if from_name in name_to_node:
                    name_to_node[from_name].num_out += 1
                else:
                    from_node.num_out = 1
                    name_to_node[from_name] = from_node

                if to_name in name_to_node:
                    name_to_node[to_name].in_node_names.add(from_name)
                else:
                    to_node.in_node_names.add(from_name)
                    to_node.num_out = 0
                    name_to_node[to_name] = to_node

    return name_to_node


def page_rank_snap(name_to_node, d, epsilon):
    num_nodes = len(name_to_node)

    old_rank = {}
    new_rank = {}

    for name in name_to_node:
        old_rank[name] = 1/num_nodes

    num_iterations = 0
    while(num_iterations < epsilon):
        print("Iteration:", num_iterations)
        num_iterations += 1
        for name in name_to_node:
            node = name_to_node[name]
            random_term = (1-d)*(1/num_nodes)
            incoming_sum = 0
            for in_name in node.in_node_names:
                in_node = name_to_node[in_name]
                page_rank_in_node = old_rank[in_name]
                incoming_sum += (1/in_node.num_out)*page_rank_in_node
            incoming_term = d*incoming_sum

            new_rank[name] = random_term + incoming_term

    return new_rank, num_iterations


def get_args():
    parser = argparse.ArgumentParser(description='PageRank')
    parser.add_argument('-f', '--filename', help='path to input file', required=True)
    parser.add_argument('-o', '--output', help='path to an output file', required=True)

    return vars(parser.parse_args())


def to_file(f_name,results,stats):
    f = open(f_name,"w")
    f.write(stats)
    for r in results:
        f.write(r[0] + " with pagerank: " + str(r[1]) + "\n")
    f.close()


def main():
    start_time = time.time()
    args = get_args()
    filename = args['filename']
    output = args['output']
    is_snap = filename.endswith('.txt')

    graph = Graph() 
    if is_snap:
        name_to_node = construct_snap(filename)
        read_time = time.time() - start_time
        start_time = time.time()
        node_to_rank, num_iterations = page_rank_snap(name_to_node, 0.9, 100)
    else:
        graph.construct_small(filename)
        read_time = time.time() - start_time
        start_time = time.time()
        node_to_rank, num_iterations = graph.page_rank(0.9, 100)

    processing_time = time.time() - start_time

    stats = "Read time: " + str(round(read_time, 2)) + " Processing time: " + str(round(processing_time, 2))
    stats =  stats + " Number of iterations: " +  str(num_iterations) + "\n"
    
    print("Read time: {}. Processing time: {}. Number of iterations: {}".format(round(read_time, 2), round(processing_time, 2), num_iterations))

    results = [(node, rank) for node, rank in node_to_rank.items()]
    results.sort(key=lambda t:t[1], reverse=True)

    print(results[:5])
    to_file(output,results,stats)


if __name__ == '__main__':
    main()
