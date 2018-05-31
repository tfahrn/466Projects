import argparse
import csv
import numpy as np


class Matrix:

    def __init__(self):
        self.N = 0 
        self.nodes = {} 
        self.edges = []


    def construct_small_dataset(self, file):
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


    def construct_snap_dataset(self, file):
        # do snap
        i = 1

        

def get_args():
    parser = argparse.ArgumentParser(description='PageRank')
    parser.add_argument('-f', '--filename', help='path to input file', required=True)
    parser.add_argument('-s', '--snap', help='boolean; is snap file', required=True)

    return vars(parser.parse_args())


def main():
    args = get_args()
    filename = args['filename']
    is_snap = filename.endswith('.txt')

    graph = Matrix() 


if __name__ == '__main__':
    main()
