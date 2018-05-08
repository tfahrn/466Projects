import argparse
import matplotlib.pyplot as plt
import pandas as pd
import re
import sys
from mpl_toolkits.mplot3d import Axes3D

"""
Displays 2d/3d scatterplot of data in input .csv
"""

def get_args():
    parser = argparse.ArgumentParser(description='Visualize a clustering dataset')
    parser.add_argument('-f', '--filename', help='.csv data file', required=True)

    return vars(parser.parse_args())


def get_restrictions_vector(line):
    one_hot = re.split(',|\n', line)
    by_index = [i for i, value in enumerate(one_hot) if value == '1']

    return by_index


def display_data(file_name):
    with open(file_name) as f:
        restrictions = get_restrictions_vector(f.readline())
        if len(restrictions) > 3:
            print("Too many dimensions to plot")
            sys.exit()

        df = pd.read_csv(f, header=None, skiprows=0, usecols=restrictions)

        x = restrictions[0]
        y = restrictions[1]
        if len(restrictions) == 3:
            z = restrictions[2]
            figure = plt.figure().gca(projection='3d')
            figure.scatter(df[x], df[y], df[z])
        else:
            df.plot.scatter(x=x, y=y)
        plt.show()


def main():
    args = get_args()
    file_name = args['filename']
    display_data(file_name)
    

main()
