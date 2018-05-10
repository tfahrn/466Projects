
# coding: utf-8

# In[ ]:

import argparse 
import re
import pandas as pd
import random
import sys
import numpy as np
import itertools
class Datum:
    def __init__(self, position, label):
        self.position = position
        self.label = label

class Cluster:
    def __init__(self, number, points):
        self.number = number
        self.points = points
        self.centroid = self.get_centroid()
        self.calc_distances()
    
    def get_centroid(self):
        df = pd.DataFrame(data = self.points)
        return list(df.mean())
    def calc_distances(self):
        min_d = float("inf")
        max_d = -1
        dist_sum = 0
        sse = 0
        for pt in self.points:
            dist = sq_eucledian_dist(pt,self.centroid)
            dist_sum += dist
            sse += np.power(dist,2)
            if(dist < min_d):
                min_d = dist
            if(dist > max_d):
                max_d = dist
        self.avg_dist = dist_sum/len(self.points)
        self.max_dist =  max_d
        self.min_dist =  min_d
        self.sse = sse
 
    def print_info(self):
        print("Cluster ",self.number, ":")
        print("Center: ", ",".join(str(x) for x in self.centroid))
        print("Max Dist. to Center: ", self.max_dist)
        print("Min Dist. to Center: ", self.min_dist)
        print("Avg Dist. to Center: ", self.avg_dist)
        print(len(self.points), "Points:")
        for pt in self.points:
            print(",".join(str(x) for x in pt))
    
        
def get_data(file_name):
    with open(file_name) as f:
        restrictions = get_restrictions_vector(f.readline())
        label_col = restrictions[1]
        data_cols = restrictions[0]
        df = pd.read_csv(f, header=None, skiprows=0)
        num_df = df[data_cols]
        num_df = (num_df - num_df.mean())/(num_df.std()) # normalize
        num_df['label'] = df[label_col]
        return num_df

#takes two arrays as parameters
#assumes two points being compared have same # of attributes
def sq_eucledian_dist(x,y):
    sum_sq_dist= 0
    for i in range(len(x)):
        sum_sq_dist += np.power(x[i] - y[i],2)
    #Check for the case that a cluster is being compared to itself
    #Make the value infinite because we dont want to consider this in our
    #min distance calculation
    if(sum_sq_dist == 0):
        sum_sq_dist = float("inf")
    return sum_sq_dist


def get_restrictions_vector(line):
    one_hot = re.split(',|\n', line)
    by_index = [i for i, value in enumerate(one_hot) if value == '1']
    label_index = [i for i, value in enumerate(one_hot) if value == '0'][0]
    return [by_index,label_index]
