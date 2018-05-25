
# coding: utf-8

# In[ ]:

import argparse 
import re
import pandas as pd
import random
import sys
import numpy as np
import itertools
from anytree import NodeMixin, RenderTree, AnyNode
from anytree.exporter import JsonExporter
class Datum:
    def __init__(self, position, label):
        self.position = position
        self.label = label

class Cluster(NodeMixin):
    def __init__(self, number, points, dist, parent = None):
        self.number = number
        self.datums = points
        self.get_points(points)
        self.dist = dist
        self.parent = parent
        self.get_centroid()
        self.calc_distances()
    
    def get_centroid(self):
        df = pd.DataFrame(data = self.points)
        self.centroid = list(df.mean())
    
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
        print("Max Dist. to Center:", self.max_dist)
        print("Min Dist. to Center:", self.min_dist)
        print("Avg Dist. to Center:", self.avg_dist)
        print("SSE:", self.sse)
        print(len(self.points), "Points:")
        for i in range(len(self.points)):
            if(i<3): 
                print(",".join(str(round(x,4)) for x in self.points[i]), "Label:",self.labels[i])
    
    def get_points(self, datums):
        pts = []
        labels = []
        for d in datums:
            pts.append(d.position)
            labels.append(d.label)
        self.points = pts
        self.labels = labels
         
def get_data(ground_truth,d_vectors):
    v = open(d_vectors,'r')
    vectors = []
    for line in v: 
        line = line.replace("[", '').replace("]",'')
        vectors.append([float(x) for x in line.split(',')])
    v.close()

    gt = open(ground_truth,'r')
    doc_authors = [[line.rstrip()] for line in gt]
    gt.close()
  
    clusters = []
    anynodes = []
    for i in range(len(vectors)):
        position = vectors[i]
        label = doc_authors[i]
        point = [Datum(position,label)]
        c = Cluster(i,point,0)
        clusters.append(c)
        anynodes.append(AnyNode(type="leaf",
                                height = 0.0, data = position))
    return [clusters,i+1,anynodes]
      
#takes two arrays as parameters
#assumes two points being compared have same # of attributes
def sq_eucledian_dist(x,y):
    sum_sq_dist= 0
    for i in range(len(x)):
        sum_sq_dist += np.power(x[i] - y[i],2)
    return sum_sq_dist


def get_restrictions_vector(line):
    one_hot = re.split(',|\n', line)
    by_index = [i for i, value in enumerate(one_hot) if value == '1']
    label_index = [i for i, value in enumerate(one_hot) if value == '0']
    #Some data sets do now have row labels/class labels
    if(len(label_index)> 0):
        label_index = label_index[0]
    else:
        label_index = -1
    return [by_index,label_index]


