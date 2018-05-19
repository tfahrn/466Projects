
# coding: utf-8

# In[23]:

import source
import numpy as np
import pandas as pd
from anytree import NodeMixin, RenderTree,PostOrderIter, ContStyle, AnyNode
from anytree.exporter import JsonExporter
import argparse

def cl_dist(c1, c2):
    max_dist = -1
    for pt1 in c1.datums:
        for pt2 in c2.datums:
            dist = source.sq_eucledian_dist(pt1.position,pt2.position)
            if(dist > max_dist):
                max_dist = dist
    return max_dist

def dist_matrix(clusters):
    dm = []
    for i in range(len(clusters)):
        distances = []
        for j in range(len(clusters)):
            if(j < i):
                dist = cl_dist(clusters[i], clusters[j])
                distances.append(dist)
            #Check for case cluster is being compared with itself
            if(j == i):
                dist =  float('inf')
                distances.append(dist)
        dm.append(distances)
    return dm

#Each data point starts as its own cluster, returns list of 
#tree nodes
def init_clusters(df):
    clusters = []
    anynodes = []
    for i, row in df.iterrows():
        position = [coord for coord in row]
        #Get label from row data then remove it
        label = position[len(position)-1]
        position.remove(label)
        point = [source.Datum(position, label)]
        #Put old code here
        c = source.Cluster(i,point,0)
        clusters.append(c)
        anynodes.append(AnyNode(type="leaf",
                                height = 0.0, data = position))
    return [clusters,i+1,anynodes]

def get_min_dist(dm):
    #Need to find the clusters with the smallest distance between them
    #Get min dist of each column
    col_mins = []
    for col in dm:
        col_mins.append(np.min(col))
    min_dist = np.min(col_mins)
    #get two clusters that have the min dist
    i = 0
    idx = 0
    for col in dm:
        try: 
            idx = col.index(min_dist)
        except ValueError:
            idx = -1
        #Break out of for loop if min dist was found
        if(idx >=0):
            break
        i += 1
    return [[i,idx],min_dist]

def recalc_dm(closest,dm):
    not_merged = list(set(list(range(0, len(dm)))).difference(set(closest)))
    new_dm = new_dmat(len(dm))
    #add all columns left of merged clusters whose distances are not affected
    min_idx = np.min(closest)
    #Add new values for merged clusters and their relations to other clusters
    for col_idx in not_merged:
        dist1 = 0
        dist2 = 0
        
        if (col_idx > closest[0]):
            dist1 = dm[col_idx][closest[0]]
        else:
            dist1 = dm[closest[0]][col_idx]
        if (col_idx > closest[1]): 
            dist2 = dm[col_idx][closest[1]]
        else:
            dist2 = dm[closest[1]][col_idx]
        max_dist = np.max([dist1,dist2])

        if(col_idx < min_idx):
            new_dm[min_idx][col_idx] = max_dist
        else:
            if(col_idx > np.max(closest)):
                new_dm[col_idx-1][min_idx] = max_dist
            else:
                new_dm[col_idx][min_idx] = max_dist
        
    #Add values for clusters unaffected by merged clusters  
    #i = Column
    #j = Row
    for i in range(len(not_merged)):
        for j in range(len(not_merged)):
            if(j < i):
                #Check if we need to shift value up a row
                if (not_merged[j] > np.max(closest)):
                    if(not_merged[i] < np.max(closest)):
                        new_dm[not_merged[i]][not_merged[j]-1] = dm[not_merged[i]][not_merged[j]]
                    else:
                        new_dm[not_merged[i]-1][not_merged[j]-1] = dm[not_merged[i]][not_merged[j]]
                #We dont need to shift up, check column next  
                else: 
                    #If unaffected value is less than the merged column do not shift it
                    if(not_merged[i] < np.max(closest)):
                        new_dm[not_merged[i]][not_merged[j]] = dm[not_merged[i]][not_merged[j]]
              
                    #Shift unaffected value one column left
                    else:
                        new_dm[not_merged[i]-1][not_merged[j]] = dm[not_merged[i]][not_merged[j]]
    return new_dm
   
def new_dmat(l):
    nm = []
    for i in range(l-1):
        sub = [0] * (i+1)
        sub[len(sub)-1] = float('inf')
        nm.append(sub)
    return nm

def agg_clustering(df):
    #Assign each data point to its own cluster
    res = init_clusters(df)
    clusters = res[0]
    cluster_id = res[1]
    anynodes = res[2]
    #calculate distance matrix for current clusters
    dm = dist_matrix(clusters)
    #While we have more than one cluster
    i = 0
    while(len(clusters) > 1):
        #Find minimum distance in the matrix
        min_res= get_min_dist(dm)
        merge_loc = min_res[0]
        min_dist = min_res[1]
        #Merge clusters and recalculate matrix
        dm = recalc_dm(merge_loc,dm)
        merged_datums = clusters[merge_loc[0]].datums + clusters[merge_loc[1]].datums
        m_cluster = source.Cluster(cluster_id, merged_datums, min_dist)
        
        #If last merge we need to label root node with type root
        if (len(clusters) == 2):
            m_anynode =  AnyNode(height = min_dist, type="root")
        else:
            m_anynode =  AnyNode(height = min_dist, type="node")
        
        #Set parent of merged nodes to new cluster created
        
        clusters[merge_loc[0]].parent = m_cluster
        clusters[merge_loc[1]].parent = m_cluster
        
        anynodes[merge_loc[0]].parent = m_anynode
        anynodes[merge_loc[1]].parent = m_anynode
        #Remove clusters thats were merged from list
        del clusters[merge_loc[0]]
        del clusters[merge_loc[1]]
        del anynodes[merge_loc[0]]
        del anynodes[merge_loc[1]]
        
        #Insert merged cluster into list
        clusters.insert(merge_loc[0],m_cluster)
        anynodes.insert(merge_loc[0],m_anynode)
        cluster_id +=1
        i+=1
    return [clusters,anynodes]

def print_dendogram(node):
    for pre, _, node in RenderTree(node):
        treestr = u"%s%s\t%s" % (pre, node.dist,node.number)
        print(treestr.ljust(8))
def print_clusters(root,threshold,print_res):
    c_final = []
    for node in PostOrderIter(root):
        if(node.dist < threshold and node.parent is not None and node.parent.dist > threshold):
            c_final.append(node)
    if(len(c_final) == 0):
        c_final.append(root)
    if print_res:
        print("Number of clusters found:", len(c_final))
    for node in c_final:
        if print_res:
            node.print_info()
            
    return c_final

def get_args():
    parser = argparse.ArgumentParser(description='hierarchical clustering')
    parser.add_argument('-f', '--filename', help='.csv data file', required=True)
    parser.add_argument('-t', '--threshold', help='threshold value', required=True)
    parser.add_argument('-o', '--out', help='output file', required=True)
    parser.add_argument('-b', '--boolean', help='True/False print output', required=True)

    return vars(parser.parse_args())


#args(filepath to data,threshhold,output file name for dendogram,boolean for printing)
#returns number of clusters found for given threshold
def agg_main():
    args = get_args()
    fname = args['filename']
    threshold = int(args['threshold'])
    output_fn = args['out']
    output_res = (args['boolean'] == 'True')

    df = source.get_data(fname)
    res = agg_clustering(df)
    root = res[0][0]
    json_root = res[1][0]
    #print_dendogram(root)
    exporter = JsonExporter(indent=2)
    json = exporter.export(json_root)
    c_final = print_clusters(root,threshold,output_res)
    if output_res:
        fo = open(output_fn, "w")
        fo.write(json);
        fo.close()
    return c_final


if __name__=='__main__':
    agg_main()


'''
#Code used for heat map visualization
import matplotlib.pyplot as plt
import seaborn as sns

i = 1
x = []
avg_avgD = []
num_clusters = []
while (i < 10):
    avg_dists = []
    f_clusters= agg_main("data/4clusters.csv",i,"many.txt",False)
    for c in f_clusters:
        avg_dists.append(c.avg_dist)
    avg_avgD.append(np.mean(avg_dists))
    x.append(i)
    num_clusters.append(len(f_clusters))
    i+=1
    
df = pd.DataFrame(data=x,columns = ["threshold"])
df['avg_avgDist'] = avg_avgD
df['#clusters'] =  num_clusters
pivot_data = df.pivot(index = "#clusters",columns = "threshold",
                              values = "avg_avgDist")
fig = plt.figure()
sns.heatmap(pivot_data)
plt.title("Average of Average Distance to Center:\nClusters vs Threshold")
plt.subplots_adjust(bottom=0.15)
plt.show()

'''

