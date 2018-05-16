
# coding: utf-8

# In[133]:

import source
import numpy as np
import pandas as pd

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
            if(j <= i):
                dist = cl_dist(clusters[i], clusters[j])
                distances.append(dist)
        dm.append(distances)
    return dm

#Each data point starts as its own cluster, returns list of 
#tree nodes
def init_clusters(df):
    clusters = []
    for i, row in df.iterrows():
        position = [coord for coord in row]
        #Get label from row data then remove it
        label = position[len(position)-1]
        position.remove(label)
        point = [source.Datum(position, label)]
        c = source.Cluster(i,point,0)
        clusters.append(c)
    return [clusters,i+1]

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
    
    for i in range(len(dm)):
        if i < min_idx:
            new_dm[i] = dm[i]
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
        '''
        print("{X", closest[0],",X", closest[1] ,"}:X",col_idx)
        print("Distances: ",dist1,dist2)
        print("Max_dist: ", max_dist)
        
        '''
        
        if(col_idx < min_idx):
            new_dm[min_idx][col_idx] = max_dist
        else:
            if(col_idx > np.max(closest)):
                new_dm[col_idx-1][min_idx] = max_dist
            else:
                new_dm[col_idx][min_idx] = max_dist
        
    #Add values for clusters unaffected by merged clusters  
    '''
    print("DM before unnaffected values: ")
    ix = 0
    for col in new_dm:
        print("X:"+str(ix),col)
        ix+=1
    '''
    for i in range(len(not_merged)):
        for j in range(len(not_merged)):
            if(j < is):
                #if distance is in last row it needs to be shifted up one due to matrix collapsing
                #if (not_merged[j] == len(dm[len(dm)-1])-2):
                     #new_dm[not_merged[i]-1][not_merged[j]-1] = dm[not_merged[i]][not_merged[j]]
                if (not_merged[j] > min_idx):
                    new_dm[not_merged[i]-1][not_merged[j]-1] = dm[not_merged[i]][not_merged[j]]
                    '''
                     if(dm[not_merged[i]][not_merged[j]] == float("inf")):
                        print("DM # Cols", len(dm))
                        print("Merge Location:",closest)
                        print("Col:",not_merged[i],"Row:",not_merged[j])
                    
                    '''
                #distance is not in last row    
                else: 
                    #If unaffected value is less than the merged column do not shift it
                    if(not_merged[i] < np.max(closest)):
                         new_dm[not_merged[i]][not_merged[j]] = dm[not_merged[i]][not_merged[j]]
                    #Shift unaffected value one column left
                    else:
                        new_dm[not_merged[i]-1][not_merged[j]] = dm[not_merged[i]][not_merged[j]]
    '''
    print("DM AFTER unnaffected values: ")
    ix = 0
    for col in new_dm:
        print("X:"+str(ix),col)
        ix+=1
    '''
    
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
    #calculate distance matrix for current clusters
    dm = dist_matrix(clusters)
    '''
    ix = 0
    print("Original DM:")
    for col in dm:
        print("X:"+str(ix),col)
        ix+=1
    
    '''

    #While we have more than one cluster
    i = 0
    while(len(dm) > 1):
        #Find minimum distance in the matrix
        min_res= get_min_dist(dm)
        merge_loc = min_res[0]
        min_dist = min_res[1]
        #print("Min Location: ", merge_loc)
        #print("Cluster Size:", len(clusters))

        '''
          if(len(clusters)==18):
            ix = 0
            print("Error DM:")
            for col in dm:
                print("X:"+str(ix),col)
                ix+=1
            break
           if(min_dist == 0):
        
            print("Merge #: ",i)
            print("Min_dist: ",min_dist)
            print("Merging clusters: ",merge_loc)
            print("Error")
            break
        '''      
        #Merge clusters and recalculate matrix
        dm = recalc_dm(merge_loc,dm)
        merged_datums = clusters[merge_loc[0]].datums + clusters[merge_loc[1]].datums
        m_cluster = source.Cluster(cluster_id, merged_datums, min_dist)
        
        #Set parent of merged nodes to new cluster created
        clusters[merge_loc[0]].parent = m_cluster
        clusters[merge_loc[1]].parent = m_cluster
        #Remove clusters thats were merged from list
        del clusters[merge_loc[0]]
        del clusters[merge_loc[1]]
        #Insert merged cluster into list
        clusters.insert(merge_loc[0],m_cluster)
        cluster_id +=1
        i+=1
    return clusters


# In[134]:

df = source.get_data("data/planets.csv")
clusters = agg_clustering(df)


# In[69]:

def print_dendogram(node):
    for pre, _, node in RenderTree(node):
        treestr = u"%s%s" % (pre, node.dist)
        print(treestr.ljust(8))


# In[113]:


#dm = [[float('inf')],[4,float('inf')],[3,2,float('inf')],[8,7,6,float('inf')],[11,5,6,3,float('inf')]]
#dm = [[float('inf')],[4,float('inf')],[3,2,float('inf')],[8,7,6,float('inf')],[1,5,6,3,float('inf')]]
dm = [[float('inf')],[4,float('inf')],[3,2,float('inf')],[8,1,6,float('inf')],[11,5,6,3,float('inf')]]
#dm = [[float('inf')],[4,float('inf')],[3,9,float('inf')],[8,7,2,float('inf')],[11,5,6,3,float('inf')]]
while(len(dm) > 1):
        #Find minimum distance in the matrix
        min_res= get_min_dist(dm)
        merge_loc = min_res[0]
        min_dist = min_res[1]
        #print("DM: ", dm)
        #Merge clusters and recalculate matrix
        dm = recalc_dm(merge_loc,dm)
        print(dm)

