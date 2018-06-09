
# coding: utf-8

# In[67]:

import pandas as pd
import numpy as np
import random
import sys
import argparse
import itertools
from matplotlib import pyplot as plt
##################AlL code related to KNN ##############
class Datum:
    def __init__(self, position, cluster):
        self.position = position
        self.cluster = cluster

class Cluster:
    def __init__(self, number, position):
        self.number = number
        self.position = position
        self.min_dist = sys.float_info.max
        self.max_dist = 0
        self.tot_dist = 0
        self.num_data = 0

def get_distance(datum, cluster):
    datum_position = np.asarray(datum.position)
    cluster_position = np.asarray(cluster.position)
    dist = np.linalg.norm(datum_position - cluster_position)
    return dist

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
def init_data(r_mat):
    data = []
    for i in range(len(r_mat)):
        data.append(Datum(r_mat[i].tolist(), -1))
    return data 
def cust_mean(array):
    rate_sum = 0
    n = 0
    for rating in array:
        if rating!=99:
            rate_sum+=rating
            n+=1
    return rate_sum/n       

def kmeans(data, clusters, stop_ratio):
    ratio_moved = 1

    while(ratio_moved > stop_ratio):
        num_changed = 0
        cluster_pos_totals = [[] for c in clusters] # LoL, sum of positions of data points per cluster 
        cluster_count_totals = [0 for c in clusters] # list of num data points per cluster

        for datum in data:
            min_dist = sys.float_info.max
            nearest = datum.cluster 
            for cluster in clusters:
                dist = get_distance(datum, cluster)
                if dist < min_dist:
                    min_dist = dist
                    nearest = cluster.number
                    
            # nearest cluster has changed
            if datum.cluster != nearest:
                num_changed += 1
                datum.cluster = nearest
            
            """
            if len(cluster_pos_totals) < datum.cluster + 1:
                cluster_pos_totals[datum.cluster] = None
            """
            cluster_pos_totals[datum.cluster] =                 [sum(pos) for pos in
                itertools.zip_longest(datum.position, cluster_pos_totals[datum.cluster], fillvalue=0)]
            
            cluster_count_totals[datum.cluster] += 1

        for i, cluster in enumerate(clusters):
            mean_position = [coord/cluster_count_totals[i]
                for coord in cluster_pos_totals[i]]
            cluster.position = mean_position

        ratio_moved = num_changed/len(data)

    return data, clusters

def statistics(data, clusters):
    sse = 0

    for cluster in clusters:
        for datum in data:
            if datum.cluster == cluster.number:
                dist = get_distance(datum, cluster)
                if dist > cluster.max_dist:
                    cluster.max_dist = dist
                if dist < cluster.min_dist:
                    cluster.min_dist = dist
                cluster.tot_dist += dist
                cluster.num_data += 1
                sse += dist*dist
    return sse
################## End code related to KNN ##############

#Transposes matrix so we can use our algorithms as index based
def get_matrix(file_path):
    #Converts to np array
    r_df = pd.read_csv(file_path,header = None)
    num_jokes = r_df[0]
    r_df = r_df.drop(r_df.columns[0],axis =1)
    return (r_df,num_jokes)

#def pearson_corr(r_mat, user, item):

def get_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    
def sum_similarity(r_mat,user,item):
    sum_sim = 0
    norm_sim = 0
    for i in range(len(r_mat)):
        if(i!=user):
            sim = get_similarity(r_mat[user],r_mat[i])
            sum_sim += np.abs(sim)
            c_prime_m_util =  mean_utility(r_mat,i,item)
            c_prime_mean = cust_mean(np.concatenate((r_mat[i][0:item],r_mat[i][item+1:])))
            norm_sim += sim * (c_prime_m_util - c_prime_mean)
    k= 1/sum_sim
    return [k,norm_sim]

# Gets the mean rating for a given item S for all other users
def mean_utility(r_mat,user,item):
    rate_sum = 0
    n = 0
    for i in range(len(r_mat)):
        if(i!=user):
            rating = r_mat[i][item]
            if(rating != 99):
                rate_sum += rating
                n+=1
    return rate_sum/n


#Only run this item based. Run time is too high for user based
def adj_weighted_sum(r_mat, user, item):
    #Matrix is transposed so we have to swap user/item indices
    temp = user
    user =  item
    item = temp
    
    #Get all other item ratings for the given user
    s_prime = np.concatenate((r_mat[user][0:item],r_mat[user][item+1:]))
    #Mean rating for other items
    u_mean = cust_mean(s_prime)
    res = sum_similarity(r_mat,user,item)
    k = res[0]
    sum_prod = res[1]
    
    return u_mean + k * sum_prod

def adj_weighted_sum_nn(r_mat, user, item,datums):
    #Matrix is transposed so we have to swap user/item indices
    temp = user
    user =  item
    item = temp
    
    #Get all other item ratings for the given user
    s_prime = np.concatenate((r_mat[user][0:item],r_mat[user][item+1:]))
    #Mean rating for other items
    u_mean = cust_mean(s_prime)
    res = sum_similarity_nn(r_mat,user,item,datums)
    k = res[0]
    sum_prod = res[1]
    
    return u_mean + k * sum_prod

#user = item , item = user
def sum_similarity_nn(r_mat,user,item,datums):
    sum_sim = 0
    norm_sim = 0
    for i in range(len(r_mat)):
        if(i!=user):
            other_item = datums[i]
            #Check if items belong to the same cluster
            if(other_item.cluster==datums[user].cluster):
                sim = get_similarity(r_mat[user],r_mat[i])
                sum_sim += np.abs(sim)
                c_prime_m_util =  mean_utility(r_mat,i,item)
                c_prime_mean = cust_mean(np.concatenate((r_mat[i][0:item],r_mat[i][item+1:])))
                norm_sim += sim * (c_prime_m_util - c_prime_mean)
    k= 1/sum_sim
    return [k,norm_sim]

def get_args():
    parser = argparse.ArgumentParser(description='recommendation system')
    parser.add_argument('-f', '--filepath', help='path to file', required=True)
    parser.add_argument('-u', '--user', help='integer; user you want a recommendation for', required=True)
    parser.add_argument('-i', '--item', help='integer; item you want a recommendation for', required=True)
    return vars(parser.parse_args())

    
def main():
    args = get_args()
    filepath = args['filepath']
    user = int(args['user'])
    item = int(args['item'])
    r_df = get_matrix(filepath)
    #User based
    r_matrix_user = r_df.as_matrix()
    #Item Based
    r_matrix_item = r_df.transpose().as_matrix()

    if(user > len(r_matrix_item[0])):
        print("User index exceeds availabe users")
    elif(item > len(r_matrix_user[0])):
        print("Item index exceeds available items")
    else:
        #Call collab functions
        mean_utility(r_matrix_user,user,item)


# In[58]:

res = get_matrix("data/jester-data-1.csv")
njokes_user = res[1]
r_mat_user = res[0].as_matrix()
r_mat_item= res[0].transpose().as_matrix()
#Item based
print(adj_weighted_sum(r_mat_item,1,10))
#User based
print(mean_utility(r_mat_user,1,10))

#Clustering item based
data = init_data(r_mat_item)
#Using optimal k, k=4
clusters = init_clusters(4,data)
stop_ratio = 1/len(data)
data, clusters = kmeans(data, clusters, stop_ratio)
sse = statistics(data, clusters)

#Item based KNN
print(adj_weighted_sum_nn(r_mat_item,1,10,data))


# In[49]:

'''
ks = [k for k in range(1, min(20, len(data)-1))]
sse_per_k = []
max_sse = 0
for k in ks:
    clusters = init_clusters(k, data)
    stop_ratio = 1/len(data)
    data, clusters = kmeans(data, clusters, stop_ratio)
    sse = statistics(data, clusters)
    sse_per_k.append(sse)
    if sse > max_sse:
        max_sse = sse

plt.plot(ks, sse_per_k, 'bo')
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.axis([0, 20, 0, max_sse*(7/5)])
plt.show()
'''

