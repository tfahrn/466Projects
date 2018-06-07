
# coding: utf-8

# In[65]:

import pandas as pd
import numpy as np
#Transposes matrix so we can use our algorithms as index based
def get_matrix(file_path):
    #Converts to np array
    r_df = pd.read_csv(file_path,header = None).transpose()
    #Each array is one row of the df
    return r_df.as_matrix()

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
            c_prime_mean = np.mean(np.concatenate((r_mat[i][0:item],r_mat[i][item+1:])))
            norm_sim += sim * (c_prime_m_util - c_prime_mean)
    k= 1/sum_sim
    return [k,norm_sim]

# Gets the mean rating for a given item S for all other users
def mean_utility(r_mat,user,item):
    rate_sum = 0
    n = 0
    for i in range(len(r_mat)):
        if(i!=user):
            rate_sum += r_mat[i][item]
            n+=1
    return rate_sum/n

def adj_weighted_sum(r_mat, user, item):
    #Get all other item ratings for the given user
    s_prime = np.concatenate((r_mat[user][0:item],r_mat[user][item+1:]))
    #Mean rating for other items
    u_mean = np.mean(s_prime)
    res = sum_similarity(r_mat,user,item)
    k = res[0]
    sum_prod = res[1]
    
    return u_mean + k * sum_prod
    
#def adj_wsum_nn(r_mat,user,item):

r_mat = get_matrix("data/jester-data-1.csv")
#Item based, I havent accounted for the 99 (no rating) values so our recommended value is way too high!
adj_weighted_sum(r_mat,0,10)

