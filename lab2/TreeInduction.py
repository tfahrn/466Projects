
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
import math


# In[34]:


def setup():
    y_vars = {}
    df = pd.read_csv("data/tree03-20-words.csv",skiprows=[1,2])
    df.drop("Id",inplace = True,axis = 1)
    for col in df.columns:
        y_vars[col] = list(set(df[col]))
    y_vars
    return[df,y_vars]


# In[186]:


#Assumes we have a subset that has already been split on an attribute
#pass in a series object

#Writing generic solution so we cant assume column names

def entropy_d(d):
    freqs = d.value_counts().values
    tot_n = np.sum(freqs)
    entropy = 0 
    #Sums up all of the entropys for each possible value for a given
    #attribute Ai
    for f in freqs:
        entropy += -(f/tot_n)*math.log((f/tot_n),2)
    return entropy

#D = the original data
#Di = data subset after the partition/split
def gain(split_data,a_name,r_list):
    return entropy_d(r_list) - entropy_a(split_data,a_name)
def gain_ratio(df, a_name, resp):   
    split_data = df[[a_name, resp]]
    #pd.pivot_table(temp, index = "Education",columns = 'Vote')
    split_data = split_data.groupby([a_name, resp]).size().to_frame()
    split_data.reset_index(inplace=True)
    split_data.rename(columns = {0:"count"},inplace = True)
    
    edges = list(set(split_data[a_name]))
    a_sum = 0
    for edge in edges:
        edge_df = split_data[split_data[a_name] == edge]
        prop = np.sum(edge_df['count'])/np.sum(split_data['count'])
        a_sum += -prop * math.log(prop,2)
    return gain(split_data,a_name,df[resp])/a_sum

def entropy_a(split_data, a_name):
    edges = list(set(split_data[a_name]))
    a_sum = 0
    for edge in edges:
        edge_df = split_data[split_data[a_name] == edge]
        prop = np.sum(edge_df['count'])/np.sum(split_data['count'])
        counts = edge_df['count']
        entropy_d = 0
        for c in counts:
            prop_d = c/np.sum(edge_df['count'])
            entropy_d += -prop_d * math.log(prop_d,2)   
        
        a_sum += prop * entropy_d
    return a_sum
    


# In[175]:


res = setup()
df = res[0]


# In[192]:


split_data = df[["Education","Vote"]]
#pd.pivot_table(temp, index = "Education",columns = 'Vote')
split_data = split_data.groupby(["Education","Vote"]).size().to_frame()
split_data.reset_index(inplace=True)
split_data.rename(columns = {0:"count"},inplace = True)
split_data


# In[187]:


gain_ratio(df,"Education","Vote")


# In[203]:


-(4/10) * math.log((4/10),2) + -(6/10) * math.log((6/10),2)



-(6/8) * math.log((6/8),2) + -(2/8) * math.log((2/8),2)

0.4 * .811
.3244+ .48545
1-.80985


# In[206]:


-(10/20) * math.log((10/20),2) + -(8/20) * math.log((8/20),2) + -(2/20) * math.log((2/20),2)


# In[207]:


0.19/1.361

