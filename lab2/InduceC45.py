import pandas as pd
import numpy as np
import math
from anytree import Node, RenderTree
from anytree.exporter import JsonExporter


# returns tuple: (most frequent class in D, there is only one class in D)
def get_most_freq_class(D, c_name):
    class_values = D[c_name].values
    class_to_freq = {}
    
    for data in class_values:
        #TODO: find class
        class_val = data

        if class_val in class_to_freq:
            class_to_freq[class_val] = class_to_freq[class_val] + 1
        else:
            class_to_freq[class_val] = 1

    total = 0
    max_freq = 0
    max_class = None

    for k, v in class_to_freq.items():

        if v > max_freq:
            max_freq = v
            max_class = k

        total += v

    return (max_class, total == max_freq)

# uses information gain
# returns attr with largest gain else None if all gain < threshold
def select_split_attr_reg(D, A,c_name,threshold):
    
    #TODO: build get_entropy()
    #Get entropy of the data set, pass in the class column name
    entropy = entropy_d(D[c_name])
    max_gain = threshold
    best_attr = None

    for attr in A:
        #TODO: build get_entropy()
        #We are doing a groupby by the attribute and getting
        #a dataframe with the class frequencies
        SD = D.groupby([attr,c_name]).size().to_frame()
        SD.reset_index(inplace=True)
        SD.rename(columns = {0:"count"},inplace = True)
        #Entropy of the attribute
        entropy_of_split = entropy_a(SD, attr)
        gain = entropy - entropy_of_split
        
        if gain > max_gain:
            max_gain = gain
            best_attr = attr

    return best_attr

# uses information gain ratio
def select_split_attr_ratio(D, A, c_name, threshold):

    #TODO: build get_entropy()
    entropy = entropy_d(D[c_name])
    max_gainRatio = threshold
    best_attr = None

    for attr in A:
        SD = D.groupby([attr,c_name]).size().to_frame()
        SD.reset_index(inplace=True)
        SD.rename(columns = {0:"count"},inplace = True)
        
        #TODO: build get_entropy()
        entropy_of_split = entropy_a(SD, attr)
        gain = entropy - entropy_of_split
        gainRatio = gain / entropy_of_split
        
        if gainRatio > max_gainRatio:
            max_gainRatio = gainRatio
            best_attr = attr

    return best_attr

def build_decision_tree(dataset, attributes, tree, threshold, c_name):
    most_freq_class, is_only_class = get_most_freq_class(dataset,c_name)

    if is_only_class or len(attributes) == 0:
        leaf = Node(most_freq_class)
        tree = leaf
        return tree
    else:
        split_attr = select_split_attr_reg(dataset, attributes,
                                           c_name,threshold)

        if split_attr is None:
            leaf = Node(most_freq_class)
            tree = leaf
            return tree
        else:
            parent = Node(split_attr)
            
            attr_val_to_data = {}            
            for index,data in dataset.iterrows():
                # TODO: correct indexing
                attr_val = data[split_attr]

                if attr_val not in attr_val_to_data:
                    attr_val_to_data[attr_val] = []
                attr_val_to_data[attr_val].append(data)
            for k, v in attr_val_to_data.items():
                
                #new_df = pd.DataFrame(data = v,columns = attributes + [c_name])
                #print(new_df.columns)
                child = build_decision_tree(pd.DataFrame(data = v,columns = attributes + [c_name]), 
                                            [attr for attr in attributes if attr != split_attr],
                                            None,threshold,c_name)
                child.parent = parent
                child.edge = k

            return parent

def setup():
    y_vars = {}
    df = pd.read_csv("data/tree01-1000-words.csv",skiprows=[1,2])
    df.drop("Id",inplace = True,axis = 1)
    for col in df.columns:
        y_vars[col] = list(set(df[col]))
    y_vars
    return[df,y_vars]

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

if __name__ == '__main__':
    res = setup()
    df = res[0]
    tree = build_decision_tree(df,list(df.columns[:-1]),None,.01,"Vote")
    exporter = JsonExporter(indent=2)
    print(exporter.export(tree))
    #print(RenderTree(tree))
