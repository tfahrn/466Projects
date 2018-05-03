import pandas as pd
import numpy as np
import math
from anytree import Node, RenderTree
from anytree.exporter import JsonExporter
import sys
import argparse


# returns tuple: (most frequent class in D, there is only one class in D)
def get_most_freq_class(D, c_name):
    class_values = D[c_name].values
    class_to_freq = {}
    
    for data in class_values:
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
def select_split_attr(D, A, c_name, threshold, use_ratio, is_cont):
    
    entropy = entropy_dataset(D, c_name)
    max_gain = threshold
    best_attr = None
    best_split_value = 0

    for attr in A:
        if is_cont:
            entropy_of_attr, split_value = entropy_attr_cont(D, c_name, attr)
        else:
            entropy_of_attr = entropy_attr(D, c_name, attr)
        gain = entropy - entropy_of_attr

        if use_ratio:
            if entropy_of_attr == 0.0:
                gain = sys.maxsize
            else:
                gain = gain/entropy_of_attr
        
        if gain > max_gain:
            max_gain = gain
            best_attr = attr
            best_split_value = split_value

    #print(best_attr, best_split_value)
    return best_attr, best_split_value

def entropy_dataset(df, category_variable):
    entropy = 0

    class_counts = df[category_variable].value_counts().tolist()
    total_count = np.sum(class_counts)
    for class_count in class_counts:
        prob_of_class = class_count/total_count
        entropy += prob_of_class * math.log(prob_of_class, 2)
    entropy *= -1

    return entropy

def entropy_attr(df, category_variable, attr):
    entropy = 0

    value_to_count = df[attr].value_counts().to_dict()
    total_count = sum(value_to_count.values())
    for value, count in value_to_count.items():
        filtered_df = df[df[attr] == value]
        filtered_entropy = entropy_dataset(filtered_df, category_variable)
        entropy += filtered_entropy * (count/total_count)
        #print("filtered_entropy:", filtered_entropy)
        #print("count/total_count:", count/total_count)

    return entropy

def entropy_attr_cont(df, category_variable, attr):
    entropy = 0

    value_to_count = df[attr].value_counts().to_dict()
    total_count = sum(value_to_count.values())
    cum_count = 0
    max_entropy = 0
    max_split = 0

    for value in sorted(value_to_count.keys()):
        cum_count = value_to_count[value]
        filtered_df = df[df[attr] <= value] 
        filtered_entropy = entropy_dataset(filtered_df, category_variable)
        entropy += filtered_entropy * (cum_count/total_count)
        if entropy > max_entropy:
            max_entropy = entropy
            max_split = value

    #print("max_entropy", max_entropy, "max_split", max_split, "entropy:", entropy)
    return entropy, max_split

def build_decision_tree(dataset, attributes, tree, threshold, c_name, use_ratio, is_cont):
    most_freq_class, is_only_class = get_most_freq_class(dataset,c_name)

    if is_only_class or len(attributes) == 0:
        leaf = Node(most_freq_class)
        tree = leaf
        return tree
    else:
        split_attr, split_value = select_split_attr(dataset, attributes, c_name, threshold, use_ratio, is_cont)

        if split_attr is None:
            leaf = Node(most_freq_class)
            tree = leaf
            return tree
        else:
            parent = Node(split_attr)
            
            attr_val_to_data = {}            
            for index,data in dataset.iterrows():
                attr_val = data[split_attr]
    
                if attr_val not in attr_val_to_data:
                    attr_val_to_data[attr_val] = []
                attr_val_to_data[attr_val].append(data)
            for k, v in attr_val_to_data.items():
                child = build_decision_tree(pd.DataFrame(data = v,
                    columns = attributes + [c_name]),
                    [attr for attr in attributes if attr != split_attr],
                    None,threshold,c_name, use_ratio, is_cont)
                child.parent = parent
                child.edge = k

            return parent

def get_args():
    parser = argparse.ArgumentParser(description='Build Decision Tree Input Parameters, see README')
    parser.add_argument('-x', '--csv', required=True, help="Path to csv file of training entries")
    parser.add_argument('-z', '--res', required=False, help="Path to optional restrictions file")

    return vars(parser.parse_args())

def preprocess(csv_file, res_file):
    is_iris = False
    with open(csv_file) as f:
        if "Iris" in f.readline():
            is_iris = True
    category_variable = "Class" if is_iris else "Vote"
    if is_iris:
        df = pd.read_csv(csv_file, names = ["Sepal Length", "Sepal Width",
                                            "Pedal Length", "Pedal Width",
                                            "Class"])
    else:
        df = pd.read_csv(csv_file, skiprows=[1,2])

    if res_file:
        with open(res_file) as f:
            res = f.readline().split(',')
            cols_to_drop = [i for i in range(len(res)) if int(res[i]) == 0]
            df.drop(df.columns[cols_to_drop], inplace=True, axis=1)

    if not is_iris:
        df.drop("Id", inplace=True, axis=1)

    return df, category_variable

if __name__ == '__main__':
    args = get_args()
    csv_file = args['csv']
    res_file = args['res']

    df, category_variable = preprocess(csv_file, res_file)
    is_cont = category_variable == "Class"
    tree = build_decision_tree(df, list(df.columns[:-1]), None, .01,
                               category_variable, False, is_cont)

    exporter = JsonExporter(indent=2)
    print(exporter.export(tree))
    #print(RenderTree(tree))
