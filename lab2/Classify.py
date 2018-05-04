from anytree.importer import JsonImporter
from anytree import RenderTree
import pandas as pd
import sys
import argparse
import InduceC45

def tree_from_file(file):
    importer = JsonImporter()
    data = open(file, 'r').read()
    root = importer.import_(data)
    
    return root

def get_pred(tree, df, row):
    if len(tree.children) == 0:
        return tree.name
    
    split = tree.name
    row_value = row[df.columns.get_loc(split) + 1]
    
    for child in tree.children:
        if child.edge == row_value:
            return get_pred(child, df, row)

def main(csv_file, tree):
    df, category_variable = InduceC45.preprocess(csv_file, None)
    
    preds = []
    num_correct = 0
    num_incorrect = 0
    for row in df.itertuples():
        pred = get_pred(tree, df, row)
        if __name__ == '__main__':
            print(row[0:], ",", pred)
        preds.append(pred)
        
        if category_variable in df.columns:
            index = df.columns.get_loc(category_variable)
            if pred == row[index+1]:
                num_correct += 1
            else:
                num_incorrect += 1
    
    return preds, num_correct, num_incorrect

def get_args():
    parser = argparse.ArgumentParser(description='Random Forest Input Parameters, see README')
    parser.add_argument('-x', '--csv', required=True, help="Path to csv file of entries to classify")
    parser.add_argument('-y', '--tree', required=True, help="Path to JSON file of decision tree")

    return vars(parser.parse_args())

if __name__ == '__main__':
    args = get_args()
    csv_file = args['csv']
    tree_file = args['tree']
    tree = tree_from_file(tree_file)
    
    preds, num_corrcet, num_incorrect = main(csv_file, tree)
    
    print("Number of records classified:", len(preds))
    if category_variable in df.columns:
        print("Number of records correctly classified:", num_correct)
        print("Number of records incorrectly classified:", num_incorrect)
        print("Accuracy:", num_correct/(num_correct + num_incorrect))
        print("Error rate:", 1 - num_correct/(num_correct + num_incorrect))
