from anytree.importer import JsonImporter
from anytree import RenderTree
import pandas as pd
import sys

def tree_from_file(file):
    importer = JsonImporter()
    data = open(file, 'r').read()
    root = importer.import_(data)
    
    return root

def get_pred(tree, df, row):
    if tree.name == "Obama" or tree.name == "McCain":
        return tree.name
    
    split = tree.name
    row_value = row[df.columns.get_loc(split) + 1]
    
    for child in tree.children:
        if child.edge == row_value:
            return get_pred(child, df, row)

def main(csv_file, json_file):
    """
    json_file = "data/tree01-1000.json"
    csv_file = "data/tree01-1000-words.csv"
    """
    category_variable = "Vote"
    
    df = pd.read_csv(csv_file,skiprows=[1,2])
    df.drop("Id",inplace = True,axis = 1)
    tree = tree_from_file(json_file)
    
    preds = []
    num_correct = 0
    num_incorrect = 0
    for row in df.itertuples():
        pred = get_pred(tree, df, row)
        preds.append(pred)
        
        if category_variable in df.columns:
            index = df.columns.get_loc(category_variable)
            if pred == row[index+1]:
                num_correct += 1
            else:
                num_incorrect += 1
    
    #print(preds)
    print("Number of records classified:", len(preds))
    if category_variable in df.columns:
        print("Number of records correctly classified:", num_correct)
        print("Number of records incorrectly classified:", num_incorrect)
        print("Accuracy:", num_correct/(num_correct + num_incorrect))
        print("Error rate:", 1 - num_correct/(num_correct + num_incorrect))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        json_file = sys.argv[2]
    main(csv_file, json_file)
