import InduceC45
import sys
import pandas as pd
from anytree import Node, RenderTree
from sklearn.model_selection import KFold

"""
TODO: handle n=-1, n=0




"""

def main(csv_file, n):
    df = pd.read_csv(csv_file, skiprows=[1,2])
    # shuffle
    df = df.sample(frac=1)
    df.drop("Id", inplace = True, axis=1)
    attr = [col for col in df.columns]

    kf = KFold(n_splits = n, shuffle = True)
    for fold in kf.split(df):
        train_data = df.iloc[fold[0]]
        val_data = df.iloc[fold[1]]
        tree = InduceC45.build_decision_tree(train_data, attr[:-1], None, 0.01, attr[-1]) 
        print(RenderTree(tree))



if len(sys.argv) > 2:
    csv_file = sys.argv[1]
    n = int(sys.argv[2])
    if len(sys.argv) > 3:
        z = sys.argv[3]

    main(csv_file, n)
