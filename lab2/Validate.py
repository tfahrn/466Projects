import InduceC45
import Classify
import sys
import pandas as pd
from anytree import Node, RenderTree
from sklearn.model_selection import KFold
import argparse

def main(df, n, category_variable, is_cont):
    # shuffle
    df = df.sample(frac=1)
    attr = [col for col in df.columns]
    is_binary = len(df[category_variable].value_counts()) == 2
    overall_num_correct = 0
    overall_num_incorrect = 0
    overall_true_pos = 0
    overall_true_neg = 0
    overall_false_pos = 0
    overall_false_neg = 0

    if n == -1:
        n = len(df) - 1

    try:
        kf = KFold(n_splits = n, shuffle = True).split(df)
    except:
        kf = [df]
    for i, fold in enumerate(kf):
        if n == 0:
            train_data = df
            val_data = df
        else:
            train_data = df.iloc[fold[0]]
            val_data = df.iloc[fold[1]]

        tree = InduceC45.build_decision_tree(train_data, attr[:-1], None, 0.01, category_variable, False, is_cont) 

        num_correct, num_incorrect, true_pos, true_neg, false_pos, false_neg \
            = 0, 0, 0, 0, 0, 0
        for row in val_data.itertuples():
            pred = Classify.get_pred(tree, val_data, row)
            if category_variable in val_data.columns:
                index = val_data.columns.get_loc(category_variable)
                if pred == row[index+1]:
                    num_correct += 1
                    if pred == "Obama":
                        true_pos += 1
                    else:
                        true_neg += 1
                else:
                    num_incorrect += 1
                    if pred == "Obama":
                        false_pos += 1
                    else:
                        false_neg += 1
       
        if __name__ == '__main__':
            print("Fold Number:", i+1)

            # try/except for div by 0
            try:
                accuracy = num_correct/(num_correct+num_incorrect) 
            except:
                accuracy = 1.0

            if is_binary:
                try:
                    precision = true_pos/(true_pos+false_pos)
                except:
                    precision = 1.0
                try:
                    recall = true_pos/(true_pos+false_neg)
                except:
                    recall = 1.0
                try:
                    pf = false_pos/(false_pos+true_neg)
                except:
                    pf = 1.0
                try:
                    f_measure = 2*precision*recall/(precision+recall)
                except:
                    f_measure = 1.0
                print("Matrix:")
                print("[", true_pos, false_neg, "]")
                print("[", false_pos, true_neg, "]")
                print("recall:", round(recall, 3))
                print("precision:", round(precision, 3))
                print("pf:", round(pf, 3))
                print("f-measure:", round(f_measure, 3))
            print("acc:", round(accuracy, 3))
            print("err:", round(1-accuracy, 3))
            print("\n")

        overall_num_correct += num_correct
        overall_num_incorrect += num_incorrect
        overall_true_pos += true_pos
        overall_false_pos += false_pos
        overall_true_neg += true_neg
        overall_false_neg += false_neg

    if __name__ == '__main__':
        if is_binary:
            try:
                precision = overall_true_pos/(overall_true_pos+overall_false_pos)
            except:
                precision = 1.0
            try:
                recall = overall_true_pos/(overall_true_pos+overall_false_neg)
            except:
                recall = 1.0
            try:
                pf = overall_false_pos/(overall_false_pos+overall_true_neg)
            except:
                pf = 1.0
            try:
                f_measure = 2*precision*recall/(precision+recall)
            except:
                f_measure = 1.0
            print("\n")
            print("overall recall:", round(recall, 3))
            print("overall precision:", round(precision, 3))
            print("overall pf:", round(pf, 3))
            print("overall f-measure:", round(f_measure, 3))

    return overall_num_correct, overall_num_incorrect, overall_true_pos, overall_true_neg, overall_false_pos, overall_false_neg

def get_args():
    parser = argparse.ArgumentParser(description='Random Forest Input Parameters, see README')
    parser.add_argument('-x', '--csv', required=True, help="Path to csv file of training entries")
    parser.add_argument('-n', '--numFolds', required=True, help="Number of folds for cross-validation")
    parser.add_argument('-z', '--res', required=False, help="Optional restrictions file")

    return vars(parser.parse_args())

if __name__ == '__main__':
    args = get_args()
    csv_file = args['csv']
    n = int(args['numFolds'])
    res_file = args['res']
    df, category_variable = InduceC45.preprocess(csv_file, res_file)
    is_cont = category_variable == "Class"

    num_correct, num_incorrect, true_pos, true_neg, false_pos, false_neg = main(df, n, category_variable, is_cont)

    print("Overall Matrix:")
    print("[", true_pos, false_neg, "]")
    print("[", false_pos, true_neg, "]")

    print("Overall Accuracy:", num_correct/(num_correct+num_incorrect))
    print("Overall Error:", round(1 - (num_correct/(num_correct+num_incorrect)), 3))
