import argparse
import InduceC45
import Classify
import random
import Validate

def dataset_selection(df, num_attr, num_data, category_variable):
    selected_attr = random.sample(list(df.columns[:-1]), num_attr)
    df = df.sample(n=num_data)
    df = df[selected_attr + [category_variable]]

    return df

def plurality(l):
    l = [x for x in l if x is not None]
    most_freq = max(set(l), key=l.count)
    return most_freq

def get_args():
    parser = argparse.ArgumentParser(description='Random Forest Input Parameters, see README')
    parser.add_argument('-x', '--csv', required=True, help="Path to csv file of training entries")
    parser.add_argument('-m', '--numAttr', required=True, help="Number of attributes for each decision tree")
    parser.add_argument('-k', '--numData', required=True, help="Number of data points for each decision tree")
    parser.add_argument('-n', '--numTree', required=True, help="Number of decision trees to build")

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = get_args()
    csv_file = args['csv']
    num_attr = int(args['numAttr'])
    num_data = int(args['numData'])
    num_tree = int(args['numTree'])
    num_correct, num_incorrect, true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0, 0, 0

    df, category_variable = InduceC45.preprocess(csv_file, None)
    is_cont = category_variable == "Class"
    all_preds = []

    for i in range(num_tree):
        tree_df = dataset_selection(df, num_attr, num_data, category_variable)
        tree = InduceC45.build_decision_tree(tree_df, list(tree_df.columns[:-1]), None, 0.01,
                                             category_variable, False, is_cont)
        preds, num_correct, num_incorrect = Classify.main(csv_file, tree)
        c_num_correct, c_num_incorrect, c_true_pos, c_true_neg, c_false_pos, c_false_neg = Validate.main(tree_df, 10, category_variable, is_cont)

        num_correct += c_num_correct
        num_incorrect += c_num_incorrect
        true_pos += c_true_pos
        true_neg += c_true_neg
        false_pos += c_false_pos
        false_neg += c_false_neg
        all_preds.append(preds)

    all_preds = [list(pred) for pred in zip(*all_preds)]
    final_preds = [plurality(pred) for pred in all_preds]

    print(','.join(final_preds))
    print("Matrix:")
    print("[", true_pos, false_neg, "]")
    print("[", false_pos, true_neg, "]")
    print("Accuracy:", num_correct/(num_correct+num_incorrect))
