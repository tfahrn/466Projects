import argparse

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

    print(csv_file, num_attr, num_data, num_tree)
