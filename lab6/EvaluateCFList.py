import Filter
import argparse
import random
import sys
import statistics


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate Collaborative Filtering Random')
    parser.add_argument('-f', '--filepath', help='path to joke data file', required=True)
    parser.add_argument('-m', '--method', help='CF method: mean | weighted | knn', required=True)
    parser.add_argument('-t', '--testfilepath', help='path to test file', required=True)

    return vars(parser.parse_args())


# returns pairs (userId, itemId) for valid test cases
def get_test_cases(matrix, filepath):
    test_cases = []

    with open(filepath) as f:
        for line in f:
            line = line.split(', ')
            userId = int(line[0])
            itemId = int(line[1])

            if matrix[userId, itemId] != 99:
                test_cases.append((userId, itemId))

    return test_cases


# returns prediction for given test_case and method
def predict(method_name, test_case, source_user_ids, matrix, jokes, users,
            corr_matrix):
    if method_name == 'mean':
        prediction = Filter.predict_mean_utility(matrix, jokes, users, test_case[0],
                                                 test_case[1])
    elif method_name == 'weighted':
        prediction = Filter.predict_weighted_sum(matrix, jokes, users, corr_matrix,
                                                 test_case[0], test_case[1])
    else:
        k = 1000 
        userId_to_knn = Filter.get_knn(corr_matrix, k, source_user_ids)
        prediction = Filter.predict_knn_weighted_sum(matrix, jokes, users,
                                                     corr_matrix, userId_to_knn,
                                                     k, test_case[0], test_case[1])
    return prediction


"""
 receives csv file of list of (user, item) test cases
 for all valid test cases, displays prediction using input method
 dispalys mae over all valid test cases
"""
def main():
    args = get_args()
    filepath = args['filepath']
    method = args['method']
    test_filepath = args['testfilepath']


    if method != 'mean' and method != 'weighted' and method != 'knn':
        print("Incorrect method choice; choose one of: mean | weighted | knn")
        sys.exit(0)

    matrix, jokes, users = Filter.get_ratings(filepath)
    corr_matrix = Filter.get_pearson_corr(users)

    test_cases = get_test_cases(matrix, test_filepath)
    source_user_ids = [test_case[0] for test_case in test_cases]
    
    total_absolute_error = 0

    for test_case in test_cases:
        prediction = predict(method, test_case, source_user_ids, matrix, jokes,
                             users, corr_matrix)
        actual = matrix[test_case[0], test_case[1]]
        absolute_error = abs(prediction-actual)

        print("{}, {}, {}, {}, {}".format(test_case[0], test_case[1], actual,
                                          prediction, absolute_error))

        total_absolute_error += absolute_error

    mae = total_absolute_error / len(test_cases)
    print(mae)


if __name__ == '__main__':
    main()
