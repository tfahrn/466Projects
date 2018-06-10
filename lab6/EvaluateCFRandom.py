import Filter
import argparse
import random
import sys
import statistics


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate Collaborative Filtering Random')
    parser.add_argument('-f', '--filepath', help='path to joke data file', required=True)
    parser.add_argument('-m', '--method', help='CF method: mean | weighted | knn', required=True)
    parser.add_argument('-s', '--size', help='number of predictions to evaluate', required=True)
    parser.add_argument('-r', '--repeats', help='number of times the test is repeated', required=True)

    return vars(parser.parse_args())


# returns pairs (userId, itemId) for valid test cases
def get_test_cases(matrix, size):
    test_cases = []
    num = 0
    num_users = len(matrix)
    num_items = len(matrix[0])

    while num < size:
        rand_userId = random.randint(0, num_users-1)
        rand_itemId = random.randint(0, num_items-1)

        if matrix[rand_userId, rand_itemId] != 99:
            test_cases.append((rand_userId, rand_itemId))
            num += 1

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
 executes "repeats" number of trials. each trial find "size" (user, item) test cases
 and makes a prediction. each prediction is shown, each trial's mae is shown.
 the average and std mae over all trials is shown
"""
def main():
    args = get_args()
    filepath = args['filepath']
    method = args['method']
    size = int(args['size'])
    repeats = int(args['repeats'])

    if method != 'mean' and method != 'weighted' and method != 'knn':
        print("Incorrect method choice; choose one of: mean | weighted | knn")
        sys.exit(0)

    matrix, jokes, users = Filter.get_ratings(filepath)
    corr_matrix = Filter.get_pearson_corr(users)

    maes = []

    for run in range(repeats):
        print("Run: {}".format(run+1))

        test_cases = get_test_cases(matrix, size)
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

        mae = total_absolute_error / size
        maes.append(mae)
        print(mae)
        print("-------------------------------------------------------------------")

    print("Mean MAE: {}".format(round(statistics.mean(maes), 3)))
    if len(maes) > 1:
        print("Standard Deviation of MAE: {}".format(statistics.stdev(maes)))
    else:
        print("Not enough repeats for std of MAE")


if __name__ == '__main__':
    main()
