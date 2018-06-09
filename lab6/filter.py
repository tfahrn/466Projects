import argparse
import pandas as pd
import numpy as np
import math
import queue

import time

class User:
    # ratings is an ordered (by joke) list of ratings (including unrated/99)
    def __init__(self, num_ratings, ratings):
        self.num_ratings = num_ratings
        self.ratings = ratings
        self.num_ratings = 0
        self.sum_ratings = 0

        for rating in ratings:
            if rating != 99:
                self.num_ratings += 1
                self.sum_ratings += rating

        self.mean_rating = self.sum_ratings / self.num_ratings


class Joke:
    # ratings is an ordered (by user) list of ratings
    def __init__(self, ratings):
        self.ratings = ratings
        self.num_users = len(self.ratings)
        self.num_ratings = 0
        self.sum_ratings = 0

        for rating in ratings:
            if rating != 99:
                self.num_ratings += 1
                self.sum_ratings += rating

        self.mean_rating = self.sum_ratings / self.num_ratings
        self.inverse_user_freq = math.log2(self.num_users / self.num_ratings)

        # print("Avg rating: {}\t IUF: {}\t Weighted Rating: {}".format(self.mean_rating, self.inverse_user_freq, self.mean_rating*self.inverse_user_freq))


def get_args():
    parser = argparse.ArgumentParser(description='Collaborative Filtering')
    parser.add_argument('-f', '--filepath', help='path to joke data file', required=True)

    return vars(parser.parse_args())


# parses csv
# constructs [jokes] and [users]
# returns matrix, jokes, users
def get_ratings(filepath):
    df = pd.read_csv(filepath, sep=',', header=None)
    num_ratings_per_user = df[0]
    df = df.drop(df.columns[0], axis=1)

    jokes = []
    users = []

    for joke in df:
        joke_ratings = np.array(df[joke])
        jokes.append(Joke(joke_ratings))

    for i, user in df.iterrows():
        user_ratings = np.array(user)
        users.append(User(num_ratings_per_user[i], user_ratings))

    return df.values, jokes, users


# inputs: list of all user's ratings
# transforms 99 ratings to 0 ratings 
# returns Pearson Correlation [-1, 1] Matrix
def get_pearson_corr(users):
    dense_user_ratings = []

    for user in users:
        ratings = np.array([rating if rating != 99 else 0 for rating in user.ratings])
        dense_user_ratings.append(ratings)

    corr_matrix = np.corrcoef(dense_user_ratings)

    return corr_matrix


# inputs: Pearson Correlation Matrix, k
# returns: map from userId : [userId] of nearest neighbors
# uses all neighbors instead of k incase neighbor doesn't share same items rated
def get_knn(corr_matrix, k, source_user_ids):
    user_to_knn = {}

    # for userId, user in enumerate(corr_matrix):
    for userId in source_user_ids:

        """
        # min priority queue used to find knn
        pq = queue.PriorityQueue(len(user))
        for neighborId, neighbor_sim in enumerate(user):
            if userId != neighborId:
                priority = 1 - abs(neighbor_sim)
                pq.put((priority, neighborId))

        knn = []
        while pq.full():
            neighborId = pq.get()[1]
            knn.append(neighborId)

        user_to_knn[userId] = knn
        """

        user = corr_matrix[userId]
        sorted_neighbors = sorted(range(len(user)), key=lambda k: 1-abs(user[k]))
        # remove self (perfect correlation)
        user_to_knn[userId] = sorted_neighbors[1:]

        print("found neighbors for userId:", userId)

    return user_to_knn


def predict_mean_utility(matrix, jokes, users, userId, itemId):
    user = users[userId]
    joke = jokes[itemId]
    prev_rating = matrix[userId, itemId]

    if prev_rating != 99:
        matrix[userId, itemId] = 99

    pred_rating = joke.mean_rating

    if pred_rating > 10:
        pred_rating = 10
    elif pred_rating < -10:
        pred_rating = -10

    # restore only previously not 99?
    matrix[userId, itemId] = prev_rating

    return pred_rating


def predict_weighted_sum(matrix, jokes, users, corr_matrix, userId, itemId):
    user = users[userId]
    joke = jokes[itemId]
    prev_rating = matrix[userId, itemId]

    if prev_rating != 99:
        matrix[userId, itemId] = 99

    sum_of_weights = 0
    sum_of_products = 0

    for i, user_rating in enumerate(joke.ratings):
        if user_rating != 99:
            sim = corr_matrix[userId, i]
            sum_of_weights += abs(sim)
            sum_of_products += sim * user_rating

    pred_rating = (1/sum_of_weights) * sum_of_products

    if pred_rating > 10:
        pred_rating = 10
    elif pred_rating < -10:
        pred_rating = -10

    # restore only previously not 99?
    matrix[userId, itemId] = prev_rating

    return pred_rating


def predict_knn_weighted_sum(matrix, jokes, users, corr_matrix, userId_to_knn, k,
                             userId, itemId):
    user = users[userId]
    joke = jokes[itemId]
    prev_rating = matrix[userId, itemId]

    if prev_rating != 99:
        matrix[userId, itemId] = 99

    sum_of_weights = 0
    sum_of_products = 0

    knn = userId_to_knn[userId]
    num_neighbors_added = 0

    for neighborId in knn:
        if num_neighbors_added == k:
            break
        user_rating = joke.ratings[neighborId]
        if user_rating != 99:
            sim = corr_matrix[userId, neighborId]
            sum_of_weights += abs(sim)
            sum_of_products += sim * user_rating
            num_neighbors_added += 1

    pred_rating = (1/sum_of_weights) * sum_of_products

    if pred_rating > 10:
        pred_rating = 10
    elif pred_rating < -10:
        pred_rating = -10

    # restore only previously not 99?
    matrix[userId, itemId] = prev_rating

    return pred_rating


def main():
    args = get_args()
    filepath = args['filepath']
    k = 4
    
    matrix, jokes, users = get_ratings(filepath)
    corr_matrix = get_pearson_corr(users)
    userId_to_knn = get_knn(corr_matrix, k, [0, 1, 3, 4])

    m_pred_rating = predict_mean_utility(matrix, jokes, users, 2, 5)
    print(m_pred_rating)

    w_pred_rating = predict_weighted_sum(matrix, jokes, users, corr_matrix, 2, 5)
    print(w_pred_rating)

    k_w_pred_rating = predict_knn_weighted_sum(matrix, jokes, users, corr_matrix, userId_to_knn, k, 2, 5)
    print(k_w_pred_rating)


if __name__ == '__main__':
    main()
