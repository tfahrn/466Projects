import argparse
import numpy as np


class Author:
    def __init__(self, name):
        self.name = name
        self.hits = 0
        self.strikes = 0
        self.misses = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0

    def calc_stats(self):
        self.precision = self.hits/(self.hits+self.strikes)
        self.recall = self.hits/(self.hits+self.misses)
        self.f1 = 2*self.hits/(2*self.hits+self.strikes+self.misses)


def get_args():
    parser = argparse.ArgumentParser(description='evaluates knn classification')
    parser.add_argument('-p', '--pred', help='predictions output file from knnAuthorship', required=True)
    parser.add_argument('-g', '--ground', help='file of ground truth', required=True)

    return vars(parser.parse_args())


def get_file_to_author(input_file):
    file_to_author = {}

    with open(input_file, 'r') as file:
        for line in file:
            filename = line.split(',')[0].rstrip()
            author = line.split(',')[1].rstrip()

            file_to_author[filename] = author

    return file_to_author


# returns map from author to a list of files they wrote
def get_author_to_files(input_file):
    author_to_files = {}

    with open(input_file, 'r') as file:
        for line in file:
            filename = line.split(',')[0].rstrip()
            author = line.split(',')[1].rstrip()

            if author in author_to_files:
                author_to_files[author].append(filename)
            else:
                author_to_files[author] = [filename]

    return author_to_files


def evaluate(pred_file_to_author, pred_author_to_files, true_author_to_files):
    num_correct_preds = 0
    num_incorrect_preds = 0
    matrix = np.zeros((50, 50))
    author_objects = [None]*50

    for i, (author, files) in enumerate(true_author_to_files.items()):
        author_objects[i] = Author(author)
        hits = strikes = misses = 0
        pred_files = pred_author_to_files[author]

        for file in files:
            if file in pred_files:
                num_correct_preds += 1
                hits += 1
            else:
                misses += 1
        
        strikes = len(pred_files) - hits

        author_objects[i].hits = hits
        author_objects[i].strikes = strikes
        author_objects[i].misses = misses 

    for i, (author, files) in enumerate(true_author_to_files.items()):
        pred_files = pred_author_to_files[author]

        for file in files:
            if file in pred_files:
                matrix[i][i] += 1
            else:
                predicted_author = pred_file_to_author[file]
                pred_author_index = 0
                for author_index, author_object in enumerate(author_objects):
                    if author_object.name == predicted_author:
                        pred_author_index = author_index 
                        break

                matrix[pred_author_index][i] += 1

    num_incorrect_preds = 5000 - num_correct_preds
    matrix = np.ndarray.tolist(matrix)

    return num_correct_preds, num_incorrect_preds, matrix, author_objects


def main():
    args = get_args()
    predictions_file = args['pred']
    ground_file = args['ground']

    pred_file_to_author = get_file_to_author(predictions_file)
    pred_author_to_files = get_author_to_files(predictions_file)
    true_author_to_files = get_author_to_files(ground_file)

    num_cor, num_inc, matrix, author_objects = evaluate(pred_file_to_author, pred_author_to_files, true_author_to_files)

    print("Number of documents with correctly predicted documents:", num_cor)
    print("Number of documents with incorrectly predicted documents:", num_inc)

    for author in author_objects:
        print("Name: {:20}\tHits: {}\tStrikes: {}\tMisses: {}\tPrecision: {}\tRecall: {}\tF1: {}".format(author.name, author.hits, author.strikes, author.misses, author.precision, author.recall, author.f1))

    author_names = [a.name for a in author_objects]
    print(author_names)
    for row in matrix:
        print(row)

if __name__ == '__main__':
    main()
