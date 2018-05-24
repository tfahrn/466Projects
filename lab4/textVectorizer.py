import argparse
import os
import string
import stemmer
import math

"""
TODO:
    normalization?

"""


class Vector:
    def __init__(self, filename, author, tf, threshold):
        self.filename = filename
        self.author = author
        self.tf = tf
        self.tf_idf = {} 
        self.threshold = threshold


# constructs tf_idf {word : tf-idf} for each vector using vector.tf and vocab
def set_tf_idfs(vectors, vocab, num_docs):
    # threshold and normalize each vector
    for vector in vectors:
        for word, freq in vector.tf.items():
            if vector.threshold > 0 and vector.tf[word] > vector.threshold:
                vector.tf[word] = vector.threshold
        max_freq_word = max(vector.tf, key=lambda freq: vector.tf[freq])
        max_freq = vector.tf[max_freq_word]
        for word, freq in vector.tf.items():
            vector.tf[word] = freq/max_freq

    for word, doc_freq in vocab.items():
        idf = math.log(num_docs/vocab[word], 2)

        for vector in vectors:
            if word in vector.tf:
                vector.tf_idf[word] = vector.tf[word] * idf 
            else:
                vector.tf_idf[word] = 0


def get_args():
    parser = argparse.ArgumentParser(description='tf-idf vectorizer')
    parser.add_argument('-d', '--dir', help='root of 50-50 dataset directory', required=True)
    parser.add_argument('-o', '--output', help='name of output file for tf-idf representations', required=True)
    parser.add_argument('-g', '--ground', help='name of output file for ground truth', required=True)
    parser.add_argument('-s', '--stem', help='boolean; whether to include stemming in preprocessing', required=True)
    parser.add_argument('-r', '--remove', help='boolean; whether to remove stop words in preprocessing', required=True)
    parser.add_argument('-t', '--threshold', help='integer; threshold value for term frequency; enter 0 to not use a threshold', required=True)

    return vars(parser.parse_args())


# creates mapping of {txt_file_path : author}
# writes ground truth to filename in format 'txt_file_name, author'
# returns map
def write_ground_truth(root_path, filename):
    path_to_author = {}

    for data_dir in os.scandir(root_path):
        if not data_dir.name.startswith('.'):
            for author_dir in os.scandir(data_dir.path):
                if not author_dir.name.startswith('.'):
                    for news_file in os.scandir(author_dir.path):
                        if not news_file.name.startswith('.'):
                            path_to_author[news_file.path] = author_dir.name

    with open(filename, 'w') as ground_truth_file:
        for txt_file_path, author in path_to_author.items():
            txt_file_name = os.path.basename(txt_file_path)
            print('{},{}'.format(txt_file_name, author), file=ground_truth_file)

    return path_to_author


# returns set of stopwords
def get_stopwords():
    with open('stopwords.txt', 'r') as file:
        stopwords = {word for line in file for word in line.split()}

    return stopwords

# creates, returns tf {term : frequency} of words in file_path 
def file_to_tf(file_path, to_stem, to_remove):
    tf = {}

    if to_remove:
        stopwords = get_stopwords()

    with open(file_path, 'r', encoding='utf8', errors='ignore') as file:
        for line in file:
            for word in line.split():
                word = (word.strip(string.punctuation)).lower()

                if word.isnumeric() or word.isdigit():
                    continue

                if to_remove:
                    if word in stopwords:
                        continue
                    
                if to_stem:
                    porter_stemmer = stemmer.PorterStemmer()
                    word = porter_stemmer.stem(word, 0, len(word)-1)

                if word in tf:
                    tf[word] += 1
                else:
                    tf[word] = 1

    return tf


# iterates through each file in path_to_author, constructing vocab {term : corpus_freq} and a vector for each file
# defines tf-idf of each vector
# returns vocab, list of all vectors
def tf_idf(path_to_author, to_stem, to_remove, threshold):
    vocab = {}
    vectors = []

    for txt_file_path, author in path_to_author.items():
        txt_file_name = os.path.basename(txt_file_path)
        tf = file_to_tf(txt_file_path, to_stem, to_remove)

        for word, freq in tf.items():
            if word in vocab:
                vocab[word] += freq
            else:
                vocab[word] = freq
        
        vector = Vector(txt_file_name, author, tf, threshold)
        vectors.append(vector)

    set_tf_idfs(vectors, vocab, len(path_to_author))

    return vocab, vectors


def main():
    args = get_args()
    root_path = args['dir']
    output_file = args['output']
    ground_file = args['ground']
    to_stem = True if args['stem']=='True' else False
    to_remove = True if args['remove']=='True' else False
    threshold = int(args['threshold'])

    path_to_author = write_ground_truth(root_path, ground_file)
    vocab, vectors = tf_idf(path_to_author, to_stem, to_remove, threshold)
    #file_to_tf('data/C50/C50train/JoWinterbottom/144390newsML.txt', to_stem, to_remove)

    for vector in vectors:
        print(vector.tf_idf)


if __name__ == '__main__':
    main()
