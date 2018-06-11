import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

def get_data():
    data = np.loadtxt(open('data/encoded_data.csv', "rb"), delimiter=",")
    labels = np.loadtxt(open('data/encoded_labels.csv', "rb"), delimiter=",")

    data_and_labels = np.concatenate((data, labels), axis=1)
    np.random.shuffle(data_and_labels)

    train_data, train_labels = data_and_labels[:135, :-6], data_and_labels[:135, -6:]
    test_data, test_labels = data_and_labels[135:, :-6], data_and_labels[135:, -6:]

    return train_data, train_labels, test_data, test_labels


def main():
    total_results = 0
    accuracies = []

    for i in range(100):
        train_data, train_labels, test_data, test_labels = get_data()

        model = models.Sequential()
        model.add(layers.Dense(32, activation='relu', input_shape=(40, )))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(6, activation='softmax'))

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(train_data, train_labels, epochs=20, batch_size=8, verbose=0)
        results = model.evaluate(test_data, test_labels)

        accuracies.append(results[1])
        total_results += results[1]

    print(total_results/100)

    sns.set_style('whitegrid')
    sns.distplot(accuracies, kde=False, rug=True)
    plt.title("Neural Network Classification of Turkey Political Parties")
    plt.xlabel("Classification Accuracy")
    plt.ylabel("Density over 100 Trials")
    plt.show()


if __name__ == '__main__':
    main()
