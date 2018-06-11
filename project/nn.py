import numpy as np
from keras import models
from keras import layers


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

    for i in range(50):
        train_data, train_labels, test_data, test_labels = get_data()

        model = models.Sequential()
        model.add(layers.Dense(32, activation='relu', input_shape=(40, )))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(6, activation='softmax'))

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(train_data, train_labels, epochs=20, batch_size=8)
        results = model.evaluate(test_data, test_labels)

        total_results += results[1]

    print(total_results/50)

    






if __name__ == '__main__':
    main()
