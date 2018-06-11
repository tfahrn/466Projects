import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder


# one-hot encodes a line into a vector of dimension 40; excludes timestamp and party
def transform(row, value_to_index):
    encoding = np.zeros(40)

    for col_index, column in enumerate(row):
        # encodes indices [0, 19]
        if col_index < 4:
            one_hot_index = value_to_index[column]
        # encodes T/F
        else:
            offset = 0 if column == 'Evet' else 1
            one_hot_index = (col_index - 4)*2+offset+20

        encoding[one_hot_index] = 1

        if col_index == 13:
            break

    return encoding


def main():
    data = pd.read_csv('data/data.csv')
    data = data.drop('Timestamp', 1)

    value_to_index = {
        'Erkek': 0,
        'Kadın': 1,
        '0-18': 2,
        '18-30': 3,
        '30-50': 4,
        '50-60': 5,
        '60+': 6,
        'Marmara': 7,
        'Güneydoğu': 8,
        'Akdeniz': 9,
        'Doğu Anadolu': 10,
        'İç Anadolu': 11,
        'Karadeniz': 12,
        'Ege': 13,
        'Lisans': 14,
        'Lise': 15,
        'Ön Lisans': 16,
        'Lisans Üstü': 17,
        'İlkokul': 18,
        'Ortaokul': 19
    }


    encoded_lines = []
    for i, row in data.iterrows():
        encoded = transform(row, value_to_index)
        encoded_lines.append(encoded)
    
    # save one-hot encoding to file
    np.savetxt('data/encoded_data.csv', np.array(encoded_lines),
               delimiter=',', fmt='%s')

    # one-hot encodes political party / output
    labels = np.array(data['parti'])
    encoder = LabelEncoder()
    encoder.fit(labels)
    one_hot_labels = to_categorical(encoder.transform(labels))
    np.savetxt('data/encoded_labels.csv', one_hot_labels,
               delimiter=',', fmt='%s')


if __name__ == '__main__':
    main()
