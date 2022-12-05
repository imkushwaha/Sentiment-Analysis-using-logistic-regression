import numpy as np
from utils import extract_features


def word_2_vec(train_x, freq):
    """
    collect the features 'x' and stack them into a matrix 'x'
    input:
         train_x and frequency count of every word for positive and negative sentiment
    output:
         vector for each tweet .i.e.. [1.0,0.0,245]

    """
    x = np.zeros((len(train_x), 3))
    for i in range(len(train_x)):
        x[i, :] = extract_features(train_x[i], freq)

    return x

