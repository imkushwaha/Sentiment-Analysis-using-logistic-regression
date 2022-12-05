import nltk
from nltk.corpus import twitter_samples
import numpy as np
import os

def get_data():
    """Download Data from NLTK Corpus and divide it into train and test data set.

       Return: train_x, train_y, test_x, test_y
    """
    # Download twitter samples and stopwords
    nltk.download('twitter_samples')
    nltk.download('stopwords')

    filepath = f"{os.getcwd()}/../tmp2/"
    nltk.data.path.append(filepath)

    # select the set of positive and negative tweets
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')

    # Train test split: 20% will be in the test set, and 80% in the training set
    # split the data into two pieces, one for training and one for testing (validation set)

    test_pos = all_positive_tweets[4000:]
    train_pos = all_positive_tweets[:4000]
    test_neg = all_negative_tweets[4000:]
    train_neg = all_negative_tweets[:4000]

    train_x = train_pos + train_neg
    test_x = test_pos + test_neg

    # Create the numpy array of positive labels and negative labels
    # combine positive and negative labels
    train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
    test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

    # Print the shape train and test sets
    print("train_y.shape = " + str(train_y.shape))
    print("test_y.shape = " + str(test_y.shape))

    return train_x, train_y, test_x, test_y

