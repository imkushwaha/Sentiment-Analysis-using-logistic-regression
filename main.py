import pickle
import numpy as np
from DataGetter import get_data
from utils import build_freqs
from utils import gradientDescent
from Embedding import word_2_vec
from LogisticRegression import predict_tweet
from LogisticRegression import test_logistic_regression
from Config import Config

# get data
train_x, train_y, test_x, test_y = get_data()
print("Data Loaded and divided successfully!!\n")

# create frequency dictionary
freq = build_freqs(train_x, train_y)
print("Frequency dictionary created successfully!!\n")

# extracting features and converting them into vectors for training data
X = word_2_vec(train_x, freq)
print("Features extracted and encoded into vectors\n")

# training labels corresponding to X
Y = train_y

# model training
cost, theta = gradientDescent(X, Y, theta=Config.theta, alpha=Config.alpha, num_iters=Config.num_iters)

print("Training Completed!!\n")
print(f"The cost after training is {cost:.8f}.\n")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")

print("--------------------------------------Testing with Test Data--------------------------------------------\n")

accuracy = test_logistic_regression(test_x, test_y, freq, theta, predict_tweet=predict_tweet)
print("Accuracy is:", accuracy)

# pickle frequency dictionary and final theta values for inference purpose
pickle.dump(freq, open('frequency.pkl', 'wb'))
pickle.dump(theta, open('weights.pkl', 'wb'))

print("Frequency dictionary and final weights saved!!")
