from LogisticRegression import predict_tweet
from utils import process_tweet
import pickle

freq = pickle.load(open('frequency.pkl', 'rb'))
theta = pickle.load(open('weights.pkl', 'rb'))


def prediction(tweet):
    print(f"Given tweet is: {tweet}\n")
    print(f"Processed Tweet: {process_tweet(tweet)}]\n")
    y_hat = predict_tweet(tweet, freq, theta)
    if y_hat > 0.5:
        return 'Positive sentiment'
    else:
        return 'Negative sentiment'


my_tweet = 'I hate this policy from government'
sentiment = prediction(my_tweet)
print(f"Given tweet is: {sentiment}")

