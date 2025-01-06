# https://medium.com/@zhonghong9998/quantitative-finance-predicting-stock-prices-with-python-and-deep-learning-06a9538377c9
# https://arxiv.org/abs/2311.06273

import tweepy
from textblob import TextBlob
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np

# Twitter API credentials (replace with your own)
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'

# Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Fetch tweets related to a stock (for example, Apple)
public_tweets = api.search('Apple stock')

# Perform sentiment analysis
def get_sentiment(tweet):
    analysis = TextBlob(tweet.text)
    return analysis.sentiment.polarity

# Create a DataFrame to store the tweets and sentiments
tweets_data = pd.DataFrame(columns=['Tweet', 'Sentiment'])

for tweet in public_tweets:
    sentiment = get_sentiment(tweet)
    tweets_data = tweets_data.append({'Tweet': tweet.text, 'Sentiment': sentiment}, ignore_index=True)

# Now let's build a simple LSTM model to predict stock trends based on sentiment

# Prepare the data for the LSTM model
data = tweets_data['Sentiment'].values
data = np.reshape(data, (len(data), 1, 1))

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(data.shape[1], data.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
model.fit(data, data, epochs=10, batch_size=32)

# Make predictions (dummy prediction in this case)
predictions = model.predict(data)

print("Sentiment Analysis and Predictions Complete!")
