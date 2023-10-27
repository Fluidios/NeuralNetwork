#region Importing
print('Importing...')

import pandas as pd
print('pandas import - done')
import numpy as np # math
print('numpy import - done')
import tflearn # neuro nets learning
print('tflearn import - done')
import tensorflow.compat.v1 as tf # neuro nets
print('tenserflow import - done')
import re # strings work
print('re import - done')
import time

from collections import Counter
from sklearn.model_selection import train_test_split
from tflearn.data_utils import to_categorical
from nltk.stem.snowball import RussianStemmer
from nltk.tokenize import TweetTokenizer

# print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
#endregion
 
#region Constants
POSITIVE_TWEETS_CSV = 'positive.csv'
NEGATIVE_TWEETS_CSV = 'negative.csv'

SAVED_NN = 'saved_nn.tflearn'

VOCAB_SIZE = 5000
#endregion
total_timer = time.time()
timer = time.time()
#region Load data
print('Reading input files...')
tweets_col_number = 3

negative_tweets = pd.read_csv(
    'negative.csv', header=None, delimiter=';')[[tweets_col_number]]
positive_tweets = pd.read_csv(
    'positive.csv', header=None, delimiter=';')[[tweets_col_number]]
print('Reading takes: {0}'.format(time.time()-timer))
#endregion

#region Stemmer
stemer = RussianStemmer()
regex = re.compile('[^а-яА-Я ]')
stem_cache = {}

def get_stem(token):
    stem = stem_cache.get(token, None)
    if stem:
        return stem
    token = regex.sub('', token).lower()
    stem = stemer.stem(token)
    stem_cache[token] = stem
    return stem
#endregion

#region Vocabulary creation
timer = time.time()
print('Creating vocabulary...')
stem_count = Counter()
tokenizer = TweetTokenizer()

def count_unique_tokens_in_tweets(tweets):
    for _, tweet_series in tweets.iterrows():
        tweet = tweet_series[3]
        tokens = tokenizer.tokenize(tweet)
        for token in tokens:
            stem = get_stem(token)
            stem_count[stem] += 1

count_unique_tokens_in_tweets(negative_tweets)
count_unique_tokens_in_tweets(positive_tweets)

print("Total unique stems found:{0}. Top {1} from them is our vocabulary.".format(len(stem_count), VOCAB_SIZE))
vocab = sorted(stem_count, key=stem_count.get, reverse=True)[:VOCAB_SIZE]
print('Vocabulary building takes: {0}'.format(time.time()-timer))
#endregion

#region Converting tweets to vectors
timer = time.time()
print('Converting tweets to vectors...')
token_2_idx = {vocab[i] : i for i in range(VOCAB_SIZE)}

def tweet_to_vector(tweet, show_unknowns=False):
    vector = np.zeros(VOCAB_SIZE, dtype=np.int_)
    for token in tokenizer.tokenize(tweet):
        stem = get_stem(token)
        idx = token_2_idx.get(stem, None)
        if idx is not None:
            vector[idx] = 1
        elif show_unknowns:
            print("Unknown token: {}".format(token))
    return vector

tweet_vectors = np.zeros(
    (len(negative_tweets) + len(positive_tweets), VOCAB_SIZE), 
    dtype=np.int_)
tweets = []
for ii, (_, tweet) in enumerate(negative_tweets.iterrows()):
    tweets.append(tweet[3])
    tweet_vectors[ii] = tweet_to_vector(tweet[3])
for ii, (_, tweet) in enumerate(positive_tweets.iterrows()):
    tweets.append(tweet[3])
    tweet_vectors[ii + len(negative_tweets)] = tweet_to_vector(tweet[3])
print('Tweets converting takes: {0}'.format(time.time()-timer))
#endregion

#region Load NN
def build_model(learning_rate=0.1):
    tf.reset_default_graph()
    
    net = tflearn.input_data([None, VOCAB_SIZE])
    net = tflearn.fully_connected(net, 125, activation='ReLU')
    net = tflearn.fully_connected(net, 25, activation='ReLU')
    net = tflearn.fully_connected(net, 2, activation='softmax')
    regression = tflearn.regression(
        net, 
        optimizer='sgd', 
        learning_rate=learning_rate, 
        loss='categorical_crossentropy')
    
    model = tflearn.DNN(net)
    return model
model = build_model(learning_rate=0.75)
model.load(SAVED_NN)
#endregion

#region Wild testing
print('Wild testing NN...')

def test_tweet(tweet):
    tweet_vector = tweet_to_vector(tweet, True)
    positive_prob = model.predict([tweet_vector])[0][1]
    print('Original tweet: {}'.format(tweet))
    print('P(positive) = {:.5f}. Result: '.format(positive_prob), 
          'Positive' if positive_prob > 0.5 else 'Negative')


tweets_for_testing = [
    "меня оштрафовали по дороге домой"
]
for tweet in tweets_for_testing:
    test_tweet(tweet) 
    print("---------")
#endregion

print('Total procedure takes: {0}'.format(time.time()-total_timer))