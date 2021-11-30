#!/usr/bin/env python
# coding: utf-8

# ## Sentiment Analysis on US Airline Reviews

import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding


class ModelClassifier:

    def __init__(self):
        self.__GITHUB_BASEURL = "https://raw.githubusercontent.com/"
        self.__DATA_PATH = "natanascimento/sentiment-analysis/main/app/data/tweets-airline-sentiment.csv"
        self.__embedding_vector_length = 32

    @staticmethod
    def __generate_github_uri(base_url: str, data_path: str):
        return f"{base_url}{data_path}"

    @staticmethod
    def save_model(history):
        #Save entire model to a HDF5 file
        return history.save('app/models/store/model.h5')

    def create_dataframe(self):
        df = pd.read_csv(self.__generate_github_uri(base_url=self.__GITHUB_BASEURL,
                                                    data_path=self.__DATA_PATH))
        return df

    def clean_dataframe(self, df: pd.DataFrame):
        tweet_df = df[['text','airline_sentiment']]        
        tweet_df = tweet_df[tweet_df['airline_sentiment'] != 'neutral']
        sentiment_label = tweet_df.airline_sentiment.factorize()

        return tweet_df, sentiment_label
    
    @staticmethod
    def train(vocab_size, vector_length, padded_sequence, sentiment_label):
        model = Sequential() 
        model.add(Embedding(vocab_size, vector_length, input_length=200) )
        model.add(SpatialDropout1D(0.25))
        model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid')) 
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(padded_sequence,sentiment_label[0],validation_split=0.2, epochs=5, batch_size=32)

        return model

    def execute(self):
        df = self.create_dataframe()
        tweet_df, sentiment_label = self.clean_dataframe(df=df)

        tweet = tweet_df.text.values
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(tweet)
        vocab_size = len(tokenizer.word_index) + 1
        encoded_docs = tokenizer.texts_to_sequences(tweet)
        padded_sequence = pad_sequences(encoded_docs, maxlen=200)

        history = self.train(vocab_size, self.__embedding_vector_length, padded_sequence, sentiment_label)

        self.save_model(history=history)