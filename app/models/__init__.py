#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
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
        return history.save('app/models/store/model-v1.h5')

    @staticmethod
    def load():
        return load_model('app/models/store/model-v1.h5')
        
    def create_dataframe(self):
        df = pd.read_csv(self.__generate_github_uri(base_url=self.__GITHUB_BASEURL,
                                                    data_path=self.__DATA_PATH))
        return df

    @staticmethod
    def clean_dataframe(df: pd.DataFrame):
        tweet_df = df[['text','airline_sentiment']]
        tweet_df = tweet_df[tweet_df['airline_sentiment'] != 'neutral']
        sentiment_label = tweet_df.airline_sentiment.factorize()

        return tweet_df, sentiment_label
    
    @staticmethod
    def __classify_sentiment(sentiment: str) -> str:
        if sentiment == "negative":
            return "Negativo"
        elif sentiment == "positive":
            return "Positivo"
        else:
            return "Neutro"

    def predict_sentiment(self, model, text):
        tw = self.tokenizer.texts_to_sequences([text])
        tw = pad_sequences(tw,maxlen=200)
        prediction = int(model.predict(tw).round().item())
        sentiment = self.sentiment_label[1][prediction]
        
        return self.__classify_sentiment(sentiment)

    def execute(self, text):
        df = self.create_dataframe()
        self.tweet_df, self.sentiment_label = self.clean_dataframe(df=df)

        tweet = self.tweet_df.text.values
        self.tokenizer = Tokenizer(num_words=2500)
        self.tokenizer.fit_on_texts(tweet)
        vocab_size = len(self.tokenizer.word_index) + 1
        encoded_docs = self.tokenizer.texts_to_sequences(tweet)
        padded_sequence = pad_sequences(encoded_docs, maxlen=200)

        model = Sequential() 
        model.add(Embedding(vocab_size, self.__embedding_vector_length, input_length=200) )
        model.add(SpatialDropout1D(0.25))
        model.add(LSTM(75, dropout=0.4, recurrent_dropout=0.4))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid')) 
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = model.fit(padded_sequence, self.sentiment_label[0], validation_split=0.5, epochs=1, batch_size=500)
        
        label = self.predict_sentiment(model, text)

        return label
