import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class SentimentAnalyze:

    def __init__(self):
        self.__sid = SentimentIntensityAnalyzer()

    @staticmethod
    def classify_predict(predict: dict) -> str:
        if predict['neg'] < 0.5 and predict['pos'] > 0.5:
            return "Positivo"
        elif predict['neg'] > 0.5 and predict['pos'] < 0.5:
            return "Negativo"
        else:
            return "Neutro"

    def predict(self, text: str) -> str:
        predict = self.__sid.polarity_scores(text)
        return self.classify_predict(predict)




def app():
    st.markdown("## Sentiment Analysis with VADER")
    user_input = st.text_input('Insert your review')
    
    if st.button('Send Review'):
        st.markdown(f"#### Sentimento {SentimentAnalyze().predict(user_input)}")