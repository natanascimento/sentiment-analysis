import streamlit as st
from core.multipage import MultiPage
from pages import sentiment_analysis_with_vader, sentiment_analysis_with_lstm

app = MultiPage()

st.title("Sentiment Analysis - Machine Learning Class")

app.add_page("Sentiment Analysis with VADER", sentiment_analysis_with_vader.app)
app.add_page("Sentiment Analysis with LSTM", sentiment_analysis_with_lstm.app)

app.run()
