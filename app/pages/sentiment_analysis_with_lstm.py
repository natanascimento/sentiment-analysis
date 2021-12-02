import streamlit as st
from models import ModelClassifier


def app():
    st.markdown("## Sentiment Analysis with LSTM")
    input = st.text_input('Insert your idea')
    
    if st.button('Send Idea'):
        st.markdown(f"#### Sentimento {ModelClassifier().execute(input)}")