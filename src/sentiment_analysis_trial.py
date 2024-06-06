import streamlit as st
from transformers import pipeline

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

def get_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result['label'], result['score']

st.title("Sentiment Analysis App")

st.write("This app uses a Transformer model to analyze the sentiment of the given text.")

user_input = st.text_area("Enter text for sentiment analysis:")

if st.button("Analyze"):
    if user_input:
        label, score = get_sentiment(user_input)
        st.write(f"Sentiment: {label}")
        st.write(f"Confidence: {score:.2f}")
    else:
        st.write("Please enter some text for analysis.")