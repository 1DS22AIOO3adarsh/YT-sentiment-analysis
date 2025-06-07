import streamlit as st
import requests

st.title("Sentiment Analysis")

text_input = st.text_area("Enter text to analyze:")

if st.button("Analyze"):
    if text_input:
        response = requests.post(
            "http://127.0.0.1:8000/predict/",
            json={"text": text_input}
        )

        if response.status_code == 200:
            result = response.json()
            st.subheader("Prediction:")
            st.success(f"Sentiment: {result['sentiment']}")
            st.info(f"Confidence: {result['confidence']}")
        else:
            st.error("API Error: Could not get prediction.")
    else:
        st.warning("Please enter some text first.")
