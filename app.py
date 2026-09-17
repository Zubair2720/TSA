import streamlit as st
import pickle
import re
import numpy as np
import tensorflow as tf


# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)


# =========================
# Load ML Model
# =========================

with open("sentiment_svm_pipeline.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)


# =========================
# Load DL Model
# =========================

rnn_model = tf.keras.models.load_model(
    "sentiment_rnn_model.keras"
)


# =========================
# Text Cleaning
# =========================

def clean_tweet(text):

    text = str(text).lower()

    text = re.sub(
        r'http\S+|www\.\S+',
        ' ',
        text
    )

    text = re.sub(
        r'@\w+',
        ' ',
        text
    )

    text = re.sub(
        r'#',
        '',
        text
    )

    text = re.sub(
        r'[^a-z\s]',
        ' ',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text


# =========================
# Title
# =========================

st.title("💬 Twitter Sentiment Analysis")

st.write(
    "Enter a tweet and select a Machine Learning "
    "or Deep Learning model."
)


# =========================
# Model Dropdown
# =========================

model_type = st.selectbox(
    "Select Model",
    [
        "Machine Learning",
        "Deep Learning"
    ]
)


# =========================
# Tweet Input
# =========================

tweet = st.text_area(
    "Enter your tweet:",
    placeholder="Example: I am very happy today!"
)


# =========================
# Prediction
# =========================

if st.button("Predict Sentiment"):

    if tweet.strip() == "":
        st.warning("Please enter a tweet.")

    else:

        cleaned_tweet = clean_tweet(tweet)


        # =====================
        # Machine Learning
        # =====================

        if model_type == "Machine Learning":

            prediction = svm_model.predict(
                [cleaned_tweet]
            )[0]

            sentiment = encoder.inverse_transform(
                [prediction]
            )[0]

            st.success(
                f"Predicted Sentiment: {sentiment}"
            )


        # =====================
        # Deep Learning
        # =====================

        else:

            prediction = rnn_model.predict(
                tf.constant([cleaned_tweet], dtype=tf.string),
                verbose=0
            )

            predicted_class = np.argmax(
                prediction,
                axis=1
            )[0]

            sentiment = encoder.inverse_transform(
                [predicted_class]
            )[0]

            confidence = np.max(
                prediction
            ) * 100

            st.success(
                f"Predicted Sentiment: {sentiment}"
            )

            st.info(
                f"Confidence: {confidence:.2f}%"
            )