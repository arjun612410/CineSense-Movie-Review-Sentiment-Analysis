import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model('simple_rnn.h5')

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=100)
    return padded_review

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)

    prediction = model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return sentiment, prediction[0][0]


import streamlit as st

st.set_page_config(
    page_title="CineSense | Movie Review Sentiment",
    page_icon="🎬",
    layout="centered",
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top, #1a1a2e 0%, #0d0d17 60%, #050508 100%);
        color: #f1f1f1;
    }

    /* Hide default streamlit chrome */
    #MainMenu, header, footer {visibility: hidden;}

    .cine-header {
        text-align: center;
        padding: 10px 0 4px 0;
    }

    .cine-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 56px;
        letter-spacing: 4px;
        background: linear-gradient(90deg, #ffcc70, #e6482e, #b02e8f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .cine-subtitle {
        color: #b8b8c8;
        font-size: 16px;
        font-weight: 300;
        margin-top: -6px;
        letter-spacing: 1px;
    }

    .film-strip {
        height: 14px;
        width: 100%;
        background-image: repeating-linear-gradient(90deg, #2b2b3d 0px, #2b2b3d 18px, #ffcc70 18px, #ffcc70 22px);
        border-radius: 6px;
        margin: 18px 0 28px 0;
        opacity: 0.85;
    }

    .review-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 26px 26px 10px 26px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.45);
        backdrop-filter: blur(6px);
        margin-bottom: 22px;
    }

    .review-card label {
        color: #ffcc70 !important;
        font-weight: 600 !important;
        font-size: 15px !important;
    }

    .stTextArea textarea {
        background-color: #14141f !important;
        color: #f1f1f1 !important;
        border: 1px solid rgba(255,204,112,0.3) !important;
        border-radius: 12px !important;
        font-size: 15px !important;
    }

    .stTextArea textarea:focus {
        border: 1px solid #e6482e !important;
        box-shadow: 0 0 0 2px rgba(230,72,46,0.25) !important;
    }

    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #e6482e, #b02e8f);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 12px 0;
        font-size: 17px;
        font-weight: 600;
        letter-spacing: 1px;
        margin-top: 10px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 6px 18px rgba(230,72,46,0.35);
    }

    div.stButton > button:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 10px 24px rgba(230,72,46,0.5);
        color: white;
        border: none;
    }

    .result-box {
        border-radius: 18px;
        padding: 24px;
        margin-top: 18px;
        text-align: center;
        animation: fadeIn 0.5s ease-in;
    }

    .positive-box {
        background: linear-gradient(135deg, rgba(46,204,113,0.15), rgba(46,204,113,0.05));
        border: 1px solid rgba(46,204,113,0.4);
    }

    .negative-box {
        background: linear-gradient(135deg, rgba(230,72,46,0.15), rgba(230,72,46,0.05));
        border: 1px solid rgba(230,72,46,0.4);
    }

    .sentiment-emoji {
        font-size: 48px;
        margin-bottom: 4px;
    }

    .sentiment-label {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 32px;
        letter-spacing: 3px;
        margin: 6px 0 2px 0;
    }

    .score-text {
        color: #cfcfcf;
        font-size: 14px;
        letter-spacing: 0.5px;
    }

    .footer-note {
        text-align: center;
        color: #6c6c80;
        font-size: 12px;
        margin-top: 30px;
        letter-spacing: 0.5px;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="cine-header">
        <div class="cine-title">🎬 CINESENSE</div>
        <div class="cine-subtitle">AI-Powered Movie Review Sentiment Analysis</div>
    </div>
    <div class="film-strip"></div>
""", unsafe_allow_html=True)

st.markdown('<div class="review-card">', unsafe_allow_html=True)
st.markdown("🍿 **Drop your movie review below — let's see what the critic AI thinks:**")
user_input = st.text_area(
    "Movie Review",
    placeholder="e.g. The cinematography was stunning and the plot kept me hooked till the last scene...",
    height=150,
    label_visibility="collapsed",
)
classify_clicked = st.button("🎞️  ANALYZE REVIEW")
st.markdown('</div>', unsafe_allow_html=True)

if classify_clicked:
    if user_input.strip() == "":
        st.warning("🎥 Please enter a movie review first!")
    else:
        preprocessed_input = preprocess_text(user_input)

        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        score = prediction[0][0]

        if sentiment == 'Positive':
            st.markdown(f"""
                <div class="result-box positive-box">
                    <div class="sentiment-emoji">🍿✨</div>
                    <div class="sentiment-label" style="color:#2ecc71;">POSITIVE REVIEW</div>
                    <div class="score-text">Confidence Score: {score:.4f}</div>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
                <div class="result-box negative-box">
                    <div class="sentiment-emoji">🎬💔</div>
                    <div class="sentiment-label" style="color:#e6482e;">NEGATIVE REVIEW</div>
                    <div class="score-text">Confidence Score: {score:.4f}</div>
                </div>
            """, unsafe_allow_html=True)

        st.progress(float(score))
else:
    st.markdown(
        '<p style="text-align:center; color:#8a8a9c;">🎟️ Waiting for a review to analyze...</p>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="footer-note">Powered by a Simple RNN trained on the IMDB dataset 🎞️</div>', unsafe_allow_html=True)