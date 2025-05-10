import streamlit as st
import re
import string
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    USE_STEMMING = True
except ImportError:
    stemmer = None
    USE_STEMMING = False

st.set_page_config(page_title="SMS/Email Spam Detection", layout="centered")
st.title("SMS/Email Spam Detection")
loading_placeholder = st.empty()
loading_placeholder.info("Loading resources... please wait.")

# Tokenizer
def tokenize_text(text):
    return re.findall(r'\b\w+\b', text.lower())

# Preprocessing
def preprocess_text(text):
    tokens = tokenize_text(text)
    processed = []
    for token in tokens:
        if token in ENGLISH_STOP_WORDS:
            continue
        if USE_STEMMING and stemmer:
            token = stemmer.stem(token)
        processed.append(token)
    return " ".join(processed)

try:
    voting_clf = joblib.load("Spam_detection_voting.pkl")
    tfidf_vectorizer = joblib.load("tfidf_spam_voting.pkl")
    models_loaded = True
    loading_placeholder.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    models_loaded = False

loading_placeholder.empty()

# Input
input_text = st.text_area("Enter the message/text", height=150)

# Metrics
accuracy = 0.9845
precision = 1.0
recall = 0.8889
f1_score_value = 0.9412
cm = np.array([[890, 0], [16, 128]])

with st.expander("Model Performance Metrics", expanded=True):
    st.subheader("Model Performance Metrics")
    st.write(f"**Accuracy**: {accuracy:.4f}")
    st.write(f"**Precision**: {precision:.4f}")
    st.write(f"**Recall**: {recall:.4f}")
    st.write(f"**F1-Score**: {f1_score_value:.4f}")
    st.write("**Confusion Matrix**:")
    st.write(pd.DataFrame(cm, columns=['Predicted Ham', 'Predicted Spam'], index=['Actual Ham', 'Actual Spam']))

if st.button('Predict'):
    if not models_loaded:
        st.error("Models failed to load. Cannot make predictions.")
    elif not input_text:
        st.warning("Please enter a message to predict.")
    else:
        try:
            with st.spinner("Processing..."):
                preprocessed_text = preprocess_text(input_text)
                tfidf_input = tfidf_vectorizer.transform([preprocessed_text])
                prediction = voting_clf.predict(tfidf_input)
                if prediction == 1:
                    st.error("This is a SPAM message!")
                else:
                    st.success("This is NOT a SPAM message!")
                proba = voting_clf.predict_proba(tfidf_input)[0][1]
                st.write(f"**Spam Probability:** `{proba:.2%}`")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Try refreshing the page or contact the developer for support.")

st.markdown("<br><br><center><i>Developed by Zeeshan Akram</i></center>", unsafe_allow_html=True)
