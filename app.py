import nltk
import streamlit as st
import os
import re
import string
import numpy as np
import pandas as pd
os.makedirs('nltk_data', exist_ok=True)

# Configure NLTK to use the custom data directory
nltk.data.path.insert(0, './nltk_data')

# Download necessary NLTK resources with explicit error handling
try:
    # Try to download punkt and stopwords
    nltk.download('punkt', download_dir='./nltk_data')
    nltk.download('stopwords', download_dir='./nltk_data')
    
    # After downloading, explicitly try to load them to verify
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    # Test if they load correctly
    stopwords.words('english')[:1]  # Just testing if it works
    word_tokenize("Test sentence")  # Test tokenization
    
except Exception as e:
    st.error(f"NLTK Resource Error: {str(e)}")
    st.info("Attempting alternative download approach...")
    # Try alternative approach with default download directory
    nltk.download('punkt')
    nltk.download('stopwords')

import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



# Load
voting_clf = joblib.load("Spam_detection_voting.pkl")
tfidf_vectorizer = joblib.load("tfidf_spam_voting.pkl")

# Preprocessing function
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess_text(text):
    # Lowercase 
    text = text.lower()
    # Remove URLs,emails,digits, and special characters
    text = re.sub(r"http\S+|www\S+|\S+@\S+|\d+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    # Tokenize and remove stopwords,apply stemming
    tokens = word_tokenize(text)
    filtered = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(filtered)

st.set_page_config(page_title="SMS/Email Spam Detection", layout="centered")

st.title("SMS/Email Spam Detection")

input_text = st.text_area("Enter the message/text", height=150)

# Metrics of Model
accuracy = 0.9845261121856866
precision = 1.0
recall = 0.8888888888888888
f1_score_value = 0.9411764705882353
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
    if input_text:
        # Preprocess the input
        preprocessed_text = preprocess_text(input_text)
        
        # Transform the text
        tfidf_input = tfidf_vectorizer.transform([preprocessed_text])

        prediction = voting_clf.predict(tfidf_input)
        if prediction == 1:
            st.header("This is a SPAM message!")
        else:
            st.header("This is NOT a SPAM message!")
    else:
        st.warning("Please enter a message to predict.")
st.markdown("<br><br><center><i>Developed by Zeeshan Akram</i></center>", unsafe_allow_html=True)
