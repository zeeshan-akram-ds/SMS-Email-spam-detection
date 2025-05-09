import streamlit as st
import re
import string
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="SMS/Email Spam Detection", layout="centered")
st.title("SMS/Email Spam Detection")
loading_placeholder = st.empty()
loading_placeholder.info("Loading resources... please wait.")

# custom tokenizer function
def simple_tokenize(text):
    """Simple tokenizer that splits on whitespace and punctuation"""
    # Remove punctuation
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    # Split on whitespace and filter empty strings
    return [token for token in text.split() if token]

# Basic stopword list 
STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 
    't', 'can', 'will', 'just', 'don', 'should', 'now'
])

# Simple stemming function
def simple_stem(word):
    """Extremely simplified stemmer that just removes common endings"""
    if len(word) > 4:
        for ending in ['ing', 'ed', 'es', 's']:
            if word.endswith(ending):
                return word[:-len(ending)]
    return word


try:
    voting_clf = joblib.load("Spam_detection_voting.pkl")
    tfidf_vectorizer = joblib.load("tfidf_spam_voting.pkl")
    models_loaded = True
    loading_placeholder.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    models_loaded = False

def preprocess_text(text):
    # Lowercase 
    text = text.lower()
    # Remove URLs, emails, digits, and special characters
    text = re.sub(r"http\S+|www\S+|\S+@\S+|\d+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    
    tokens = simple_tokenize(text)
    
    # Remove stopwords and apply stemming
    filtered = [simple_stem(word) for word in tokens if word not in STOPWORDS]
    return " ".join(filtered)

loading_placeholder.empty()

st.title("SMS/Email Spam Detection")
input_text = st.text_area("Enter the message/text", height=150)

# Metrics 
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
    if not models_loaded:
        st.error("Models failed to load. Cannot make predictions.")
    elif not input_text:
        st.warning("Please enter a message to predict.")
    else:
        try:
            # Show processing status
            with st.spinner("Processing..."):
                preprocessed_text = preprocess_text(input_text)
                
                # Transform the text
                tfidf_input = tfidf_vectorizer.transform([preprocessed_text])
                prediction = voting_clf.predict(tfidf_input)
                
                if prediction == 1:
                    st.error("This is a SPAM message!")
                else:
                    st.success("This is NOT a SPAM message!")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Try refreshing the page or contact the developer for support.")

st.markdown("<br><br><center><i>Developed by Zeeshan Akram</i></center>", unsafe_allow_html=True)
