import streamlit as st
import os
import re
import string
import numpy as np
import pandas as pd
import joblib

# Display initial loading message
st.set_page_config(page_title="SMS/Email Spam Detection", layout="centered")
st.title("SMS/Email Spam Detection")
loading_placeholder = st.empty()
loading_placeholder.info("Loading NLTK resources... please wait.")

# Create custom tokenizer function that doesn't rely on punkt_tab
def simple_tokenize(text):
    """Simple tokenizer that splits on whitespace and punctuation"""
    # Remove punctuation
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    # Split on whitespace and filter empty strings
    return [token for token in text.split() if token]

# Configure NLTK with explicit error handling
import nltk

# Try multiple download locations
nltk_dirs = ['./nltk_data', '/app/nltk_data', os.path.expanduser('~/nltk_data')]
for nltk_dir in nltk_dirs:
    os.makedirs(nltk_dir, exist_ok=True)
    nltk.data.path.insert(0, nltk_dir)

# Download necessary resources with fallback
try:
    nltk.download('stopwords', download_dir=nltk_dirs[0])
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
    loading_placeholder.success("NLTK resources loaded successfully!")
except Exception as e:
    st.warning(f"Could not load NLTK stopwords: {str(e)}. Using a basic stopword list.")
    # Fallback to a basic stopword list
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

# Try to import stemmer or use a simple lambda as fallback
try:
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer().stem
except Exception as e:
    st.warning(f"Could not load NLTK PorterStemmer: {str(e)}. Using basic stemming.")
    # Very simple stemming function as fallback
    def stemmer(word):
        """Extremely simplified stemmer that just removes common endings"""
        if len(word) > 4:
            for ending in ['ing', 'ed', 'es', 's']:
                if word.endswith(ending):
                    return word[:-len(ending)]
        return word

# Load models with error handling
try:
    voting_clf = joblib.load("Spam_detection_voting.pkl")
    tfidf_vectorizer = joblib.load("tfidf_spam_voting.pkl")
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    models_loaded = False

# Preprocessing function with fallback tokenizer
def preprocess_text(text):
    # Lowercase 
    text = text.lower()
    # Remove URLs, emails, digits, and special characters
    text = re.sub(r"http\S+|www\S+|\S+@\S+|\d+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    
    # Tokenize with fallback method
    try:
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text)
    except Exception:
        tokens = simple_tokenize(text)
    
    # Remove stopwords and apply stemming
    filtered = [stemmer(word) for word in tokens if word not in STOPWORDS]
    return " ".join(filtered)

# Clear loading message
loading_placeholder.empty()

# UI Elements
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
    if not models_loaded:
        st.error("Models failed to load. Cannot make predictions.")
    elif not input_text:
        st.warning("Please enter a message to predict.")
    else:
        try:
            # Show processing status
            with st.spinner("Processing..."):
                # Preprocess the input
                preprocessed_text = preprocess_text(input_text)
                
                # Transform the text
                tfidf_input = tfidf_vectorizer.transform([preprocessed_text])
                prediction = voting_clf.predict(tfidf_input)
                
                # Display result with appropriate styling
                if prediction == 1:
                    st.error("🚨 This is a SPAM message! 🚨")
                else:
                    st.success("✅ This is NOT a SPAM message!")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Try refreshing the page or contact the developer for support.")

st.markdown("<br><br><center><i>Developed by Zeeshan Akram</i></center>", unsafe_allow_html=True)
