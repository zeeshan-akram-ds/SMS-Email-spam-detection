import streamlit as st
import re
import string
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

# Load
bernouli_clf = joblib.load("Spam_detection_bernouli.pkl")
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
st.sidebar.title("Explore Visualizations")
viz_option = st.sidebar.radio(
    "Select a visualization category:",
    ("None", 
     "WordClouds", 
     "Most Frequent Words", 
     "Top Bigrams", 
     "Threshold vs Metrics", 
     "ROC and PR Curves", 
     "Top 20 Spam Indicator Words",
     "Confusion Matrix")
)
st.title("SMS/Email Spam Detection")

input_text = st.text_area("Enter the message/text", height=150)

# Metrics of Model
accuracy = 0.9845261121856866
precision = 0.9848484848484849
recall = 0.9027777777777778
f1_score_value = 0.9420289855072463
cm = [[888, 2],
          [14, 130]]
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
        proba = bernouli_clf.predict_proba(tfidf_input)
        threshold = 0.45
        prediction = (proba[:, 1] >= threshold).astype(int)
        if prediction[0] == 1:
            st.header("This is a SPAM message!")
        else:
            st.header("This is NOT a SPAM message!")
        with st.spinner("Explaining with LIME..."):
            try:
                class_names = ['ham', 'spam']
                explainer = LimeTextExplainer(class_names=class_names)

                def predict_proba(texts):
                    preprocessed = [preprocess_text(t) for t in texts]
                    tfidf_vectors = tfidf_vectorizer.transform(preprocessed)
                    return bernouli_clf.predict_proba(tfidf_vectors)

                explanation = explainer.explain_instance(input_text, predict_proba, num_features=10)

                with st.expander("LIME Explanation", expanded=True):
                    st.markdown("### Top 10 Word Contributions")
                    components.html(explanation.as_html(), height=500, scrolling=True)

            except Exception as e:
                st.error(f"LIME explanation failed: {e}")
    else:
        st.warning("Please enter a message to predict.")

if viz_option != "None":
    st.markdown("---")
    st.subheader("Visualization")

    if viz_option == "WordClouds":
        st.image("plots/wordcloud_spam.png", caption="Spam WordCloud", use_container_width=True)
        st.image("plots/wordcloud_ham.png", caption="Ham WordCloud", use_container_width=True)

    elif viz_option == "Most Frequent Words":
        st.image("plots/spam_ham_top_words.png", caption="Most Frequent Words in Spam and Ham", use_container_width=True)

    elif viz_option == "Top Bigrams":
        st.image("plots/spam_ham_top_bigrams.png", caption="Top 10 Bigrams in Spam and Ham", use_container_width=True)

    elif viz_option == "Threshold vs Metrics":
        st.image("plots/thres_vs_metrics_spam_detection.png", caption="Precision, Recall, F1 vs Threshold", use_container_width=True)

    elif viz_option == "ROC and PR Curves":
        st.image("plots/roc_curve_spam_detection.png", caption="ROC Curve", use_container_width=True)
        st.image("plots/pr_curve_spam_detection.png", caption="Precision-Recall Curve", use_container_width=True)

    elif viz_option == "Top 20 Spam Indicator Words":
        st.image("plots/top_20_spam_indicators.png", caption="Top 20 Spam Indicator Words from BernoulliNB", use_container_width=True)

    elif viz_option == "Confusion Matrix":
        st.image("plots/confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
st.markdown("<br><br><center><i>Developed by Zeeshan Akram</i></center>", unsafe_allow_html=True)
