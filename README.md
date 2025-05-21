# Spam Detection System using Bernouli Classifier

A robust SMS/Email Spam Detection project that uses advanced NLP techniques, multiple classification models, and an optimized ensemble voting strategy to accurately distinguish spam from ham (not spam). This project also includes an interactive Streamlit web app for real-time predictions.

---

## Project Overview

This project aims to build a high-precision spam detection system using ensemble learning with multiple base classifiers. The main goal is to maximize precision, ensuring that legitimate (ham) messages are never incorrectly flagged as spam — a critical requirement in real-world applications like email filtering systems.

---

## Key Highlights

- Comprehensive EDA  
- Advanced Text Preprocessing  
- Feature Engineering (text length & word count)  
- Visual Explorations (Histograms, WordClouds, Bigrams, Scatter Plots)  
- TF-IDF Vectorization  
- Baseline and Advanced Models (Naive Bayes, XGBoost, Random Forest)  
- Hyperparameter Tuning with GridSearchCV & Optuna  
- Final Model: Optimized Voting Classifier  
- Streamlit App for Real-Time Spam Detection  

---

## Project Workflow

### 1. Exploratory Data Analysis
- Removal of duplicates and irrelevant records
- Class distribution visualization
- Message length and word count distributions
- Correlation via scatter plot between text length and word count

### 2. Text Preprocessing
- Lowercasing
- Removal of URLs, emails, digits, and punctuations
- Tokenization
- Stopword removal
- Stemming using `PorterStemmer`

### 3. Feature Engineering
- `text_length`: Total number of characters in the message
- `word_count`: Total number of words per message

### 4. Visualizations
- Spam vs. Ham distribution
- WordClouds of frequent terms (side-by-side comparison)
- Bar plots of top 10 most common words
- Top 10 bigrams in spam and ham messages
- Scatter plots to analyze feature relationships

### 5. Model Building
- TF-IDF used to vectorize preprocessed text
- Trained multiple models:
  - Random Forest Classifier
  - XGBoost
  - Gaussian, Multinomial, and Bernoulli Naive Bayes
- Best base models: MultinomialNB and BernoulliNB

### 6. Model Evaluation
- Used K-Fold Cross-Validation
- Tuned models using:
  - GridSearchCV for BernoulliNB
  - Optuna for BernoulliNB, MultinomialNB, and RFC
- Plotted Precision-Recall curve for performance insights

### 7. Final Model: Voting Classifier
- Combined BernoulliNB (tuned), MultinomialNB (tuned), and RFC (default)
- Tested different voting strategies (soft/hard) and weight combinations
- Final selection based on achieving maximum precision

---

## Final Model Performance

| Metric     | Value     |
|------------|-----------|
| Accuracy   | 0.9845    |
| Precision  | 1.0000    |
| Recall     | 0.8889    |
| F1-Score   | 0.9412    |

**Confusion Matrix:**

|                | Predicted Ham | Predicted Spam |
|----------------|---------------|----------------|
| **Actual Ham** | 890           | 0              |
| **Actual Spam**| 16            | 128            |

This model was optimized specifically to never flag a ham message as spam — achieving 100% precision. This trade-off slightly affects recall but ensures important messages are never lost.

---

## Streamlit App

A simple yet effective UI built using Streamlit. The app:
- Takes user input via a text box
- Preprocesses text using the same training logic
- Applies the trained TF-IDF vectorizer
- Predicts using the final Voting Classifier
- Displays prediction: Spam or Not Spam
- Also shows model performance metrics in an expandable section

---

## Tech Stack

- Python, Pandas, NumPy
- NLTK for NLP preprocessing
- Scikit-learn for ML models and TF-IDF
- Optuna & GridSearchCV for tuning
- Streamlit for UI
- Matplotlib / Seaborn for visualizations

---

## Developed By

**Zeeshan Akram**  
Data Science Enthusiast | Machine Learning Explorer

---

**Project Structure:**


┣━━ app.py
┣━━ spam classifier.ipynb
┣━━ tfidf_spam_voting.pkl
┣━━ Spam_detection_voting.pkl
┗━━ README.md

