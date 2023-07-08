import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model Naive Bayes
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load TF-IDF vectorizer
with open('tf-idf.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Function to extract features from text
def extract_features(text):
    text_tfidf = vectorizer.transform([text])
    return text_tfidf

# Title of the application
st.title("MSIB Sentiment Analysis")

# Input text
text_input = st.text_area("Masukkan komentar:")

# Prediction button
if st.button("Prediksi"):
    # Check if comment data has been entered
    if text_input:
        # Extract features from input text
        text_features = extract_features(text_input)

        # Perform prediction using Naive Bayes model
        prediction = model.predict(text_features)[0]

        # Set sentiment label based on prediction
        if prediction == 0:
            sentiment = "Positif"
        elif prediction == 1:
            sentiment = "Negatif"
        else:
            sentiment = "Netral"

        # Display prediction result
        st.subheader("Hasil Prediksi")
        st.write(f"Sentimen: {sentiment}")
    else:
        # If no comment data is entered
        st.warning("Mohon masukkan data komentar terlebih dahulu!")

