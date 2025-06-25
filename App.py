# app.py (Streamlit UI for Mental Health Detection)

import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Load saved model and vectorizer
with open(r"C:\Users\sunil\Desktop\NIT Intership proj\Project List\9. Mental_Health_Detection\mental_health_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(r"C:\Users\sunil\Desktop\NIT Intership proj\Project List\9. Mental_Health_Detection\tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Streamlit UI
st.title("🧠 Mental Health Text Classifier")
st.markdown("Enter a sentence or paragraph to detect possible mental health concerns like **depression**, **anxiety**, or **stress**.")

user_input = st.text_area("Your Text", height=150)

if st.button("Analyze"):
    if user_input:
        clean_input = preprocess(user_input)
        vector = vectorizer.transform([clean_input])
        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]

        st.subheader("🔍 Prediction Result")
        st.markdown(f"**Predicted Label:** `{prediction}`")

        st.subheader("📊 Class Probabilities")
        for cls, prob in zip(model.classes_, probabilities):
            st.write(f"{cls}: {prob:.2%}")
    else:
        st.warning("Please enter some text to analyze.")