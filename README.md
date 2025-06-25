# Mental_Health_Detection-NLP-ML

#  Mental Health Detection using NLP & Machine Learning

This project implements a **mental health text classifier** that analyzes user-input text to predict potential mental health concerns such as **depression**, **anxiety**, or **stress**.

 Built using **Python, Streamlit, scikit-learn, NLTK**
 Supports **real-time text input and analysis**
 Uses **TF-IDF vectorization** + **ML model (e.g., Logistic Regression / SVM)**

##  Features

*  User-friendly **Streamlit web app**
*  Text preprocessing: lowercase, punctuation removal, stopword removal, lemmatization
*  Vectorization: **TF-IDF**
*  Machine Learning classification: detects depression, anxiety, stress, or neutral
*  Displays class probabilities for transparency

---

##  Tech Stack

* **Python 3**
* **Streamlit**
* **scikit-learn**
* **NLTK (stopwords, WordNet lemmatizer)**
* **Pickle (for model & vectorizer persistence)**

---

##  How to Run

### Clone the repository

```bash
git clone https://github.com/Darshanikant/Mental_Health_Detection-NLP-ML-.git
cd Mental_Health_Detection-NLP-ML-
```

### Install dependencies

```bash
pip install -r requirements.txt
```

(*Example libraries: `streamlit`, `scikit-learn`, `nltk`, `pandas`, etc.*)

### Download NLTK data (if not already done)

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### Run the app

```bash
streamlit run App.py
```

---

##  Project Structure

```
mental-health-detection/
├── App.py                # Streamlit frontend
├── Mental_Health_Detection.ipynb  # Model development notebook
├── mental_health_model.pkl        # Trained ML model
├── tfidf_vectorizer.pkl           # Saved vectorizer
└── README.md               # Project documentation
```

---

##  How it Works

 User enters text in Streamlit app
 Text is preprocessed → tokenized, stopwords removed, lemmatized
 Preprocessed text is vectorized using TF-IDF
 Trained ML model predicts the mental health category
 App displays prediction and class probabilities

---

##  Example

> **Input:**
> "I feel overwhelmed and anxious all the time."

> **Predicted Label:** `Anxiety`
> **Class Probabilities:**
>
> * Anxiety: 85%
> * Depression: 10%
> * Stress: 5%

---

##  Applications

* Early detection of mental health concerns from text
* Supporting mental health chatbots
* Social media sentiment & mental health monitoring

