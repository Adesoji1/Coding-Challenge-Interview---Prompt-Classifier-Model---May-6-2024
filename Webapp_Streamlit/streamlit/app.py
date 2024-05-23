
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import string
from nltk.corpus import stopwords
import nltk
import os

# Ensure necessary NLTK data packages are downloaded
nltk.download('stopwords')

# Load the saved model and tokenizer
checkpoint_dir = 'checkpoints'
model_path = os.path.join(checkpoint_dir, "best_modellatest2.keras")
label_encoder_path = 'label_encoder1.pkl'
tokenizer_path = 'tokenizer1.pkl'

# Load the saved model
best_model = load_model(model_path)

# Load the LabelEncoder and Tokenizer from disk
with open(label_encoder_path, 'rb') as le_file:
    label_encoder = pickle.load(le_file)

with open(tokenizer_path, 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Define stop words, excluding some specific ones if necessary
stop_words = set(stopwords.words('english')) - {'not', 'and', 'for'}

# Function to normalize text for pre-processing
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>+', ' ', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Prediction function
def predict_label(comment):
    comment_normalized = normalize_text(comment)
    comment_seq = tokenizer.texts_to_sequences([comment_normalized])
    comment_padded = pad_sequences(comment_seq, maxlen=1020)
    prediction = best_model.predict(comment_padded)
    predicted_class_index = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class_index)
    return predicted_label[0]

# Streamlit app
def main():
    st.set_page_config(page_title="Text Classification App", layout="wide")

    # Toggle between light and dark mode
    dark_mode = st.sidebar.checkbox("Dark Mode")

    # Apply dark mode if checkbox is checked
    if dark_mode:
        st.markdown(
            """
            <style>
                body {
                    color: white;
                    background-color: #1E1E1E;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
                body {
                    color: black;
                    background-color: white;
                }
            </style>
            """,
            unsafe_allow_html=True
        )

    st.title("Text Classification App 📜🤖")

    menu = ["Home 🏠", "Predict 🔮"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home 🏠":
        st.subheader("Home 🏠")
        st.write("Welcome to the Text Classification App! 🥳")
        st.write("Use this app to classify comments into different labels. 📚")

    elif choice == "Predict 🔮":
        st.subheader("Predict 🔮")
        comment = st.text_area("Enter Comment ✍️", "")

        # Add background picture for the section
        st.markdown(
            """
            <style>
                .predict-section {
                    background-image: url('https://drive.google.com/uc?export=view&id=124-bqaS4yNevaFgDa9F8VEx7-hDg3UNH');
                    background-size: cover;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.2);
                }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Add a CSS class to the section
        st.markdown('<div class="predict-section">', unsafe_allow_html=True)

        if st.button("Predict 🚀"):
            if comment:
                label = predict_label(comment)
                st.success(f"Predicted Label: {label} 🏆")
            else:
                st.warning("Please enter a comment to classify. ⚠️")

        # Close the div
        st.markdown('</div>', unsafe_allow_html=True)

    st.sidebar.subheader("About 💡")
    st.sidebar.text("Text Classification App 📜🤖")
    st.sidebar.text("Built with Streamlit ❤️")

if __name__ == "__main__":
    main()
