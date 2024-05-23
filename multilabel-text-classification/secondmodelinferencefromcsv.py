import pandas as pd
import numpy as np
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

# Define the path to the model and pickle files
checkpoint_dir = 'path to your checkpoints'
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

# Path to the input CSV file and the output CSV file
input_csv_path = 'path_to_input_csv.csv'
output_csv_path = 'path_to_output_csv.csv'

# Load the input CSV file
df = pd.read_csv(input_csv_path)

# Ensure the input CSV has the correct column
if 'comments' not in df.columns:
    raise ValueError("The input CSV must contain a 'comments' column.")

# Preprocess the comments
df['comments'] = df['comments'].apply(normalize_text)

# Tokenize the comments
comments_seq = tokenizer.texts_to_sequences(df['comments'])

# Pad the sequences
max_len = 1020  # Ensure this matches the max_len used during training
comments_padded = pad_sequences(comments_seq, maxlen=max_len)

# Set up the device to use GPU if available, otherwise use CPU
device = tf.device("GPU" if tf.config.list_physical_devices('GPU') else "CPU")

with device:
    # Make predictions
    predictions = best_model.predict(comments_padded)

    # Get the indices of the highest probability for each prediction
    predicted_class_indices = np.argmax(predictions, axis=1)

    # Decode the indices to the original labels
    predicted_labels = label_encoder.inverse_transform(predicted_class_indices)

    # Add the predicted labels to the DataFrame
    if 'label' in df.columns:
        df['label'] = predicted_labels
    else:
        df['label'] = predicted_labels

    # Save the DataFrame with the predicted labels to a new CSV file
    df.to_csv(output_csv_path, index=False)

print(f"Predictions saved to {output_csv_path}")
