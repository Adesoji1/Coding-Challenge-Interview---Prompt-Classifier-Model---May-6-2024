import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import os
import string
from nltk.corpus import stopwords
import nltk

# Ensure necessary NLTK data packages are downloaded
nltk.download('stopwords')

# Define the path to the model and pickle files
checkpoint_dir = 'path to checkpoint'
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

# Example new comment
new_comment = "This is for new comment to classify in pharmacology department but what hope is for vet."

# Preprocess the new comment
new_comment_normalized = normalize_text(new_comment)

# Tokenize the new comment
new_comment_seq = tokenizer.texts_to_sequences([new_comment_normalized])

# Pad the sequence
max_len = 1020  # Ensure this matches the max_len used during training
new_comment_padded = pad_sequences(new_comment_seq, maxlen=max_len)

# Set up the device to use GPU if available, otherwise use CPU
device = tf.device("GPU" if tf.config.list_physical_devices('GPU') else "CPU")

with device:
    # Make the prediction
    prediction = best_model.predict(new_comment_padded)

    # Get the index of the highest probability
    predicted_class_index = np.argmax(prediction, axis=1)

    # Decode the index to the original label
    predicted_label = label_encoder.inverse_transform(predicted_class_index)
    print(f"Predicted Label: {predicted_label[0]}")
