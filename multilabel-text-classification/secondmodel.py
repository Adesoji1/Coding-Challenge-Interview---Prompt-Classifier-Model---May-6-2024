import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, Conv1D, Bidirectional, LSTM, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from numpy import asarray, zeros
import tensorflow as tf
import os
import re
import string
import nltk
from nltk.corpus import stopwords
import pickle

# Ensure necessary NLTK data packages are downloaded
nltk.download('stopwords')

# Verify that TensorFlow is using the GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load your dataset
# df_data = pd.read_csv("path_to_your_dataset.csv")  # Update with your dataset path
df_data = pd.read_csv('first_1000000_comments.csv')
X = df_data['comments']  # Column containing text data
y = df_data['label']  # Column containing labels

# Visualize label distribution
val_counts = y.value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=val_counts.index, y=val_counts.values, alpha=0.8)
plt.title("Labels per Classes")
plt.xlabel("Label Types")
plt.ylabel("Counts of Labels")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

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

# Normalize the comments
X = X.apply(normalize_text)

# Encode labels to integers for computing class weights and later for one-hot encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Compute class weights
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weights_dict = dict(enumerate(class_weights))

# One-hot encode the labels for use in model training
onehot_encoder = OneHotEncoder(sparse_output=False)
y_onehot = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.15, random_state=42)

# Tokenization and Padding
max_len = 1020
max_features = 10000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_padded = pad_sequences(X_train_seq, maxlen=max_len)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_len)

# Save the LabelEncoder and Tokenizer to disk
with open('label_encoder1.pkl', 'wb') as le_file:
    pickle.dump(label_encoder, le_file)

with open('tokenizer1.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

# Load GloVe embeddings
embeddings_dictionary = dict()
with open("/home/adesoji/Downloads/archive/glove.6B.300d.txt", encoding="utf8") as glove_file:
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions

# Create the embedding matrix
vocab_size = len(tokenizer.word_index) + 1
embed_size = 300  # Ensure this matches the dimension of your GloVe vectors
embedding_matrix = zeros((min(max_features, vocab_size), embed_size))
for word, index in tokenizer.word_index.items():
    if index >= max_features:
        continue
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# Define directories for saving model and logs
checkpoint_dir = "checkpoints"
tensorboard_log_dir = "logs"

# Create directories if they don't exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(tensorboard_log_dir, exist_ok=True)

# Define the ModelCheckpoint callback to save the best model
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "best_modellatest2.keras"),
    monitor="val_accuracy",  # Monitor validation accuracy
    save_best_only=True,     # Save only the best model
    verbose=1
)

# Define the TensorBoard callback
tensorboard_callback = TensorBoard(
    log_dir=tensorboard_log_dir,
    histogram_freq=1,        # Frequency (in epochs) at which to compute activation and weight histograms for the layers of the model
    write_graph=True,        # Whether to visualize the graph in TensorBoard
    write_images=True        # Whether to write model weights to visualize as image in TensorBoard
)

# Define the model
sequence_input = Input(shape=(max_len,))
x = Embedding(min(max_features, vocab_size), embed_size, weights=[embedding_matrix], trainable=False)(sequence_input)
x = SpatialDropout1D(0.2)(x)
x = Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x)
x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
avg_pool = GlobalAveragePooling1D()(x)
x = Dense(128, activation='relu')(avg_pool)
x = Dropout(0.1)(x)
preds = Dense(y_train.shape[1], activation="softmax")(x)  # Set number of units to number of labels

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=1e-3), metrics=['accuracy'])
print(model.summary())

# Training with ModelCheckpoint and TensorBoard callbacks
history = model.fit(
    X_train_padded, 
    y_train, 
    batch_size=8, 
    epochs=10, 
    validation_split=0.1, 
    class_weight=class_weights_dict, 
    verbose=1, 
    callbacks=[checkpoint_callback, tensorboard_callback]
)

# Evaluation
loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=1)
print("Test Accuracy:", accuracy)

# Plot training & validation accuracy values
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.tight_layout()
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.tight_layout()
plt.show()
