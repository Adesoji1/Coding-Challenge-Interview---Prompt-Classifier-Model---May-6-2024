# Coding-Challenge-Interview---Prompt-Classifier-Model---May-6-2024
 Create a classifier that will accurately classify a list of reddit comments into the proper labels.
 # Multilabel Text Classification Using CNN and Bi-LSTM 🎉📚🤖

![Text Classification](https://img.icons8.com/ios/452/language.png)

This project demonstrates how to build a multilabel text classification model using a combination of Convolutional Neural Networks (CNN) and Bidirectional Long Short-Term Memory Networks (Bi-LSTM). The model is trained using the TensorFlow and Keras libraries. We also use the GloVe word embeddings to enhance the model's performance and class weights to handle class imbalance.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Streamlit Web App](#streamlit-web-app)
- [Why We Didn't Use GPT-2 or LLM](#why-we-didnt-use-gpt-2-or-llm)
- [Requirements](#requirements)
- [References](#references)

## Overview 💡

Multilabel text classification involves assigning multiple labels to a given piece of text. This project utilizes a combination of CNN and Bi-LSTM networks to capture both local and sequential patterns in text data. 

### Why GloVe Embeddings? 🧠
GloVe (Global Vectors for Word Representation) embeddings are used because they capture semantic relationships between words, providing richer contextual information compared to simple one-hot encoding or TF-IDF vectors.

### Why Class Weights? ⚖️
Class weights are employed to handle class imbalance, ensuring that the model does not become biased towards the majority class. This is critical in real-world scenarios where certain classes may be underrepresented.

## Installation 💻

To set up the project, clone the repository and install the required dependencies:

```sh
git clone https://github.com/Adesoji1/Coding-Challenge-Interview---Prompt-Classifier-Model---May-6-2024.git
cd multilabel-text-classification
pip install -r requirements.txt
```

## Data Preprocessing 🧪

1. **Normalization**: Text normalization is performed to clean and standardize the input text. This includes converting to lowercase, removing punctuation, and stop words.

2. **Encoding**: Labels are encoded using `LabelEncoder` and `OneHotEncoder` to facilitate training.

3. **Tokenization**: Text data is tokenized and padded to ensure uniform input length for the neural network.

### Why is this important? 🤔
- **Normalization** ensures that the text data is clean and consistent.
- **Encoding** is crucial for converting categorical labels into a numerical format that the model can understand.
- **Tokenization** transforms the text into sequences of numbers, making it suitable for input to the neural network.

```python
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
```

## Model Architecture 🏗️

The model architecture includes:
- **Embedding Layer**: Uses pre-trained GloVe embeddings to convert words into dense vectors.
- **SpatialDropout1D Layer**: Prevents overfitting by randomly setting entire feature maps to zero.
- **Conv1D Layer**: Extracts local patterns in text.
- **Bidirectional LSTM Layer**: Captures long-term dependencies and context in both forward and backward directions.
- **GlobalAveragePooling1D Layer**: Reduces the output dimensionality.
- **Dense Layers**: Fully connected layers for classification.

### Why this architecture? 💭
- **Embedding Layer** with GloVe vectors helps in understanding the semantic meaning of words.
- **SpatialDropout1D** helps in regularization, making the model more robust.
- **Conv1D** captures local dependencies and patterns in the text.
- **Bi-LSTM** handles long-term dependencies and context in both directions.
- **GlobalAveragePooling1D** reduces the dimensionality without losing important features.
- **Dense Layers** are used for final classification.

```python
# Define the model
sequence_input = Input(shape=(max_len,))
x = Embedding(min(max_features, vocab_size), embed_size, weights=[embedding_matrix], trainable=False)(sequence_input)
x = SpatialDropout1D(0.2)(x)
x = Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x)
x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
avg_pool = GlobalAveragePooling1D()(x)
x = Dense(128, activation='relu')(avg_pool)
x = Dropout(0.1)(x)
preds = Dense(y_train.shape[1], activation="softmax")(x)
```

## Training 🏋️‍♂️

The model is trained using class weights to handle class imbalance and `ModelCheckpoint` to save the best model based on validation accuracy.

### Why use class weights? ⚖️
Class weights ensure that the model gives proper attention to minority classes, preventing it from being biased towards the majority class.

### Why use ModelCheckpoint? 💾
`ModelCheckpoint` saves the best model during training, ensuring that the best performing model is preserved.

```python
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
```

## Evaluation 📊

After training, the model is evaluated on the test set, and training/validation accuracy and loss are plotted.

```python
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
```

## Usage 📜

To make predictions on new data, load the trained model and preprocess the input text:

```python
# Load the saved model and tokenizer
best_model = load_model(model_path)

# Load the LabelEncoder and Tokenizer from disk
with open(label_encoder_path, 'rb') as le_file:
    label_encoder = pickle.load(le_file)

with open(tokenizer_path, 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Predict function
def predict_label(comment):
    comment_normalized = normalize_text(comment)
    comment_seq = tokenizer.texts_to_sequences([comment_normalized])
    comment_padded = pad_sequences(comment_seq, maxlen=1020)
    prediction = best_model.predict(comment_padded)
    predicted_class_index = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class_index)
    return predicted_label[0]
```

## Streamlit Web App 🌐

Deploy a Streamlit web app to interact with the model:

```python
import streamlit as st

def main():
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
        if st.button("Predict 🚀"):
            if comment:
                label = predict_label(comment)
                st.success(f"Predicted Label: {label} 🏆")
            else:
                st.warning("Please enter a comment to classify. ⚠️")

    st.sidebar.subheader("About 💡")
    st.sidebar.text("Text Classification App 📜🤖")
    st.sidebar.text("Built with Streamlit ❤️")

if __name__ == '__main__':
    main()
```

Run the Streamlit app using the command:

```sh
streamlit run app.py
```

## Why We Didn't Use GPT-2 or LLM 🤔💸

We chose not to use GPT-2 or other large language models for text generation in order to save on computational costs. Instead, we used the `set_seed` function in the transformers pipeline to create more comments, usernames, and labels. This approach allowed us to generate the necessary data efficiently and cost-effectively.

## Requirements 📋

- Python 3.11 virtual environment
- Ubuntu 23.04
- NVIDIA GPU with 6GB memory

Training the model took

 approximately 166 minutes on a dataset of about 3,000 rows, achieving a test accuracy of 94%.

### `requirements.txt`
```
absl-py==2.1.0
asttokens==2.4.1
astunparse==1.6.3
certifi==2024.2.2
charset-normalizer==3.3.2
click==8.1.7
comm==0.2.2
contourpy==1.2.1
cycler==0.12.1
debugpy==1.8.1
decorator==5.1.1
executing==2.0.1
filelock==3.14.0
flatbuffers==24.3.25
fonttools==4.51.0
fsspec==2024.5.0
gast==0.5.4
google-pasta==0.2.0
grpcio==1.63.0
h5py==3.11.0
huggingface-hub==0.23.0
idna==3.7
ipykernel==6.29.4
ipython==8.24.0
jedi==0.19.1
joblib==1.4.2
jupyter_client==8.6.1
jupyter_core==5.7.2
keras==3.3.3
kiwisolver==1.4.5
libclang==18.1.1
Markdown==3.6
markdown-it-py==3.0.0
MarkupSafe==2.1.5
matplotlib==3.9.0
matplotlib-inline==0.1.7
mdurl==0.1.2
ml-dtypes==0.3.2
namex==0.0.8
nest-asyncio==1.6.0
nltk==3.8.1
numpy==1.26.4
nvidia-cublas-cu12==12.3.4.1
nvidia-cuda-cupti-cu12==12.3.101
nvidia-cuda-nvcc-cu12==12.3.107
nvidia-cuda-nvrtc-cu12==12.3.107
nvidia-cuda-runtime-cu12==12.3.101
nvidia-cudnn-cu12==8.9.7.29
nvidia-cufft-cu12==11.0.12.1
nvidia-curand-cu12==10.3.4.107
nvidia-cusolver-cu12==11.5.4.101
nvidia-cusparse-cu12==12.2.0.103
nvidia-nccl-cu12==2.19.3
nvidia-nvjitlink-cu12==12.3.101
opt-einsum==3.3.0
optree==0.11.0
packaging==24.0
pandas==2.2.2
parso==0.8.4
pexpect==4.9.0
pillow==10.3.0
platformdirs==4.2.1
prompt-toolkit==3.0.43
protobuf==4.25.3
psutil==5.9.8
ptyprocess==0.7.0
pure-eval==0.2.2
Pygments==2.18.0
pyparsing==3.1.2
python-dateutil==2.9.0.post0
pytz==2024.1
PyYAML==6.0.1
pyzmq==26.0.3
regex==2024.5.15
requests==2.31.0
rich==13.7.1
safetensors==0.4.3
scikit-learn==1.4.2
scipy==1.13.0
seaborn==0.13.2
six==1.16.0
stack-data==0.6.3
streamlit==1.34.0
tensorboard==2.16.2
tensorboard-data-server==0.7.2
tensorflow==2.16.1
tensorflow-io-gcs-filesystem==0.37.0
termcolor==2.4.0
tf_keras==2.16.0
threadpoolctl==3.5.0
tokenizers==0.19.1
tornado==6.4
tqdm==4.66.4
traitlets==5.14.3
transformers==4.41.0
typing_extensions==4.11.0
tzdata==2024.1
urllib3==2.2.1
wcwidth==0.2.13
Werkzeug==3.0.3
wrapt==1.16.0
```

## References 📚

- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [TensorFlow and Keras Documentation](https://www.tensorflow.org/api_docs)
- [Streamlit Documentation](https://docs.streamlit.io/en/stable/)

---

![GitHub](https://img.icons8.com/ios/452/github.png)
Feel free to contribute to this project on GitHub! 🖥️

---

This README provides a comprehensive guide to understand, setup, train, evaluate, and deploy the multilabel text classification model using CNN and Bi-LSTM networks. 🏆🎉

Happy Coding! 💻😃
