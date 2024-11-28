import nltk
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.models import Sequential

nltk.download('stopwords')
from nltk.corpus import stopwords


# Load and clean data
def load_data(file_path):
    """
    Load data from a CSV file and perform initial preprocessing.
    - Removes stop words.
    """
    data = pd.read_csv(file_path)
    stop_words = set(stopwords.words('english'))
    data['text'] = data['text'].apply(lambda x: ' '.join(
        word for word in x.split() if word.lower() not in stop_words))
    return data


# Preprocess text data
def preprocess_data(data):
    """
    Tokenize text and encode labels.
    Returns padded sequences and encoded labels.
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(data['text'])
    sequences = tokenizer.texts_to_sequences(data['text'])
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data['label'])
    return padded_sequences, labels, tokenizer


# Define TextCNN model
def build_model(vocab_size, embedding_dim=100):
    """
    Build a TextCNN model for text classification.
    - Uses Conv1D layers with global max-pooling.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=100),
        Conv1D(128, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dense(7, activation='softmax')  # Assuming 7 emotion classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Training and evaluation
if __name__ == "__main__":
    # Load dataset
    data = load_data("../lstm_tuning/processed/emotions.csv")

    # Preprocess the data
    X, y, tokenizer = preprocess_data(data)
    vocab_size = len(tokenizer.word_index) + 1

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model(vocab_size)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
