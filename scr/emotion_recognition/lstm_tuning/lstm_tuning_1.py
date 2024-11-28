import nltk
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential

# Ensure NLTK stopwords are available
nltk.download('stopwords')
from nltk.corpus import stopwords


# Load data
def load_data(file_path):
    """Load and clean the text data."""
    data = pd.read_csv(file_path)
    stop_words = set(stopwords.words('english'))
    data['text'] = data['text'].apply(
        lambda x: ' '.join(word for word in x.split() if word not in stop_words)
    )
    unique_emotions = data['label'].nunique()
    print(f"Number of unique emotions: {unique_emotions}")
    return data


# Preprocessing
def preprocess_data(data):
    """Tokenize and encode text."""
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(data['text'])
    sequences = tokenizer.texts_to_sequences(data['text'])
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data['label'])
    return padded_sequences, labels, tokenizer


# Build Model
def build_model(vocab_size, embedding_dim=100, num_classes=7, lstm_units=128, dense_units=64):
    """Define a bidirectional LSTM model."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        Bidirectional(LSTM(lstm_units, activation='tanh')),  # Explicit activation
        Dense(dense_units, activation='relu'),
        Dense(num_classes, activation='softmax')  # Use num_classes instead of 7
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Train and Evaluate
if __name__ == "__main__":
    try:
        # Load and preprocess data
        data = load_data("processed/emotions.csv")
        X, y, tokenizer = preprocess_data(data)
        vocab_size = len(tokenizer.word_index) + 1

        # Explicit train/test split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Build model with hyperparameters
        model = build_model(vocab_size, lstm_units=128, dense_units=64)

        # Define early stopping and model checkpointing with correct file extension
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')

        # Train the model
        model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, model_checkpoint]
        )

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

    except Exception as e:
        print(f"An error occurred: {e}")
