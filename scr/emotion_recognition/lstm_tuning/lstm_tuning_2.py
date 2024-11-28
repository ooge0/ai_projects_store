import nltk
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

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


# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch > 5:
        return lr * 0.9
    return lr


# Build Model
def build_model(vocab_size, embedding_dim=100, num_classes=7):
    """Define a bidirectional LSTM model with tuning options."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=100),
        Bidirectional(LSTM(128, activation='tanh', return_sequences=True)),
        Dropout(0.3),
        LSTM(128, activation='tanh'),
        Dense(64, kernel_regularizer=l2(0.01)),
        LeakyReLU(alpha=0.1),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
        lr_scheduler = LearningRateScheduler(scheduler)

        # Build model
        model = build_model(vocab_size)

        # Train the model
        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val),
                  callbacks=[early_stopping, model_checkpoint, lr_scheduler])

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

    except Exception as e:
        print(f"An error occurred: {e}")
