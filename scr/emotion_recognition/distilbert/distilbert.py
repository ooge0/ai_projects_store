from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
import pandas as pd
import numpy as np


# Load dataset
def load_data(file_path):
    """
    Load data from a CSV file and preprocess it.
    """
    data = pd.read_csv(file_path)
    return data['text'].tolist(), data['label'].tolist()


# Preprocess data for DistilBERT
def preprocess_data(texts, labels, tokenizer, max_length=128):
    """
    Tokenize and encode text for DistilBERT.
    Returns encoded inputs and labels.
    """
    encodings = tokenizer(
        texts, truncation=True, padding=True, max_length=max_length, return_tensors="tf"
    )
    label_dict = {label: idx for idx, label in enumerate(set(labels))}
    encoded_labels = [label_dict[label] for label in labels]
    return encodings, np.array(encoded_labels), label_dict


# Build and compile DistilBERT model
def build_model(num_labels):
    """
    Create a DistilBERT model for text classification.
    """
    model = TFDistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels
    )
    optimizer = Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=["accuracy"])
    return model


# Evaluation and reporting
def evaluate_model(model, X_test, y_test, label_dict):
    """
    Evaluate the model on test data and print classification report.
    """
    predictions = model.predict(X_test).logits
    predicted_labels = tf.argmax(predictions, axis=1).numpy()
    reverse_label_dict = {v: k for k, v in label_dict.items()}
    y_test_labels = [reverse_label_dict[label] for label in y_test]
    predicted_labels_readable = [reverse_label_dict[label] for label in predicted_labels]

    print("\nClassification Report:")
    print(classification_report(y_test_labels, predicted_labels_readable))


# Main
if __name__ == "__main__":
    # Load and preprocess dataset
    texts, labels = load_data("../lstm_tuning/processed/emotions.csv")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    encodings, encoded_labels, label_dict = preprocess_data(texts, labels, tokenizer)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        encodings, encoded_labels, test_size=0.2, random_state=42
    )

    # Build and train model
    model = build_model(num_labels=len(label_dict))
    model.fit(
        x={"input_ids": X_train["input_ids"], "attention_mask": X_train["attention_mask"]},
        y=y_train,
        validation_split=0.1,
        batch_size=16,
        epochs=3,
    )

    # Evaluate model
    evaluate_model(model, X_test, y_test, label_dict)
