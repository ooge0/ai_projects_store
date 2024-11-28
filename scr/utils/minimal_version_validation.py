import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Mock data
data = pd.DataFrame({'text': ['happy day', 'sad moment'], 'label': ['positive', 'negative']})
stop_words = set(stopwords.words('english'))
data['text'] = data['text'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

# Tokenize
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(data['text'])
sequences = tokenizer.texts_to_sequences(data['text'])
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=10)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['label'])

# Model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_shape=(10,)),  # Define input_shape
    Bidirectional(LSTM(128)),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

