import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, SpatialDropout1D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Load the dataset
df = pd.read_csv('FINAL_DATASET.csv')

# Split into features (X) and labels (y)
X = df['content']
y = df['target']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.astype(str)
X_test = X_test.astype(str)

# Tokenization settings
MAX_VOCAB_SIZE = 10000  
MAX_LENGTH = 200

# Tokenize the text data
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Pad sequences for uniform size
X_train_padded = pad_sequences(X_train_sequences, padding='post', maxlen=MAX_LENGTH)
X_test_padded = pad_sequences(X_test_sequences, padding='post', maxlen=MAX_LENGTH)

# Build LSTM model
model = Sequential([
    Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=128, input_length=MAX_LENGTH),
    SpatialDropout1D(0.3),
    Bidirectional(LSTM(150, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
    BatchNormalization(),
    LSTM(100, dropout=0.3, recurrent_dropout=0.3),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])

# Train the model
history = model.fit(X_train_padded, y_train, epochs=10, batch_size=64, validation_data=(X_test_padded, y_test), verbose=2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test_padded, y_test, verbose=2)
print(f"✅ Test Accuracy: {test_acc}")

# Save model using pickle
with open('fake_news_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save tokenizer using pickle
with open('tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

print("✅ Model and tokenizer saved as pickle files.")