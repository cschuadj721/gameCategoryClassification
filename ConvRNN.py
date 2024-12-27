import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, MaxPool1D, Flatten, GlobalMaxPooling1D
from tensorflow.keras.models import load_model
import pickle
#
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
exit()

# Load the prepared datasets
X_train = np.load('datasets/filtered_reviews_X_train.npy')
Y_train = np.load('datasets/filtered_reviews_Y_train.npy')
X_test = np.load('datasets/filtered_reviews_X_test.npy')
Y_test = np.load('datasets/filtered_reviews_Y_test.npy')

with open('./models/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)


# Vocabulary size and maximum sequence length
vocab_size = min(len(tokenizer.word_index) + 1, 10000)  # Include padding token and limit to 10,000
max_length = X_train.shape[1]  # The padded sequence length

# Build the model
model = Sequential([

    # #단어 수가 많기 때문에 output_dim 128로 설정
    # Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),  # Embedding layer
    # # Embedding(input_dim=vocab_size, output_dim=300, input_length=max_length),
    # Conv1D(32, kernel_size=5, padding = 'same', activation = 'relu'),
    # # MaxPool1D(pool_size=1),
    # GlobalMaxPooling1D(),
    # LSTM(128, activation = 'tanh', return_sequences=True),
    # Dropout(0.3),
    # LSTM(64, activation = 'tanh', return_sequences=True),  # LSTM layer
    # Dropout(0.3),  # Dropout for regularization
    # LSTM(64, activation = 'tanh'),
    # Dropout(0.3),
    # Dense(128, activation='relu'),  # Fully connected layer
    # Dropout(0.3),
    # Flatten(),
    # Dense(128, activation='relu'),
    # Dense(Y_train.shape[1], activation='softmax')  # Output layer for classification

    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
    LSTM(128, return_sequences=False),  # A single LSTM layer
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(Y_train.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.1)

# Save the model
model.save('./models/news_classification_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")




#
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# exit()

# Load the prepared datasets
# X_train = np.load('./filtered_reviews_X_train.npy')
# Y_train = np.load('./filtered_reviews_Y_train.npy')
# X_test = np.load('./filtered_reviews_X_test.npy')
# Y_test = np.load('./filtered_reviews_Y_test.npy')
#
# # Vocabulary size and maximum sequence length
# vocab_size = X_train.max() + 1  # Assuming tokenizer indices are used, max index + 1 gives vocab size
# max_length = X_train.shape[1]  # The padded sequence length
#
# # Build the model
# model = Sequential([
#     Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),  # Embedding layer
#     LSTM(128, return_sequences=False),  # LSTM layer
#     Dropout(0.5),  # Dropout for regularization
#     Dense(128, activation='relu'),  # Fully connected layer
#     Dropout(0.5),
#     Dense(Y_train.shape[1], activation='softmax')  # Output layer for classification
# ])
#
# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # Train the model
# history = model.fit(X_train, Y_train, epochs=30, batch_size=64, validation_split=0.1)
#
# # Save the model
# model.save('./news_classification_model.h5')
#
# # Evaluate the model
# loss, accuracy = model.evaluate(X_test, Y_test)
# print(f"Test Loss: {loss}")
# print(f"Test Accuracy: {accuracy}")
