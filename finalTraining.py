# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import os
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input, Embedding, LSTM, Dense, Dropout, Conv1D, GlobalMaxPooling1D, \
    Bidirectional, concatenate, Lambda, Multiply, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# # Restrict TensorFlow to GPU only
# physical_devices = tf.config.list_physical_devices('GPU')
# if physical_devices:
#     try:
#         tf.config.experimental.set_memory_growth(physical_devices[0], True)  # Prevent memory overflow
#         tf.config.set_visible_devices(physical_devices[0], 'GPU')
#         print("GPU is being used for training.")
#     except RuntimeError as e:
#         print(e)
# else:
#     print("No GPU found. Running on CPU.")


class AttentionWeightedSum(Layer):
    def __init__(self, **kwargs):
        super(AttentionWeightedSum, self).__init__(**kwargs)

    def call(self, inputs):
        """
        inputs: score_vec, shape (batch_size, time_steps, 1)
        """
        # Remove the last dimension: (batch_size, time_steps)
        score_vec_squeezed = tf.squeeze(inputs, axis=-1)

        # Apply softmax to get attention weights: (batch_size, time_steps)
        attention_weights = tf.nn.softmax(score_vec_squeezed, axis=1)

        # Expand dimensions to multiply with lstm_out: (batch_size, time_steps, 1)
        attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1)

        return attention_weights_expanded


class ReduceSumCustom(Layer):
    def __init__(self, **kwargs):
        super(ReduceSumCustom, self).__init__(**kwargs)

    def call(self, inputs):
        """
        inputs: weighted_lstm_out, shape (batch_size, time_steps, units)
        """
        # Sum over the time_steps dimension: (batch_size, units)
        return tf.reduce_sum(inputs, axis=1)


def create_model(vocab_size, embedding_dim, max_len, num_classes):
    input_layer = Input(shape=(max_len,), name="input_layer")
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)

    # CNN branches
    cnn_branches = []
    filter_sizes = [3, 5, 7]
    num_filters = 32
    for fsz in filter_sizes:
        branch = Conv1D(filters=num_filters, kernel_size=fsz, padding='same', activation='relu')(embedding_layer)
        branch = GlobalMaxPooling1D()(branch)
        cnn_branches.append(branch)

    cnn_merged = concatenate(cnn_branches)

    # Bi-LSTM with attention mechanism
    lstm_out = Bidirectional(LSTM(48, activation='tanh', return_sequences=True, dropout=0.4, recurrent_dropout=0.4))(
        embedding_layer)
    score_dense = Dense(64, activation='tanh')(lstm_out)
    score_vec = Dense(1)(score_dense)

    # Custom Attention Layers
    attention_weights_expanded = AttentionWeightedSum()(score_vec)
    weighted_lstm_out = Multiply()([lstm_out, attention_weights_expanded])
    attention_output = ReduceSumCustom()(weighted_lstm_out)

    merged = concatenate([cnn_merged, attention_output])

    dense = Dense(128, activation='relu')(merged)
    dense = Dropout(0.4)(dense)
    output_layer = Dense(num_classes, activation='softmax')(dense)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])
    return model


if __name__ == "__main__":
    X_train = np.load('datasets/filtered_reviews_X_train.npy', allow_pickle=True)
    Y_train = np.load('datasets/filtered_reviews_Y_train.npy', allow_pickle=True)
    X_test = np.load('datasets/filtered_reviews_X_test.npy', allow_pickle=True)
    Y_test = np.load('datasets/filtered_reviews_Y_test.npy', allow_pickle=True)

    print('X_train shape:', X_train.shape, 'Y_train shape:', Y_train.shape)
    print('X_test shape:', X_test.shape, 'Y_test shape:', Y_test.shape)

    # 3. Hyperparameters
    vocab_size = 10000  # Must match the tokenizer's vocab size
    embedding_dim = 200
    max_len = 297  # Must match the padding length used during training

    num_classes = 11  # Number of genres

    # 4. Model Setup
    batch_size = 128
    epochs = 1  # Increase epochs as needed

    # Create and compile model
    model = create_model(vocab_size, embedding_dim, max_len, num_classes)

    # Define callbacks
    checkpoint_cb = ModelCheckpoint(
        filepath='./best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    early_stopping_cb = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_cb = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-6
    )

    # Train the model
    fit_hist = model.fit(
        X_train, Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, Y_test),
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb]
    )

    # Evaluate on test
    score = model.evaluate(X_test, Y_test, verbose=1)
    print(f"Test Loss: {score[0]:.4f}")
    print(f"Test Accuracy: {score[1]:.4f}")

    # Save the model in .h5 format
    model.save('./models/final_model.h5', save_format='h5')

    print(f"Model saved as final_model.h5")
