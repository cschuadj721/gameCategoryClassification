import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Conv1D, GlobalMaxPooling1D, Bidirectional, \
    concatenate, Lambda, Multiply, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Restrict TensorFlow to GPU only
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)  # Prevent memory overflow
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
        print("GPU is being used for training.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Running on CPU.")


# 1. Set random seed for reproducibility
def set_seed(seed_value):
    """Fix random seeds for reproducibility."""
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)


# Custom Lambda function for attention mechanism to handle TensorFlow operations
def attention_weighted_sum(x):
    import tensorflow as tf  # Explicitly import TensorFlow inside the Lambda function
    # Squeeze to (batch, 16) so softmax goes across time steps
    score_vec_squeezed = tf.squeeze(x, axis=-1)
    attention_weights = tf.nn.softmax(score_vec_squeezed, axis=1)  # shape: (batch, 16)
    # Expand back to (batch, 16, 1)
    attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1)
    return attention_weights_expanded


# Custom Lambda function for reduce_sum operation
def reduce_sum_custom(x):
    import tensorflow as tf  # Explicitly import TensorFlow inside the Lambda function
    return tf.reduce_sum(x, axis=1)


# Model definition with attention_output fixed
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

    # Lambda for attention mechanism
    attention_weights_expanded = Lambda(attention_weighted_sum)(score_vec)
    weighted_lstm_out = Multiply()([lstm_out, attention_weights_expanded])

    # Lambda for summing over the time dimension with TensorFlow operations (now using custom function)
    attention_output = Lambda(reduce_sum_custom)(weighted_lstm_out)  # Use the custom reduce_sum_custom function

    merged = concatenate([cnn_merged, attention_output])

    dense = Dense(128, activation='relu')(merged)
    dense = Dropout(0.4)(dense)
    output_layer = Dense(num_classes, activation='softmax')(dense)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])
    return model


if __name__ == "__main__":
    # 2.1 Load the prepared datasets
    X_train = np.load('./filtered_reviews_X_train.npy', allow_pickle=True)
    Y_train = np.load('./filtered_reviews_Y_train.npy', allow_pickle=True)
    X_test = np.load('./filtered_reviews_X_test.npy', allow_pickle=True)
    Y_test = np.load('./filtered_reviews_Y_test.npy', allow_pickle=True)

    print('X_train shape:', X_train.shape, 'Y_train shape:', Y_train.shape)
    print('X_test shape:', X_test.shape, 'Y_test shape:', Y_test.shape)

    # 2.2 Hyperparameters
    vocab_size = 10000  # For example, if you capped the vocabulary size at 10,000 words
    embedding_dim = 200
    max_len = 297  # Update to match the length of your input sequences

    num_classes = 11

    # 2.3 Model Setup (Single model instead of ensemble)
    batch_size = 128
    epochs = 1

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
    model.save('./final_model.h5', save_format='h5')

    print(f"Model saved as final_model.h5")
