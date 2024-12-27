import numpy as np
import tensorflow as tf
import random
import os
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


# 2. Model architecture with CNN + Bi-LSTM with attention mechanism
def create_model(vocab_size, embedding_dim, max_len, num_classes):
    input_layer = Input(shape=(max_len,), name="input_layer")

    # Embedding layer
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="embedding")(input_layer)

    # CNN branches
    filter_sizes = [3, 5, 7]
    num_filters = 32
    cnn_branches = []
    for fsz in filter_sizes:
        branch = Conv1D(filters=num_filters, kernel_size=fsz, padding='same', activation='relu')(embedding_layer)
        branch = GlobalMaxPooling1D()(branch)
        cnn_branches.append(branch)

    cnn_merged = concatenate(cnn_branches)  # shape: (batch, 32*3)= (batch, 96)

    # Bi-LSTM with attention mechanism
    lstm_out = Bidirectional(LSTM(48, activation='tanh', return_sequences=True, dropout=0.4, recurrent_dropout=0.4))(
        embedding_layer)

    # Attention mechanism
    score_dense = Dense(64, activation='tanh')(lstm_out)  # (batch, 16, 64)
    score_vec = Dense(1)(score_dense)  # (batch, 16, 1)

    # Squeeze to (batch, 16) so softmax goes across time steps
    score_vec_squeezed = Lambda(lambda x: tf.squeeze(x, axis=-1))(score_vec)
    attention_weights = Activation('softmax', name='attention_weights')(score_vec_squeezed)  # shape: (batch, 16)

    # Expand back to (batch, 16, 1)
    attention_weights_expanded = Lambda(lambda x: tf.expand_dims(x, axis=-1))(attention_weights)

    # Multiply LSTM output with attention weights
    weighted_lstm_out = Multiply(name='weighted_lstm_out')([lstm_out, attention_weights_expanded])

    # Sum over the time dimension -> (batch, 96)
    attention_output = Lambda(
        lambda x: tf.reduce_sum(x, axis=1),
        output_shape=(96,))(weighted_lstm_out)

    # Merge CNN + Bi-LSTM (attention) outputs
    merged = concatenate([cnn_merged, attention_output])  # shape: (batch, 192)

    # Dense layers
    dense = Dense(128, activation='relu')(merged)
    dense = Dropout(0.4)(dense)
    output_layer = Dense(num_classes, activation='softmax')(dense)  # Change num_classes to 11

    # Build & compile the model
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

    # 2.3 Ensemble Setup
    n_models = 3  # number of models to ensemble
    batch_size = 128
    epochs = 500

    all_models = []
    histories = []

    # Train each model with a different seed (or same seed, up to you)
    for i in range(n_models):
        print(f"\n=== Training Model {i + 1} / {n_models} ===\n")
        set_seed(42 + i)  # e.g., seeds 42, 43, 44

        # Create and compile model
        model_i = create_model(vocab_size, embedding_dim, max_len, num_classes)

        # Define callbacks
        checkpoint_cb = ModelCheckpoint(
            filepath=f'./best_news_model_{i}.keras',
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
        fit_hist = model_i.fit(
            X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, Y_test),
            callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb]
        )

        # Evaluate on test
        score_i = model_i.evaluate(X_test, Y_test, verbose=0)
        print(f"Model {i + 1} test set accuracy: {score_i[1]:.4f}")

        # Optionally store each final model_i as an .h5 file
        final_val_acc = fit_hist.history['val_accuracy'][-1]
        model_i.save(f'./final_model_{i}_valacc_{final_val_acc:.4f}.h5', save_format='h5')

        # Add to ensemble list
        all_models.append(model_i)
        histories.append(fit_hist)

    print("\n=== All models trained! ===")

    ###########################################################################
    # 2.4 Ensemble Predictions
    ###########################################################################
    predictions_sum = None

    for i, model_i in enumerate(all_models):
        # Probability predictions: shape (test_size, num_classes)
        y_pred_probs = model_i.predict(X_test)
        if predictions_sum is None:
            predictions_sum = y_pred_probs
        else:
            predictions_sum += y_pred_probs

    # Average predicted probabilities across all models
    ensemble_probs = predictions_sum / n_models  # shape: (test_size, num_classes)
    ensemble_pred_class = np.argmax(ensemble_probs, axis=1)

    # Convert one-hot Y_test to class indices, if necessary
    true_class = np.argmax(Y_test, axis=1)

    # Compute ensemble accuracy
    ensemble_accuracy = np.mean(ensemble_pred_class == true_class)
    print(f"\nEnsemble Test Accuracy: {ensemble_accuracy:.4f}")
