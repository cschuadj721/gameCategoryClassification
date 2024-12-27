# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Multiply
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
import os

# Ensure NLTK data is downloaded
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Define Custom Layers (Must match the training script)
class AttentionWeightedSum(Layer):
    def __init__(self, **kwargs):
        super(AttentionWeightedSum, self).__init__(**kwargs)

    def call(self, inputs):
        """
        Computes attention weights and expands dimensions for multiplication.
        Args:
            inputs: Tensor of shape (batch_size, time_steps, 1)
        Returns:
            Tensor of shape (batch_size, time_steps, 1)
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
        Sums the weighted LSTM outputs over the time_steps dimension.
        Args:
            inputs: Tensor of shape (batch_size, time_steps, units)
        Returns:
            Tensor of shape (batch_size, units)
        """
        return tf.reduce_sum(inputs, axis=1)


def load_tokenizer(tokenizer_path):
    """
    Load the tokenizer from a pickle file.
    Args:
        tokenizer_path (str): Path to the tokenizer pickle file.
    Returns:
        Tokenizer object
    """
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


def load_label_encoder(encoder_path):
    """
    Load the label encoder from a pickle file.
    Args:
        encoder_path (str): Path to the label encoder pickle file.
    Returns:
        LabelEncoder object
    """
    with open(encoder_path, 'rb') as handle:
        label_encoder = pickle.load(handle)
    return label_encoder


def preprocess_text(sentence, lemmatizer, stop_words):
    """
    Preprocess the input text: tokenize, lowercase, remove non-alphabetic tokens,
    remove stopwords, and lemmatize.
    Args:
        sentence (str): The text to preprocess.
        lemmatizer (WordNetLemmatizer): An instance of WordNetLemmatizer.
        stop_words (set): A set of stopwords to remove.
    Returns:
        str: The preprocessed text.
    """
    tokens = word_tokenize(sentence)
    tokens = [word.lower() for word in tokens if word.isalpha()]  # Keep only alphabetic words
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize words
    return ' '.join(tokens)


def load_model_with_custom_layers(model_path):
    """
    Load the Keras model, including custom layers.
    Args:
        model_path (str): Path to the saved .h5 model file.
    Returns:
        Keras Model object
    """
    # Load the model with custom objects
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'AttentionWeightedSum': AttentionWeightedSum,
            'ReduceSumCustom': ReduceSumCustom
        }
    )
    return model


def main(new_csv_path, model_path, tokenizer_path, encoder_path, max_length=297, sample_size=1000):
    """
    Main function to load the model, preprocess new data, make predictions, and evaluate.
    Args:
        new_csv_path (str): Path to the new CSV file containing 'review' and 'genre' columns.
        model_path (str): Path to the saved Keras model (.h5 file).
        tokenizer_path (str): Path to the saved tokenizer pickle file.
        encoder_path (str): Path to the saved label encoder pickle file.
        max_length (int): Maximum length for padding sequences.
        sample_size (int): Number of samples to test. If the dataset has more, it will be randomly sampled.
    """
    # Load the tokenizer and label encoder
    tokenizer = load_tokenizer(tokenizer_path)
    label_encoder = load_label_encoder(encoder_path)
    print("Tokenizer and Label Encoder loaded successfully.")

    # Load the model
    model = load_model_with_custom_layers(model_path)
    print("Model loaded successfully.")

    # Load the new data
    df_new = pd.read_csv(new_csv_path)
    if 'review' not in df_new.columns or 'genre' not in df_new.columns:
        raise ValueError("The CSV file must contain 'review' and 'genre' columns.")

    # Remove duplicates
    df_new.drop_duplicates(inplace=True)
    df_new.reset_index(drop=True, inplace=True)
    print(f"Loaded {len(df_new)} records from {new_csv_path}.")

    # Sample the data if necessary
    if len(df_new) > sample_size:
        df_new = df_new.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampled {sample_size} records for testing.")
    else:
        print(f"Using all {len(df_new)} records for testing.")

    # Extract columns
    X_real = df_new['review']
    Y_real = df_new['genre']

    # Initialize preprocessing tools
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Preprocess the reviews
    X_preprocessed = X_real.apply(lambda x: preprocess_text(x, lemmatizer, stop_words))
    print("Preprocessing completed.")

    # Convert text to sequences
    tokenized_X = tokenizer.texts_to_sequences(X_preprocessed)
    print("Text converted to sequences.")

    # Pad sequences
    X_pad_new = pad_sequences(tokenized_X, maxlen=max_length, padding='post')
    print(f"Padded sequences to a maximum length of {max_length}.")

    # Encode the true labels
    try:
        true_indices = label_encoder.transform(Y_real)
    except Exception as e:
        print("Error in label encoding:", e)
        return

    # Make predictions
    pred_probabilities = model.predict(X_pad_new, batch_size=32, verbose=1)
    pred_indices = np.argmax(pred_probabilities, axis=1)
    pred_labels = label_encoder.inverse_transform(pred_indices)
    print("Predictions completed.")

    # Calculate accuracy
    accuracy = np.mean(pred_indices == true_indices)
    print(f"Test Accuracy on {new_csv_path} (sample of {len(df_new)}): {accuracy:.4f}")

    # Optionally, show some random samples of predictions
    print("\n-- Random Samples of Predictions --")
    num_samples_to_show = min(10, len(df_new))
    random_samples = random.sample(range(len(df_new)), num_samples_to_show)
    for idx in random_samples:
        print(f"Review: {df_new.loc[idx, 'review']}")
        print(f" True Genre   : {df_new.loc[idx, 'genre']}")
        print(f" Predicted Gen: {pred_labels[idx]}")
        print("--------------")


if __name__ == "__main__":
    # Define paths
    NEW_CSV_PATH = './datasets/filtered_reviews_with_word_count_500plus.csv'  # Replace with your new CSV file path
    MODEL_PATH = './models/final_model.h5'
    TOKENIZER_PATH = './models/tokenizer.pickle'
    ENCODER_PATH = './models/encoder.pickle'
    MAX_LENGTH = 297  # Must match the training max_length
    SAMPLE_SIZE = 1000  # Number of samples to test

    # Check if all files exist
    for path in [NEW_CSV_PATH, MODEL_PATH, TOKENIZER_PATH, ENCODER_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The required file '{path}' does not exist.")

    # Run the main function
    main(
        new_csv_path=NEW_CSV_PATH,
        model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        encoder_path=ENCODER_PATH,
        max_length=MAX_LENGTH,
        sample_size=SAMPLE_SIZE
    )
