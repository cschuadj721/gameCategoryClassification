# predict_genre.py

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer, Multiply
import numpy as np
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Ensure NLTK data is downloaded
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


# Load Tokenizer
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


# Load Label Encoder
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


# Preprocess Text
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


# Load Model with Custom Layers
def load_trained_model(model_path):
    """
    Load the trained Keras model, including custom layers.
    Args:
        model_path (str): Path to the saved .h5 model file.
    Returns:
        Keras Model object
    """
    try:
        model = load_model(
            model_path,
            custom_objects={
                'AttentionWeightedSum': AttentionWeightedSum,
                'ReduceSumCustom': ReduceSumCustom
            }
        )
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e


# Predict Genre
def predict_genre(review, model, tokenizer, label_encoder, max_length=297):
    """
    Predict the genre of a given review.
    Args:
        review (str): The review text.
        model (Keras Model): The pre-trained Keras model.
        tokenizer (Tokenizer): The fitted tokenizer.
        label_encoder (LabelEncoder): The fitted label encoder.
        max_length (int): Maximum length for padding.
    Returns:
        str: The predicted genre.
    """
    # Initialize preprocessing tools
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Preprocess the review
    preprocessed_review = preprocess_text(review, lemmatizer, stop_words)

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([preprocessed_review])

    # Pad the sequence
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

    # Predict
    prediction = model.predict(padded_sequence)
    predicted_index = np.argmax(prediction, axis=1)
    predicted_genre = label_encoder.inverse_transform(predicted_index)[0]

    return predicted_genre


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict the genre of a game review.")
    parser.add_argument('--review', type=str, required=True, help="The game review text to classify.")


    parser.add_argument('--model_path', type=str, default='./final_model.h5',
                        help="Path to the saved Keras model (.h5 file).")
    parser.add_argument('--tokenizer_path', type=str, default='./models/tokenizer.pickle',
                        help="Path to the tokenizer pickle file.")
    parser.add_argument('--encoder_path', type=str, default='./models/encoder.pickle',
                        help="Path to the label encoder pickle file.")
    parser.add_argument('--max_length', type=int, default=297, help="Maximum sequence length used during training.")

    args = parser.parse_args()

    # Check if files exist
    for file_path in [args.model_path, args.tokenizer_path, args.encoder_path]:
        if not os.path.exists(file_path):
            print(f"Error: The file '{file_path}' does not exist.")
            sys.exit(1)

    # Load resources
    tokenizer = load_tokenizer(args.tokenizer_path)
    label_encoder = load_label_encoder(args.encoder_path)

    # Load model
    model = load_trained_model(args.model_path)

    # Predict genre
    predicted_genre = predict_genre(
        review=args.review,
        model=model,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        max_length=args.max_length
    )

    # Output the result
    print(f"Predicted Genre: {predicted_genre}")
