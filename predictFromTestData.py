import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer

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


# 1. Set random seed for reproducibility
def set_seed(seed_value=42):
    """Fix random seeds for reproducibility."""
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

set_seed(42)  # Set the seed at the beginning

if __name__ == "__main__":
    # 2. Load the test dataset
    X_test = np.load('./datasets/filtered_reviews_X_test.npy', allow_pickle=True)
    Y_test = np.load('./datasets/filtered_reviews_Y_test.npy', allow_pickle=True)  # Fixed incorrect file name

    print('X_test shape:', X_test.shape, 'Y_test shape:', Y_test.shape)

    # 3. Path to the saved model
    model_path = './models/final_model.h5'  # Update this path to your saved model file
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the path is correct.")

    # 4. Load the pre-trained model with custom objects
    model = load_model(
        model_path,
        custom_objects={
            'AttentionWeightedSum': AttentionWeightedSum,
            'ReduceSumCustom': ReduceSumCustom
        }
    )
    model.summary()  # Optional: View the loaded model architecture

    # 5. Evaluate the model on the test data
    score = model.evaluate(X_test, Y_test, verbose=1)
    print(f"Test Loss: {score[0]:.4f}")
    print(f"Test Accuracy: {score[1]:.4f}")

    # 6. Optionally, make predictions (if required)
    # predictions = model.predict(X_test)
    # print("Sample Predictions:", predictions[:5])
