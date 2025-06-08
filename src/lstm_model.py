"""
LSTM Model Architecture Module for Gold Price Prediction
Author: ridwaanhall
Date: 2025-06-08
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, TimeDistributed,
    BatchNormalization, Attention, MultiHeadAttention,
    Input, Conv1D, MaxPooling1D, Flatten, GlobalMaxPooling1D,
    Add, Concatenate, LayerNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1, l2, l1_l2
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

from .utils import setup_logging, get_model_summary_dict


class AttentionLayer(layers.Layer):
    """
    Custom attention layer for LSTM model.
    """
    
    def __init__(self, attention_dim: int = 64, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.attention_dim),
            initializer='uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.attention_dim,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.attention_dim,),
            initializer='uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Compute attention scores
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.nn.softmax(ait, axis=1)
        
        # Apply attention weights
        ait = tf.expand_dims(ait, axis=-1)
        weighted_input = inputs * ait
        output = tf.reduce_sum(weighted_input, axis=1)
        
        return output
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'attention_dim': self.attention_dim})
        return config


class LSTMGoldPredictor:
    """
    Main LSTM model class for gold price prediction.
    
    This class provides various LSTM architectures and configurations
    for time series forecasting of gold prices.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LSTM Gold Predictor.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or {}
        self.logger = setup_logging()
        
        # Model configuration
        self.sequence_length = self.config.get('sequence_length', 60)
        self.n_features = self.config.get('n_features', 1)
        self.lstm_units = self.config.get('lstm_units', [50, 50, 50])
        self.dropout_rate = self.config.get('dropout_rate', 0.2)
        self.recurrent_dropout = self.config.get('recurrent_dropout', 0.2)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.bidirectional = self.config.get('bidirectional', False)
        self.attention = self.config.get('attention', False)
        
        # Model storage
        self.model = None
        self.model_history = None
        
        self.logger.info("LSTMGoldPredictor initialized")
    
    def build_model(self, model_type: str = 'stacked_lstm') -> keras.Model:
        """
        Build LSTM model with specified architecture.
        
        Args:
            model_type: Type of model architecture to build
        
        Returns:
            Compiled Keras model
        """
        self.logger.info(f"Building {model_type} model")
        
        if model_type == 'simple_lstm':
            model = self._build_simple_lstm()
        elif model_type == 'stacked_lstm':
            model = self._build_stacked_lstm()
        elif model_type == 'bidirectional_lstm':
            model = self._build_bidirectional_lstm()
        elif model_type == 'attention_lstm':
            model = self._build_attention_lstm()
        elif model_type == 'cnn_lstm':
            model = self._build_cnn_lstm()
        elif model_type == 'encoder_decoder':
            model = self._build_encoder_decoder()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = model
        self.logger.info(f"Model built with {model.count_params()} parameters")
        
        return model
    
    def _build_simple_lstm(self) -> keras.Model:
        """Build simple LSTM model."""
        model = Sequential([
            Input(shape=(self.sequence_length, self.n_features)),
            LSTM(self.lstm_units[0], 
                 dropout=self.dropout_rate,
                 recurrent_dropout=self.recurrent_dropout,
                 return_sequences=False),
            Dense(32, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        return model
    
    def _build_stacked_lstm(self) -> keras.Model:
        """Build stacked LSTM model."""
        model = Sequential()
        model.add(Input(shape=(self.sequence_length, self.n_features)))
        
        # Add LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            
            model.add(LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout,
                name=f'lstm_{i+1}'
            ))
            
            if return_sequences:
                model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1, activation='linear'))
        
        return model
    
    def _build_bidirectional_lstm(self) -> keras.Model:
        """Build bidirectional LSTM model."""
        model = Sequential()
        model.add(Input(shape=(self.sequence_length, self.n_features)))
        
        # Add bidirectional LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            
            model.add(Bidirectional(
                LSTM(units,
                     return_sequences=return_sequences,
                     dropout=self.dropout_rate,
                     recurrent_dropout=self.recurrent_dropout),
                name=f'bidirectional_lstm_{i+1}'
            ))
            
            if return_sequences:
                model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        
        return model
    
    def _build_attention_lstm(self) -> keras.Model:
        """Build LSTM model with attention mechanism."""
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # LSTM layers
        x = inputs
        for i, units in enumerate(self.lstm_units):
            x = LSTM(units,
                    return_sequences=True,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout,
                    name=f'lstm_{i+1}')(x)
            x = BatchNormalization()(x)
        
        # Attention layer
        attention_output = AttentionLayer(attention_dim=64)(x)
        
        # Dense layers
        x = Dense(128, activation='relu')(attention_output)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def _build_cnn_lstm(self) -> keras.Model:
        """Build CNN-LSTM hybrid model."""
        model = Sequential([
            Input(shape=(self.sequence_length, self.n_features)),
            
            # CNN layers for feature extraction
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            
            # LSTM layers
            LSTM(self.lstm_units[0], 
                 return_sequences=True,
                 dropout=self.dropout_rate,
                 recurrent_dropout=self.recurrent_dropout),
            LSTM(self.lstm_units[1] if len(self.lstm_units) > 1 else 50,
                 dropout=self.dropout_rate,
                 recurrent_dropout=self.recurrent_dropout),
            
            # Dense layers
            Dense(64, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        return model
    
    def _build_encoder_decoder(self) -> keras.Model:
        """Build encoder-decoder LSTM model."""
        # Encoder
        encoder_inputs = Input(shape=(self.sequence_length, self.n_features))
        encoder_lstm = LSTM(self.lstm_units[0], 
                           return_state=True,
                           dropout=self.dropout_rate,
                           recurrent_dropout=self.recurrent_dropout)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = Input(shape=(1, self.n_features))
        decoder_lstm = LSTM(self.lstm_units[0],
                           return_sequences=True,
                           return_state=True,
                           dropout=self.dropout_rate,
                           recurrent_dropout=self.recurrent_dropout)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        
        decoder_dense = Dense(1, activation='linear')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        return model
    
    def compile_model(self, 
                     optimizer: str = 'adam',
                     loss: str = 'mse',
                     metrics: Optional[List[str]] = None) -> None:
        """
        Compile the model with specified optimizer and loss function.
        
        Args:
            optimizer: Optimizer to use for training
            loss: Loss function to optimize
            metrics: List of metrics to track during training
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if metrics is None:
            metrics = ['mae', 'mse']
        
        # Setup optimizer
        if optimizer == 'adam':
            opt = optimizers.Adam(learning_rate=self.learning_rate)
        elif optimizer == 'rmsprop':
            opt = optimizers.RMSprop(learning_rate=self.learning_rate)
        elif optimizer == 'sgd':
            opt = optimizers.SGD(learning_rate=self.learning_rate)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
        
        self.logger.info(f"Model compiled with optimizer: {optimizer}, loss: {loss}")
    
    def get_model_summary(self) -> str:
        """
        Get model architecture summary.
        
        Returns:
            String representation of model summary
        """
        if self.model is None:
            return "Model not built yet."
        
        # Capture model summary
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines)
    
    def save_architecture(self, filepath: str) -> None:
        """
        Save model architecture to file.
        
        Args:
            filepath: Path to save the architecture
        """
        if self.model is None:
            raise ValueError("Model not built yet.")
        
        # Save model architecture as JSON
        architecture = self.model.to_json()
        
        with open(filepath, 'w') as f:
            f.write(architecture)
        
        self.logger.info(f"Model architecture saved to {filepath}")
    
    def load_architecture(self, filepath: str) -> None:
        """
        Load model architecture from file.
        
        Args:
            filepath: Path to architecture file
        """
        with open(filepath, 'r') as f:
            architecture = f.read()
        
        self.model = keras.models.model_from_json(architecture, 
                                                 custom_objects={'AttentionLayer': AttentionLayer})
        
        self.logger.info(f"Model architecture loaded from {filepath}")
    
    def get_layer_weights(self, layer_name: str) -> np.ndarray:
        """
        Get weights of a specific layer.
        
        Args:
            layer_name: Name of the layer
        
        Returns:
            Layer weights as numpy array
        """
        if self.model is None:
            raise ValueError("Model not built yet.")
        
        try:
            layer = self.model.get_layer(layer_name)
            return layer.get_weights()
        except ValueError:
            available_layers = [layer.name for layer in self.model.layers]
            raise ValueError(f"Layer '{layer_name}' not found. Available layers: {available_layers}")
    
    def visualize_model(self, output_path: str = 'model_architecture.png') -> str:
        """
        Create visual representation of model architecture.
        
        Args:
            output_path: Path to save the visualization
        
        Returns:
            Path to saved visualization
        """
        if self.model is None:
            raise ValueError("Model not built yet.")
        
        try:
            keras.utils.plot_model(
                self.model,
                to_file=output_path,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                expand_nested=True,
                dpi=300
            )
            self.logger.info(f"Model visualization saved to {output_path}")
            return output_path
        except Exception as e:
            self.logger.warning(f"Could not create model visualization: {str(e)}")
            return None


class ModelBuilder:
    """
    Factory class for creating different LSTM model architectures.
    """
    
    @staticmethod
    def create_simple_lstm(sequence_length: int, 
                          n_features: int,
                          lstm_units: int = 50,
                          dropout_rate: float = 0.2) -> keras.Model:
        """
        Create simple LSTM model.
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of features
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate
        
        Returns:
            Simple LSTM model
        """
        model = Sequential([
            Input(shape=(sequence_length, n_features)),
            LSTM(lstm_units, dropout=dropout_rate),
            Dense(1, activation='linear')
        ])
        
        return model
    
    @staticmethod
    def create_bidirectional_lstm(sequence_length: int,
                                 n_features: int,
                                 lstm_units: List[int] = [50, 50],
                                 dropout_rate: float = 0.2) -> keras.Model:
        """
        Create bidirectional LSTM model.
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of features
            lstm_units: List of LSTM units for each layer
            dropout_rate: Dropout rate
        
        Returns:
            Bidirectional LSTM model
        """
        model = Sequential()
        model.add(Input(shape=(sequence_length, n_features)))
        
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            model.add(Bidirectional(
                LSTM(units, return_sequences=return_sequences, dropout=dropout_rate)
            ))
        
        model.add(Dense(1, activation='linear'))
        
        return model
    
    @staticmethod
    def create_stacked_lstm(sequence_length: int,
                           n_features: int,
                           lstm_units: List[int] = [64, 32, 16],
                           dropout_rate: float = 0.2,
                           dense_units: List[int] = [32, 16]) -> keras.Model:
        """
        Create stacked LSTM model.
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of features
            lstm_units: List of LSTM units for each layer
            dropout_rate: Dropout rate
            dense_units: List of dense layer units
        
        Returns:
            Stacked LSTM model
        """
        model = Sequential()
        model.add(Input(shape=(sequence_length, n_features)))
        
        # LSTM layers
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            model.add(LSTM(units, return_sequences=return_sequences, dropout=dropout_rate))
            
            if return_sequences:
                model.add(BatchNormalization())
        
        # Dense layers
        for units in dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout_rate))
        
        model.add(Dense(1, activation='linear'))
        
        return model
    
    @staticmethod
    def add_attention_layer(model: keras.Model, attention_dim: int = 64) -> keras.Model:
        """
        Add attention mechanism to existing model.
        
        Args:
            model: Existing model
            attention_dim: Dimension of attention layer
        
        Returns:
            Model with attention layer added
        """
        # This is a simplified example - in practice, you'd need to modify the architecture
        # to properly integrate attention
        inputs = model.input
        
        # Get the output of the last LSTM layer (assuming it returns sequences)
        lstm_output = None
        for layer in model.layers:
            if isinstance(layer, (LSTM, Bidirectional)):
                lstm_output = layer.output
        
        if lstm_output is not None:
            attention_output = AttentionLayer(attention_dim)(lstm_output)
            final_output = Dense(1, activation='linear')(attention_output)
            
            return Model(inputs=inputs, outputs=final_output)
        else:
            return model


def create_custom_loss_functions():
    """
    Create custom loss functions for gold price prediction.
    
    Returns:
        Dictionary of custom loss functions
    """
    
    def directional_loss(y_true, y_pred, alpha=0.5):
        """
        Custom loss that penalizes wrong directional predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            alpha: Weight for directional component
        
        Returns:
            Combined MSE and directional loss
        """
        # MSE component
        mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        
        # Directional component
        true_direction = tf.sign(y_true[1:] - y_true[:-1])
        pred_direction = tf.sign(y_pred[1:] - y_pred[:-1])
        
        directional_accuracy = tf.cast(
            tf.equal(true_direction, pred_direction), tf.float32
        )
        directional_loss_val = 1.0 - tf.reduce_mean(directional_accuracy)
        
        return (1 - alpha) * mse_loss + alpha * directional_loss_val
    
    def huber_loss_custom(delta=1.0):
        """
        Create custom Huber loss function.
        
        Args:
            delta: Threshold for switching between MSE and MAE
        
        Returns:
            Huber loss function
        """
        def huber_fn(y_true, y_pred):
            error = y_true - y_pred
            is_small_error = tf.abs(error) <= delta
            squared_loss = tf.square(error) / 2
            linear_loss = delta * tf.abs(error) - tf.square(delta) / 2
            return tf.where(is_small_error, squared_loss, linear_loss)
        
        return huber_fn
    
    def quantile_loss(quantile=0.5):
        """
        Create quantile loss function for uncertainty estimation.
        
        Args:
            quantile: Quantile to optimize (0.5 for median)
        
        Returns:
            Quantile loss function
        """
        def quantile_fn(y_true, y_pred):
            error = y_true - y_pred
            return tf.reduce_mean(tf.maximum(quantile * error, (quantile - 1) * error))
        
        return quantile_fn
    
    return {
        'directional_loss': directional_loss,
        'huber_loss': huber_loss_custom(),
        'quantile_loss_median': quantile_loss(0.5),
        'quantile_loss_upper': quantile_loss(0.9),
        'quantile_loss_lower': quantile_loss(0.1)
    }


def create_custom_metrics():
    """
    Create custom metrics for model evaluation.
    
    Returns:
        Dictionary of custom metrics
    """
    
    def directional_accuracy(y_true, y_pred):
        """
        Calculate directional accuracy metric.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            Directional accuracy
        """
        true_direction = tf.sign(y_true[1:] - y_true[:-1])
        pred_direction = tf.sign(y_pred[1:] - y_pred[:-1])
        
        accuracy = tf.cast(
            tf.equal(true_direction, pred_direction), tf.float32
        )
        
        return tf.reduce_mean(accuracy)
    
    def mape_metric(y_true, y_pred):
        """
        Mean Absolute Percentage Error metric.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            MAPE value
        """
        return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true)) * 100
    
    def rmse_metric(y_true, y_pred):
        """
        Root Mean Square Error metric.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            RMSE value
        """
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
    
    return {
        'directional_accuracy': directional_accuracy,
        'mape': mape_metric,
        'rmse': rmse_metric
    }
