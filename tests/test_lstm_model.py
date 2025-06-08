"""
Unit tests for LSTM model module.
"""

import pytest
import numpy as np
import tensorflow as tf
import os
import sys
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lstm_model import LSTMGoldPredictor, AttentionLayer
from test_utils import (
    TestConfigFactory, TestDataGenerator,
    assert_valid_model, preprocessed_data, test_configs
)


class TestAttentionLayer:
    """Test cases for custom Attention layer."""
    
    def test_init(self):
        """Test attention layer initialization."""
        attention = AttentionLayer()
        
        assert attention.W is None
        assert attention.b is None
    
    def test_build(self):
        """Test attention layer build method."""
        attention = AttentionLayer()
        
        # Simulate build call
        input_shape = (None, 30, 50)  # (batch_size, sequence_length, features)
        attention.build(input_shape)
        
        assert attention.W is not None
        assert attention.b is not None
        assert attention.W.shape == (50, 1)
        assert attention.b.shape == (1,)
    
    def test_call(self):
        """Test attention layer forward pass."""
        attention = AttentionLayer()
        
        # Create dummy input
        batch_size, seq_len, features = 2, 10, 20
        inputs = tf.random.normal((batch_size, seq_len, features))
        
        # Build layer
        attention.build(inputs.shape)
        
        # Forward pass
        output = attention(inputs)
        
        assert output.shape == (batch_size, features)
        assert not tf.reduce_any(tf.math.is_nan(output))
    
    def test_get_config(self):
        """Test attention layer configuration."""
        attention = AttentionLayer()
        config = attention.get_config()
        
        assert isinstance(config, dict)
        assert 'name' in config


class TestLSTMGoldPredictor:
    """Test cases for LSTMGoldPredictor class."""
    
    def test_init(self, test_configs):
        """Test model initialization."""
        predictor = LSTMGoldPredictor(test_configs['model'])
        
        assert predictor.config == test_configs['model']
        assert predictor.model is None
        assert predictor.history is None
    
    def test_build_simple_lstm(self, test_configs):
        """Test building simple LSTM model."""
        config = test_configs['model']
        config.model_type = 'simple_lstm'
        config.lstm_units = [50]
        config.dense_units = [25]
        
        predictor = LSTMGoldPredictor(config)
        model = predictor.build_model(input_shape=(30, 10))
        
        assert_valid_model(model)
        assert isinstance(model, tf.keras.Model)
        
        # Check model structure
        assert len(model.layers) >= 3  # LSTM + Dense + Output layers
        assert any('lstm' in layer.name.lower() for layer in model.layers)
    
    def test_build_stacked_lstm(self, test_configs):
        """Test building stacked LSTM model."""
        config = test_configs['model']
        config.model_type = 'stacked_lstm'
        config.lstm_units = [50, 30]
        
        predictor = LSTMGoldPredictor(config)
        model = predictor.build_model(input_shape=(30, 10))
        
        assert_valid_model(model)
        
        # Should have multiple LSTM layers
        lstm_layers = [layer for layer in model.layers if 'lstm' in layer.name.lower()]
        assert len(lstm_layers) >= 2
    
    def test_build_bidirectional_lstm(self, test_configs):
        """Test building bidirectional LSTM model."""
        config = test_configs['model']
        config.model_type = 'simple_lstm'
        config.bidirectional = True
        config.lstm_units = [50]
        
        predictor = LSTMGoldPredictor(config)
        model = predictor.build_model(input_shape=(30, 10))
        
        assert_valid_model(model)
        
        # Check for bidirectional wrapper
        assert any('bidirectional' in layer.name.lower() for layer in model.layers)
    
    def test_build_attention_lstm(self, test_configs):
        """Test building LSTM with attention."""
        config = test_configs['model']
        config.model_type = 'attention_lstm'
        config.use_attention = True
        config.lstm_units = [50]
        
        predictor = LSTMGoldPredictor(config)
        model = predictor.build_model(input_shape=(30, 10))
        
        assert_valid_model(model)
        
        # Check for attention layer
        assert any('attention' in layer.name.lower() for layer in model.layers)
    
    def test_build_cnn_lstm(self, test_configs):
        """Test building CNN-LSTM model."""
        config = test_configs['model']
        config.model_type = 'cnn_lstm'
        config.lstm_units = [50]
        
        predictor = LSTMGoldPredictor(config)
        model = predictor.build_model(input_shape=(30, 10))
        
        assert_valid_model(model)
        
        # Should have both Conv1D and LSTM layers
        has_conv = any('conv' in layer.name.lower() for layer in model.layers)
        has_lstm = any('lstm' in layer.name.lower() for layer in model.layers)
        assert has_conv and has_lstm
    
    def test_build_encoder_decoder(self, test_configs):
        """Test building encoder-decoder model."""
        config = test_configs['model']
        config.model_type = 'encoder_decoder'
        config.lstm_units = [50]
        
        predictor = LSTMGoldPredictor(config)
        model = predictor.build_model(input_shape=(30, 10))
        
        assert_valid_model(model)
        
        # Should have multiple LSTM layers for encoder-decoder structure
        lstm_layers = [layer for layer in model.layers if 'lstm' in layer.name.lower()]
        assert len(lstm_layers) >= 2
    
    def test_compile_model(self, test_configs):
        """Test model compilation."""
        config = test_configs['model']
        config.optimizer = 'adam'
        config.learning_rate = 0.001
        config.loss_function = 'mse'
        config.metrics = ['mae']
        
        predictor = LSTMGoldPredictor(config)
        model = predictor.build_model(input_shape=(30, 10))
        
        # Model should be compiled during build
        assert model.optimizer is not None
        assert model.compiled_loss is not None
    
    def test_custom_loss_functions(self, test_configs):
        """Test custom loss functions."""
        config = test_configs['model']
        predictor = LSTMGoldPredictor(config)
        
        # Test directional loss
        y_true = tf.constant([[1.0], [2.0], [1.5]])
        y_pred = tf.constant([[1.1], [1.8], [1.6]])
        
        directional_loss = predictor.directional_loss(y_true, y_pred)
        assert not tf.math.is_nan(directional_loss)
        assert directional_loss >= 0
        
        # Test asymmetric loss
        asymmetric_loss = predictor.asymmetric_loss(y_true, y_pred)
        assert not tf.math.is_nan(asymmetric_loss)
        assert asymmetric_loss >= 0
    
    def test_custom_metrics(self, test_configs):
        """Test custom metrics."""
        config = test_configs['model']
        predictor = LSTMGoldPredictor(config)
        
        y_true = tf.constant([[1.0], [2.0], [1.5]])
        y_pred = tf.constant([[1.1], [1.8], [1.6]])
        
        # Test MAPE
        mape = predictor.mape_metric(y_true, y_pred)
        assert not tf.math.is_nan(mape)
        assert mape >= 0
        
        # Test directional accuracy
        dir_acc = predictor.directional_accuracy(y_true, y_pred)
        assert not tf.math.is_nan(dir_acc)
        assert 0 <= dir_acc <= 1
    
    def test_fit_model(self, preprocessed_data, test_configs):
        """Test model training."""
        X, y = preprocessed_data
        
        config = test_configs['model']
        predictor = LSTMGoldPredictor(config)
        
        # Build model
        model = predictor.build_model(input_shape=X.shape[1:])
        predictor.model = model
        
        # Fit model
        history = predictor.fit(
            X, y,
            validation_split=0.2,
            epochs=2,
            batch_size=16,
            verbose=0
        )
        
        assert history is not None
        assert 'loss' in history.history
        assert predictor.history is not None
    
    def test_predict(self, preprocessed_data, test_configs):
        """Test model prediction."""
        X, y = preprocessed_data
        
        config = test_configs['model']
        predictor = LSTMGoldPredictor(config)
        
        # Build and train model
        model = predictor.build_model(input_shape=X.shape[1:])
        predictor.model = model
        model.fit(X, y, epochs=1, verbose=0)
        
        # Make predictions
        predictions = predictor.predict(X[:10])
        
        assert predictions is not None
        assert len(predictions) == 10
        assert not np.isnan(predictions).any()
    
    def test_save_and_load_model(self, preprocessed_data, test_configs, temp_dir):
        """Test saving and loading model."""
        X, y = preprocessed_data
        
        config = test_configs['model']
        predictor = LSTMGoldPredictor(config)
        
        # Build and train model
        model = predictor.build_model(input_shape=X.shape[1:])
        predictor.model = model
        model.fit(X, y, epochs=1, verbose=0)
        
        # Save model
        model_path = os.path.join(temp_dir, 'test_model.h5')
        predictor.save_model(model_path)
        
        assert os.path.exists(model_path)
        
        # Load model
        new_predictor = LSTMGoldPredictor(config)
        new_predictor.load_model(model_path)
        
        assert new_predictor.model is not None
        
        # Test that loaded model can make predictions
        predictions = new_predictor.predict(X[:5])
        assert predictions is not None
        assert len(predictions) == 5
    
    def test_model_summary(self, test_configs):
        """Test model summary generation."""
        config = test_configs['model']
        predictor = LSTMGoldPredictor(config)
        
        model = predictor.build_model(input_shape=(30, 10))
        
        # Should be able to generate summary without errors
        try:
            summary = model.summary()
            # If we get here, no exception was raised
            assert True
        except Exception as e:
            pytest.fail(f"Model summary failed: {str(e)}")
    
    def test_different_optimizers(self, test_configs):
        """Test different optimizers."""
        optimizers = ['adam', 'sgd', 'rmsprop']
        
        for opt in optimizers:
            config = test_configs['model']
            config.optimizer = opt
            config.learning_rate = 0.001
            
            predictor = LSTMGoldPredictor(config)
            model = predictor.build_model(input_shape=(30, 10))
            
            assert model.optimizer is not None
            assert opt.lower() in model.optimizer.__class__.__name__.lower()
    
    def test_different_loss_functions(self, test_configs):
        """Test different loss functions."""
        loss_functions = ['mse', 'mae', 'huber', 'directional', 'asymmetric']
        
        for loss in loss_functions:
            config = test_configs['model']
            config.loss_function = loss
            
            predictor = LSTMGoldPredictor(config)
            model = predictor.build_model(input_shape=(30, 10))
            
            assert model.compiled_loss is not None
    
    def test_batch_normalization(self, test_configs):
        """Test model with batch normalization."""
        config = test_configs['model']
        config.batch_normalization = True
        config.lstm_units = [50]
        config.dense_units = [25]
        
        predictor = LSTMGoldPredictor(config)
        model = predictor.build_model(input_shape=(30, 10))
        
        assert_valid_model(model)
        
        # Check for batch normalization layers
        has_bn = any('batch_normalization' in layer.name.lower() for layer in model.layers)
        assert has_bn
    
    def test_dropout_layers(self, test_configs):
        """Test model with dropout."""
        config = test_configs['model']
        config.dropout_rate = 0.3
        config.recurrent_dropout = 0.2
        config.lstm_units = [50]
        config.dense_units = [25]
        
        predictor = LSTMGoldPredictor(config)
        model = predictor.build_model(input_shape=(30, 10))
        
        assert_valid_model(model)
        
        # Check for dropout layers
        has_dropout = any('dropout' in layer.name.lower() for layer in model.layers)
        assert has_dropout
    
    @pytest.mark.parametrize("model_type", [
        'simple_lstm', 'stacked_lstm', 'bidirectional_lstm', 
        'attention_lstm', 'cnn_lstm', 'encoder_decoder'
    ])
    def test_all_model_types(self, test_configs, model_type):
        """Test all model types can be built successfully."""
        config = test_configs['model']
        config.model_type = model_type
        if model_type == 'attention_lstm':
            config.use_attention = True
        if model_type == 'bidirectional_lstm':
            config.bidirectional = True
        
        predictor = LSTMGoldPredictor(config)
        model = predictor.build_model(input_shape=(30, 10))
        
        assert_valid_model(model)
        assert model.input_shape[1:] == (30, 10)
        assert model.output_shape[1:] == (1,)  # Single output for regression
    
    def test_model_weights_initialization(self, test_configs):
        """Test that model weights are properly initialized."""
        config = test_configs['model']
        predictor = LSTMGoldPredictor(config)
        
        # Build two identical models
        model1 = predictor.build_model(input_shape=(30, 10))
        model2 = predictor.build_model(input_shape=(30, 10))
        
        # Check that weights are different (random initialization)
        weights1 = model1.get_weights()
        weights2 = model2.get_weights()
        
        # At least one weight matrix should be different
        weights_different = False
        for w1, w2 in zip(weights1, weights2):
            if not np.allclose(w1, w2):
                weights_different = True
                break
        
        assert weights_different, "Model weights should be randomly initialized"
    
    def test_prediction_consistency(self, preprocessed_data, test_configs):
        """Test that predictions are consistent for same input."""
        X, y = preprocessed_data
        
        config = test_configs['model']
        predictor = LSTMGoldPredictor(config)
        
        # Build and train model
        model = predictor.build_model(input_shape=X.shape[1:])
        predictor.model = model
        model.fit(X, y, epochs=1, verbose=0)
        
        # Make predictions twice
        predictions1 = predictor.predict(X[:5])
        predictions2 = predictor.predict(X[:5])
        
        # Should be identical for same input
        np.testing.assert_array_almost_equal(predictions1, predictions2)
