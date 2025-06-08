"""
Test utilities and common fixtures for the gold price prediction system.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, Any, List
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.config import (
    DataConfig, ModelConfig, TrainingConfig, 
    EvaluationConfig, PredictionConfig, VisualizationConfig
)


class TestDataGenerator:
    """Generate test data for various test scenarios."""
    
    @staticmethod
    def create_sample_gold_data(
        n_samples: int = 100,
        start_date: str = "2024-01-01",
        price_range: tuple = (700000, 800000),
        add_noise: bool = True
    ) -> pd.DataFrame:
        """Create sample gold price data for testing."""
        dates = pd.date_range(start=start_date, periods=n_samples, freq='D')
        
        # Generate realistic price patterns
        base_price = np.mean(price_range)
        trend = np.linspace(0, 50000, n_samples)  # Upward trend
        seasonal = 10000 * np.sin(2 * np.pi * np.arange(n_samples) / 30)  # Monthly cycle
        
        if add_noise:
            noise = np.random.normal(0, 5000, n_samples)
        else:
            noise = np.zeros(n_samples)
        
        harga_jual = base_price + trend + seasonal + noise
        harga_beli = harga_jual - np.random.uniform(5000, 15000, n_samples)
        
        data = []
        for i, date in enumerate(dates):
            data.append({
                "tanggal": date.strftime("%Y-%m-%d"),
                "hargaJual": float(harga_jual[i]),
                "hargaBeli": float(harga_beli[i])
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_preprocessed_data(
        n_samples: int = 100,
        n_features: int = 10,
        sequence_length: int = 30
    ) -> tuple:
        """Create preprocessed data for model testing."""
        # Create sample sequences
        X = np.random.randn(n_samples, sequence_length, n_features)
        y = np.random.randn(n_samples, 1)
        
        return X, y
    
    @staticmethod
    def create_trained_model(input_shape: tuple = (30, 10)) -> tf.keras.Model:
        """Create a simple trained model for testing."""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, input_shape=input_shape),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Add some dummy training
        X_dummy = np.random.randn(10, *input_shape)
        y_dummy = np.random.randn(10, 1)
        model.fit(X_dummy, y_dummy, epochs=1, verbose=0)
        
        return model


class TestConfigFactory:
    """Factory for creating test configurations."""
    
    @staticmethod
    def create_data_config(**kwargs) -> DataConfig:
        """Create test data configuration."""
        defaults = {
            'data_path': 'test_data.json',
            'target_column': 'hargaJual',
            'date_column': 'tanggal',
            'sequence_length': 30,
            'prediction_horizon': 1,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'scaling_method': 'minmax',
            'handle_missing': 'interpolate',
            'outlier_method': 'iqr',
            'feature_engineering': True
        }
        defaults.update(kwargs)
        return DataConfig(**defaults)
    
    @staticmethod
    def create_model_config(**kwargs) -> ModelConfig:
        """Create test model configuration."""
        defaults = {
            'model_type': 'simple_lstm',
            'lstm_units': [50],
            'dense_units': [25],
            'dropout_rate': 0.2,
            'recurrent_dropout': 0.1,
            'activation': 'relu',
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'loss_function': 'mse',
            'metrics': ['mae', 'mape'],
            'use_attention': False,
            'bidirectional': False,
            'batch_normalization': False
        }
        defaults.update(kwargs)
        return ModelConfig(**defaults)
    
    @staticmethod
    def create_training_config(**kwargs) -> TrainingConfig:
        """Create test training configuration."""
        defaults = {
            'epochs': 5,
            'batch_size': 32,
            'validation_split': 0.2,
            'early_stopping': True,
            'patience': 3,
            'reduce_lr': True,
            'lr_patience': 2,
            'lr_factor': 0.5,
            'min_lr': 1e-6,
            'save_best_only': True,
            'verbose': 0
        }
        defaults.update(kwargs)
        return TrainingConfig(**defaults)


@pytest.fixture
def sample_gold_data():
    """Fixture providing sample gold price data."""
    return TestDataGenerator.create_sample_gold_data(n_samples=50)


@pytest.fixture
def preprocessed_data():
    """Fixture providing preprocessed data for model testing."""
    return TestDataGenerator.create_preprocessed_data(n_samples=50)


@pytest.fixture
def test_configs():
    """Fixture providing test configurations."""
    return {
        'data': TestConfigFactory.create_data_config(),
        'model': TestConfigFactory.create_model_config(),
        'training': TestConfigFactory.create_training_config()
    }


@pytest.fixture
def trained_model():
    """Fixture providing a trained model."""
    return TestDataGenerator.create_trained_model()


@pytest.fixture
def temp_dir(tmp_path):
    """Fixture providing temporary directory."""
    return str(tmp_path)


class MockFileSystem:
    """Mock file system operations for testing."""
    
    def __init__(self):
        self.files = {}
    
    def write_file(self, path: str, content: str):
        """Mock file writing."""
        self.files[path] = content
    
    def read_file(self, path: str) -> str:
        """Mock file reading."""
        return self.files.get(path, "")
    
    def file_exists(self, path: str) -> bool:
        """Mock file existence check."""
        return path in self.files


def assert_valid_dataframe(df: pd.DataFrame, required_columns: List[str] = None):
    """Assert that dataframe is valid."""
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    
    if required_columns:
        for col in required_columns:
            assert col in df.columns, f"Column {col} not found in dataframe"


def assert_valid_model(model):
    """Assert that model is valid."""
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'fit')


def assert_valid_predictions(predictions: np.ndarray, expected_shape: tuple = None):
    """Assert that predictions are valid."""
    assert isinstance(predictions, np.ndarray)
    assert not np.isnan(predictions).any()
    assert not np.isinf(predictions).any()
    
    if expected_shape:
        assert predictions.shape == expected_shape


def assert_metrics_in_range(metrics: Dict[str, float], metric_ranges: Dict[str, tuple]):
    """Assert that metrics are within expected ranges."""
    for metric_name, (min_val, max_val) in metric_ranges.items():
        assert metric_name in metrics, f"Metric {metric_name} not found"
        value = metrics[metric_name]
        assert min_val <= value <= max_val, f"Metric {metric_name} = {value} not in range [{min_val}, {max_val}]"


# Test configuration constants
TEST_DATA_SIZE = 100
TEST_SEQUENCE_LENGTH = 30
TEST_N_FEATURES = 10
TEST_BATCH_SIZE = 16
TEST_EPOCHS = 2

# Performance thresholds for testing
PERFORMANCE_THRESHOLDS = {
    'mape': (0, 10),  # MAPE should be between 0-10%
    'mae': (0, 50000),  # MAE should be reasonable for gold prices
    'rmse': (0, 100000),  # RMSE should be reasonable
    'r2': (0, 1),  # R² should be between 0-1
    'direction_accuracy': (0.4, 1.0)  # At least 40% direction accuracy
}
