"""
Unit tests for data preprocessing module.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import GoldDataPreprocessor
from test_utils import (
    TestDataGenerator, TestConfigFactory, 
    assert_valid_dataframe, sample_gold_data, test_configs
)


class TestGoldDataPreprocessor:
    """Test cases for GoldDataPreprocessor class."""
    
    def test_init(self, test_configs):
        """Test preprocessor initialization."""
        preprocessor = GoldDataPreprocessor(test_configs['data'])
        
        assert preprocessor.config == test_configs['data']
        assert preprocessor.scaler is None
        assert preprocessor.feature_scaler is None
        assert preprocessor.data is None
    
    def test_load_data_json(self, sample_gold_data, temp_dir, test_configs):
        """Test loading data from JSON file."""
        # Save sample data to JSON
        json_file = os.path.join(temp_dir, 'test_gold_data.json')
        sample_gold_data.to_json(json_file, orient='records', date_format='iso')
        
        # Update config with test file path
        config = test_configs['data']
        config.data_path = json_file
        
        preprocessor = GoldDataPreprocessor(config)
        df = preprocessor.load_data()
        
        assert_valid_dataframe(df, ['tanggal', 'hargaJual', 'hargaBeli'])
        assert len(df) == len(sample_gold_data)
        assert 'tanggal' in df.columns
    
    def test_load_data_csv(self, sample_gold_data, temp_dir, test_configs):
        """Test loading data from CSV file."""
        # Save sample data to CSV
        csv_file = os.path.join(temp_dir, 'test_gold_data.csv')
        sample_gold_data.to_csv(csv_file, index=False)
        
        # Update config
        config = test_configs['data']
        config.data_path = csv_file
        
        preprocessor = GoldDataPreprocessor(config)
        df = preprocessor.load_data()
        
        assert_valid_dataframe(df, ['tanggal', 'hargaJual', 'hargaBeli'])
        assert len(df) == len(sample_gold_data)
    
    def test_load_data_invalid_format(self, temp_dir, test_configs):
        """Test loading data with invalid format."""
        # Create invalid file
        invalid_file = os.path.join(temp_dir, 'invalid.txt')
        with open(invalid_file, 'w') as f:
            f.write("invalid data")
        
        config = test_configs['data']
        config.data_path = invalid_file
        
        preprocessor = GoldDataPreprocessor(config)
        
        with pytest.raises(ValueError):
            preprocessor.load_data()
    
    def test_load_data_missing_file(self, test_configs):
        """Test loading data from non-existent file."""
        config = test_configs['data']
        config.data_path = 'non_existent_file.json'
        
        preprocessor = GoldDataPreprocessor(config)
        
        with pytest.raises(FileNotFoundError):
            preprocessor.load_data()
    
    def test_clean_data(self, sample_gold_data, test_configs):
        """Test data cleaning functionality."""
        # Add some issues to test data
        dirty_data = sample_gold_data.copy()
        dirty_data.loc[5, 'hargaJual'] = None  # Missing value
        dirty_data.loc[10, 'hargaBeli'] = -1000  # Negative value
        dirty_data = pd.concat([dirty_data, dirty_data.iloc[0:1]], ignore_index=True)  # Duplicate
        
        preprocessor = GoldDataPreprocessor(test_configs['data'])
        preprocessor.data = dirty_data
        
        cleaned_data = preprocessor.clean_data()
        
        assert_valid_dataframe(cleaned_data)
        assert cleaned_data['hargaJual'].notna().all()
        assert cleaned_data['hargaBeli'].notna().all()
        assert (cleaned_data['hargaJual'] > 0).all()
        assert (cleaned_data['hargaBeli'] > 0).all()
        assert len(cleaned_data) <= len(dirty_data)  # Should remove duplicates
    
    def test_handle_missing_values_interpolate(self, sample_gold_data, test_configs):
        """Test missing value handling with interpolation."""
        # Add missing values
        data_with_missing = sample_gold_data.copy()
        data_with_missing.loc[5:7, 'hargaJual'] = None
        
        config = test_configs['data']
        config.handle_missing = 'interpolate'
        
        preprocessor = GoldDataPreprocessor(config)
        preprocessor.data = data_with_missing
        
        result = preprocessor.handle_missing_values()
        
        assert result['hargaJual'].notna().all()
        # Check that interpolated values are reasonable
        assert result.loc[6, 'hargaJual'] > 0
    
    def test_handle_missing_values_forward_fill(self, sample_gold_data, test_configs):
        """Test missing value handling with forward fill."""
        data_with_missing = sample_gold_data.copy()
        data_with_missing.loc[5, 'hargaJual'] = None
        
        config = test_configs['data']
        config.handle_missing = 'forward_fill'
        
        preprocessor = GoldDataPreprocessor(config)
        preprocessor.data = data_with_missing
        
        result = preprocessor.handle_missing_values()
        
        assert result['hargaJual'].notna().all()
        # Forward fill should use previous value
        assert result.loc[5, 'hargaJual'] == result.loc[4, 'hargaJual']
    
    def test_detect_outliers_iqr(self, sample_gold_data, test_configs):
        """Test outlier detection using IQR method."""
        # Add extreme outliers
        data_with_outliers = sample_gold_data.copy()
        data_with_outliers.loc[0, 'hargaJual'] = 2000000  # Extreme high value
        data_with_outliers.loc[1, 'hargaJual'] = 100000   # Extreme low value
        
        config = test_configs['data']
        config.outlier_method = 'iqr'
        
        preprocessor = GoldDataPreprocessor(config)
        preprocessor.data = data_with_outliers
        
        outliers = preprocessor.detect_outliers()
        
        assert isinstance(outliers, np.ndarray)
        assert len(outliers) > 0  # Should detect some outliers
        assert 0 in outliers or 1 in outliers  # Should detect our added outliers
    
    def test_detect_outliers_zscore(self, sample_gold_data, test_configs):
        """Test outlier detection using Z-score method."""
        data_with_outliers = sample_gold_data.copy()
        data_with_outliers.loc[0, 'hargaJual'] = 2000000
        
        config = test_configs['data']
        config.outlier_method = 'zscore'
        
        preprocessor = GoldDataPreprocessor(config)
        preprocessor.data = data_with_outliers
        
        outliers = preprocessor.detect_outliers()
        
        assert isinstance(outliers, np.ndarray)
        assert 0 in outliers  # Should detect our added outlier
    
    def test_engineer_features(self, sample_gold_data, test_configs):
        """Test feature engineering."""
        preprocessor = GoldDataPreprocessor(test_configs['data'])
        preprocessor.data = sample_gold_data.copy()
        
        # Convert date column to datetime
        preprocessor.data['tanggal'] = pd.to_datetime(preprocessor.data['tanggal'])
        preprocessor.data = preprocessor.data.sort_values('tanggal').reset_index(drop=True)
        
        result = preprocessor.engineer_features()
        
        assert_valid_dataframe(result)
        # Check for engineered features
        expected_features = [
            'price_change', 'price_change_pct', 'volatility',
            'ma_7', 'ma_30', 'rsi', 'bb_upper', 'bb_lower'
        ]
        for feature in expected_features:
            assert feature in result.columns, f"Feature {feature} not found"
    
    def test_add_lag_features(self, sample_gold_data, test_configs):
        """Test adding lag features."""
        preprocessor = GoldDataPreprocessor(test_configs['data'])
        preprocessor.data = sample_gold_data.copy()
        
        result = preprocessor.add_lag_features(lags=[1, 2, 3])
        
        assert_valid_dataframe(result)
        # Check for lag features
        for lag in [1, 2, 3]:
            assert f'hargaJual_lag_{lag}' in result.columns
    
    def test_normalize_data_minmax(self, sample_gold_data, test_configs):
        """Test data normalization with MinMax scaling."""
        config = test_configs['data']
        config.scaling_method = 'minmax'
        
        preprocessor = GoldDataPreprocessor(config)
        preprocessor.data = sample_gold_data.copy()
        
        normalized_data, scaler = preprocessor.normalize_data()
        
        assert_valid_dataframe(normalized_data)
        assert scaler is not None
        # Check that values are scaled to [0, 1]
        target_col = config.target_column
        if target_col in normalized_data.columns:
            assert normalized_data[target_col].min() >= 0
            assert normalized_data[target_col].max() <= 1
    
    def test_normalize_data_standard(self, sample_gold_data, test_configs):
        """Test data normalization with Standard scaling."""
        config = test_configs['data']
        config.scaling_method = 'standard'
        
        preprocessor = GoldDataPreprocessor(config)
        preprocessor.data = sample_gold_data.copy()
        
        normalized_data, scaler = preprocessor.normalize_data()
        
        assert_valid_dataframe(normalized_data)
        assert scaler is not None
        # Check that target has approximately zero mean and unit variance
        target_col = config.target_column
        if target_col in normalized_data.columns:
            assert abs(normalized_data[target_col].mean()) < 0.1
            assert abs(normalized_data[target_col].std() - 1.0) < 0.1
    
    def test_create_sequences(self, test_configs):
        """Test sequence creation for LSTM."""
        # Create simple time series data
        data = pd.DataFrame({
            'value': range(100),
            'feature1': range(100, 200),
            'feature2': range(200, 300)
        })
        
        config = test_configs['data']
        config.sequence_length = 10
        
        preprocessor = GoldDataPreprocessor(config)
        
        X, y = preprocessor.create_sequences(data, target_col='value')
        
        # Check shapes
        expected_samples = len(data) - config.sequence_length
        assert X.shape == (expected_samples, config.sequence_length, data.shape[1])
        assert y.shape == (expected_samples,)
        
        # Check that sequences are correct
        assert np.array_equal(X[0, :, 0], data['value'].iloc[:10].values)
        assert y[0] == data['value'].iloc[10]
    
    def test_split_data(self, test_configs):
        """Test data splitting."""
        # Create sample data
        X = np.random.randn(100, 30, 5)
        y = np.random.randn(100)
        
        preprocessor = GoldDataPreprocessor(test_configs['data'])
        
        splits = preprocessor.split_data(X, y)
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        
        # Check that splits have correct proportions
        total_samples = len(X)
        train_size = int(test_configs['data'].train_split * total_samples)
        val_size = int(test_configs['data'].val_split * total_samples)
        test_size = total_samples - train_size - val_size
        
        assert len(X_train) == train_size
        assert len(X_val) == val_size
        assert len(X_test) == test_size
        
        # Check that data types are preserved
        assert X_train.shape[1:] == X.shape[1:]
        assert len(y_train) == len(X_train)
    
    def test_preprocess_pipeline(self, sample_gold_data, temp_dir, test_configs):
        """Test complete preprocessing pipeline."""
        # Save sample data
        json_file = os.path.join(temp_dir, 'test_data.json')
        sample_gold_data.to_json(json_file, orient='records', date_format='iso')
        
        config = test_configs['data']
        config.data_path = json_file
        config.feature_engineering = True
        
        preprocessor = GoldDataPreprocessor(config)
        
        # Run complete pipeline
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess()
        
        # Validate outputs
        assert X_train is not None and len(X_train) > 0
        assert X_val is not None and len(X_val) > 0
        assert X_test is not None and len(X_test) > 0
        assert y_train is not None and len(y_train) > 0
        assert y_val is not None and len(y_val) > 0
        assert y_test is not None and len(y_test) > 0
        
        # Check shapes consistency
        assert X_train.shape[0] == len(y_train)
        assert X_val.shape[0] == len(y_val)
        assert X_test.shape[0] == len(y_test)
        
        # Check that feature dimensions match
        assert X_train.shape[1:] == X_val.shape[1:] == X_test.shape[1:]
    
    def test_save_and_load_scaler(self, sample_gold_data, temp_dir, test_configs):
        """Test saving and loading scaler."""
        preprocessor = GoldDataPreprocessor(test_configs['data'])
        preprocessor.data = sample_gold_data.copy()
        
        # Normalize data to create scaler
        normalized_data, scaler = preprocessor.normalize_data()
        
        # Save scaler
        scaler_path = os.path.join(temp_dir, 'test_scaler.pkl')
        preprocessor.save_scaler(scaler_path)
        
        assert os.path.exists(scaler_path)
        
        # Load scaler
        loaded_scaler = preprocessor.load_scaler(scaler_path)
        
        # Test that loaded scaler works
        test_data = sample_gold_data[['hargaJual']].values
        original_transform = scaler.transform(test_data)
        loaded_transform = loaded_scaler.transform(test_data)
        
        np.testing.assert_array_almost_equal(original_transform, loaded_transform)
    
    def test_inverse_transform(self, sample_gold_data, test_configs):
        """Test inverse transformation."""
        preprocessor = GoldDataPreprocessor(test_configs['data'])
        preprocessor.data = sample_gold_data.copy()
        
        # Normalize data
        normalized_data, scaler = preprocessor.normalize_data()
        preprocessor.scaler = scaler
        
        # Get some normalized values
        normalized_values = normalized_data[test_configs['data'].target_column].values[:10]
        
        # Inverse transform
        original_values = preprocessor.inverse_transform_target(normalized_values)
        
        # Check that inverse transform is reasonable
        assert len(original_values) == len(normalized_values)
        assert all(val > 0 for val in original_values)  # Gold prices should be positive
    
    @pytest.mark.parametrize("method", ['iqr', 'zscore'])
    def test_outlier_methods(self, sample_gold_data, test_configs, method):
        """Test different outlier detection methods."""
        config = test_configs['data']
        config.outlier_method = method
        
        preprocessor = GoldDataPreprocessor(config)
        preprocessor.data = sample_gold_data.copy()
        
        outliers = preprocessor.detect_outliers()
        
        assert isinstance(outliers, np.ndarray)
        # Should not detect too many outliers in normal data
        assert len(outliers) < len(sample_gold_data) * 0.1
    
    @pytest.mark.parametrize("scaling", ['minmax', 'standard', 'robust'])
    def test_scaling_methods(self, sample_gold_data, test_configs, scaling):
        """Test different scaling methods."""
        config = test_configs['data']
        config.scaling_method = scaling
        
        preprocessor = GoldDataPreprocessor(config)
        preprocessor.data = sample_gold_data.copy()
        
        normalized_data, scaler = preprocessor.normalize_data()
        
        assert_valid_dataframe(normalized_data)
        assert scaler is not None
        
        # Test that scaler can transform new data
        test_data = sample_gold_data.iloc[:5]
        transformed = scaler.transform(test_data[['hargaJual', 'hargaBeli']])
        assert transformed.shape == (5, 2)
