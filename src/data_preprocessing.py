"""
Data Preprocessing Module for Gold Price Prediction LSTM Model
Author: ridwaanhall
Date: 2025-06-08
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from .utils import (
    setup_logging, calculate_date_features, validate_data_format,
    calculate_technical_indicators, detect_outliers, handle_missing_values,
    split_time_series_data, create_feature_lag_matrix, ensure_datetime_index,
    save_json, load_json, save_pickle, load_pickle
)


class GoldDataPreprocessor:
    """
    Comprehensive data preprocessing class for gold price prediction.
    
    This class handles data loading, cleaning, feature engineering,
    normalization, and sequence creation for LSTM model training.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration dictionary containing preprocessing parameters
        """
        self.config = config or {}
        self.logger = setup_logging()
        
        # Default configuration
        self.sequence_length = self.config.get('sequence_length', 60)
        self.prediction_horizon = self.config.get('prediction_horizon', 1)
        self.target_column = self.config.get('target_column', 'hargaJual')
        self.date_column = self.config.get('date_column', 'lastUpdate')
        
        # Scalers
        self.scaler = None
        self.target_scaler = None
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.feature_columns = []
        self.preprocessing_stats = {}
        
        self.logger.info("GoldDataPreprocessor initialized")
    
    def load_data_from_json(self, filepath: str) -> pd.DataFrame:
        """
        Load gold price data from JSON file.
        
        Args:
            filepath: Path to JSON file containing gold price data
        
        Returns:
            DataFrame with loaded data
        """
        try:
            self.logger.info(f"Loading data from {filepath}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            self.logger.info(f"Loaded {len(df)} records")
            
            # Store raw data
            self.raw_data = df.copy()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data from {filepath}: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame, 
                   remove_duplicates: bool = True,
                   handle_missing: str = 'forward_fill',
                   outlier_method: str = 'iqr',
                   outlier_threshold: float = 1.5) -> pd.DataFrame:
        """
        Clean the raw data by handling missing values, duplicates, and outliers.
        
        Args:
            df: DataFrame to clean
            remove_duplicates: Whether to remove duplicate entries
            handle_missing: Strategy for handling missing values
            outlier_method: Method for outlier detection
            outlier_threshold: Threshold for outlier detection
        
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Starting data cleaning process")
        cleaned_df = df.copy()
        cleaning_stats = {}
        
        # Convert date column to datetime
        if self.date_column in cleaned_df.columns:
            cleaned_df[self.date_column] = pd.to_datetime(cleaned_df[self.date_column])
            cleaned_df = cleaned_df.sort_values(self.date_column).reset_index(drop=True)
        
        # Convert price columns to numeric
        price_columns = ['hargaJual', 'hargaBeli']
        for col in price_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        
        # Record initial data quality
        cleaning_stats['initial_rows'] = len(cleaned_df)
        cleaning_stats['initial_missing'] = cleaned_df.isnull().sum().to_dict()
        
        # Remove duplicates
        if remove_duplicates and self.date_column in cleaned_df.columns:
            duplicates_before = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates(subset=[self.date_column], keep='last')
            duplicates_removed = duplicates_before - len(cleaned_df)
            cleaning_stats['duplicates_removed'] = duplicates_removed
            
            if duplicates_removed > 0:
                self.logger.info(f"Removed {duplicates_removed} duplicate records")
        
        # Handle missing values
        if cleaned_df.isnull().sum().sum() > 0:
            self.logger.info(f"Handling missing values using {handle_missing} strategy")
            cleaned_df = handle_missing_values(cleaned_df, strategy=handle_missing, 
                                            columns=price_columns)
        
        # Detect and handle outliers
        outliers_detected = {}
        for col in price_columns:
            if col in cleaned_df.columns:
                outlier_mask = detect_outliers(cleaned_df[col], method=outlier_method, 
                                            threshold=outlier_threshold)
                outliers_count = outlier_mask.sum()
                outliers_detected[col] = outliers_count
                
                if outliers_count > 0:
                    self.logger.info(f"Detected {outliers_count} outliers in {col}")
                    # Replace outliers with interpolated values
                    cleaned_df.loc[outlier_mask, col] = np.nan
                    cleaned_df[col] = cleaned_df[col].interpolate()
        
        cleaning_stats['outliers_detected'] = outliers_detected
        cleaning_stats['final_rows'] = len(cleaned_df)
        cleaning_stats['final_missing'] = cleaned_df.isnull().sum().to_dict()
        
        # Store cleaning statistics
        self.preprocessing_stats['cleaning'] = cleaning_stats
        
        self.logger.info(f"Data cleaning completed. Final dataset: {len(cleaned_df)} rows")
        return cleaned_df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from date column.
        
        Args:
            df: DataFrame with date column
        
        Returns:
            DataFrame with additional time features
        """
        self.logger.info("Creating time-based features")
        
        if self.date_column not in df.columns:
            self.logger.warning(f"Date column {self.date_column} not found")
            return df
        
        result_df = df.copy()
        
        # Ensure datetime format
        date_series = pd.to_datetime(result_df[self.date_column])
        
        # Calculate time features
        time_features = calculate_date_features(date_series)
        
        # Add to main dataframe
        for col in time_features.columns:
            result_df[f'time_{col}'] = time_features[col]
        
        feature_count = len(time_features.columns)
        self.logger.info(f"Added {feature_count} time-based features")
        
        return result_df
    
    def create_lag_features(self, df: pd.DataFrame, 
                           feature_columns: Optional[List[str]] = None,
                           lags: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Create lag features for specified columns.
        
        Args:
            df: DataFrame to create lag features for
            feature_columns: Columns to create lags for
            lags: List of lag periods
        
        Returns:
            DataFrame with lag features
        """
        if lags is None:
            lags = [1, 2, 3, 5, 7, 14, 30]
        
        if feature_columns is None:
            feature_columns = ['hargaJual', 'hargaBeli']
        
        self.logger.info(f"Creating lag features for {len(feature_columns)} columns with lags: {lags}")
        
        result_df = create_feature_lag_matrix(df, feature_columns, lags)
        
        lag_feature_count = len(feature_columns) * len(lags)
        self.logger.info(f"Added {lag_feature_count} lag features")
        
        return result_df
    
    def calculate_technical_indicators(self, df: pd.DataFrame,
                                     price_column: Optional[str] = None,
                                     periods: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate technical indicators for price analysis.
        
        Args:
            df: DataFrame with price data
            price_column: Column name for price data
            periods: Periods for moving averages
        
        Returns:
            DataFrame with technical indicators
        """
        if price_column is None:
            price_column = self.target_column
        
        if periods is None:
            periods = [5, 10, 20, 50]
        
        self.logger.info(f"Calculating technical indicators for {price_column}")
        
        result_df = calculate_technical_indicators(df, price_column, periods)
        
        # Count technical indicator features
        original_cols = set(df.columns)
        new_cols = set(result_df.columns) - original_cols
        
        self.logger.info(f"Added {len(new_cols)} technical indicator features")
        
        return result_df
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from existing price data.
        
        Args:
            df: DataFrame with price data
        
        Returns:
            DataFrame with derived features
        """
        self.logger.info("Creating derived features")
        result_df = df.copy()
        
        # Price difference and ratio features
        if 'hargaJual' in result_df.columns and 'hargaBeli' in result_df.columns:
            result_df['price_diff'] = result_df['hargaJual'] - result_df['hargaBeli']
            result_df['price_ratio'] = result_df['hargaJual'] / result_df['hargaBeli']
            result_df['price_spread_pct'] = (result_df['price_diff'] / result_df['hargaBeli']) * 100
        
        # Price returns and volatility
        if self.target_column in result_df.columns:
            price = result_df[self.target_column]
            
            # Returns
            result_df['return_1d'] = price.pct_change(1)
            result_df['return_3d'] = price.pct_change(3)
            result_df['return_7d'] = price.pct_change(7)
            
            # Rolling statistics
            for window in [5, 10, 20]:
                result_df[f'rolling_mean_{window}'] = price.rolling(window).mean()
                result_df[f'rolling_std_{window}'] = price.rolling(window).std()
                result_df[f'rolling_min_{window}'] = price.rolling(window).min()
                result_df[f'rolling_max_{window}'] = price.rolling(window).max()
            
            # Price position in rolling window
            for window in [10, 20, 50]:
                rolling_min = price.rolling(window).min()
                rolling_max = price.rolling(window).max()
                result_df[f'price_position_{window}'] = (price - rolling_min) / (rolling_max - rolling_min)
        
        derived_features = set(result_df.columns) - set(df.columns)
        self.logger.info(f"Added {len(derived_features)} derived features")
        
        return result_df
    
    def normalize_data(self, df: pd.DataFrame, 
                      scaler_type: str = 'MinMaxScaler',
                      feature_range: Tuple[float, float] = (0, 1),
                      fit_scaler: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features for model training.
        
        Args:
            df: DataFrame to normalize
            scaler_type: Type of scaler to use
            feature_range: Range for MinMaxScaler
            fit_scaler: Whether to fit the scaler (True for training data)
        
        Returns:
            Normalized DataFrame
        """
        self.logger.info(f"Normalizing data using {scaler_type}")
        
        result_df = df.copy()
        
        # Identify numerical columns (excluding date and categorical features)
        exclude_columns = [self.date_column, 'id'] + [col for col in df.columns if col.startswith('time_') and 'sin' not in col and 'cos' not in col]
        numerical_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col not in exclude_columns]
        
        if not numerical_columns:
            self.logger.warning("No numerical columns found for normalization")
            return result_df
        
        # Initialize scaler if needed
        if fit_scaler or self.scaler is None:
            if scaler_type == 'MinMaxScaler':
                self.scaler = MinMaxScaler(feature_range=feature_range)
            elif scaler_type == 'StandardScaler':
                self.scaler = StandardScaler()
            elif scaler_type == 'RobustScaler':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        # Fit and transform or just transform
        try:
            if fit_scaler:
                result_df[numerical_columns] = self.scaler.fit_transform(result_df[numerical_columns])
                self.logger.info(f"Fitted and transformed {len(numerical_columns)} numerical features")
            else:
                result_df[numerical_columns] = self.scaler.transform(result_df[numerical_columns])
                self.logger.info(f"Transformed {len(numerical_columns)} numerical features")
        except Exception as e:
            self.logger.error(f"Error during normalization: {str(e)}")
            raise
        
        # Store feature columns for later use
        self.feature_columns = numerical_columns
        
        return result_df
    
    def create_sequences(self, df: pd.DataFrame, 
                        sequence_length: Optional[int] = None,
                        prediction_horizon: Optional[int] = None,
                        target_column: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM model training.
        
        Args:
            df: DataFrame with features
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps to predict ahead
            target_column: Target column name
        
        Returns:
            Tuple of (X, y) arrays for model training
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        if prediction_horizon is None:
            prediction_horizon = self.prediction_horizon
        
        if target_column is None:
            target_column = self.target_column
        
        self.logger.info(f"Creating sequences with length {sequence_length} and prediction horizon {prediction_horizon}")
        
        # Select features for sequence creation
        feature_columns = [col for col in df.columns if col not in [self.date_column, 'id']]
        
        # Ensure target column is in features
        if target_column not in feature_columns:
            self.logger.error(f"Target column {target_column} not found in features")
            raise ValueError(f"Target column {target_column} not found")
        
        # Get target column index
        target_idx = feature_columns.index(target_column)
        
        # Create sequences
        X, y = [], []
        data_array = df[feature_columns].values
        
        for i in range(len(data_array) - sequence_length - prediction_horizon + 1):
            # Input sequence
            X.append(data_array[i:(i + sequence_length)])
            
            # Target value(s)
            target_start = i + sequence_length
            target_end = target_start + prediction_horizon
            
            if prediction_horizon == 1:
                y.append(data_array[target_start, target_idx])
            else:
                y.append(data_array[target_start:target_end, target_idx])
        
        X = np.array(X)
        y = np.array(y)
        
        self.logger.info(f"Created {len(X)} sequences with shape X: {X.shape}, y: {y.shape}")
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray,
                   train_ratio: float = 0.8,
                   val_ratio: float = 0.1,
                   test_ratio: float = 0.1) -> Tuple[np.ndarray, ...]:
        """
        Split sequences into train, validation, and test sets.
        
        Args:
            X: Input sequences
            y: Target values
            train_ratio: Ratio for training data
            val_ratio: Ratio for validation data
            test_ratio: Ratio for test data
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        self.logger.info(f"Splitting data with ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        n_samples = len(X)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        
        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]
        
        self.logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessed_data(self, output_dir: str) -> Dict[str, str]:
        """
        Save preprocessed data and preprocessing objects.
        
        Args:
            output_dir: Directory to save processed data
        
        Returns:
            Dictionary with paths to saved files
        """
        self.logger.info(f"Saving preprocessed data to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        
        # Save processed data
        if self.processed_data is not None:
            processed_data_path = os.path.join(output_dir, 'processed_data.csv')
            self.processed_data.to_csv(processed_data_path, index=False)
            saved_files['processed_data'] = processed_data_path
        
        # Save scaler
        if self.scaler is not None:
            scaler_path = os.path.join(output_dir, 'scaler.pkl')
            save_pickle(self.scaler, scaler_path)
            saved_files['scaler'] = scaler_path
        
        # Save target scaler
        if self.target_scaler is not None:
            target_scaler_path = os.path.join(output_dir, 'target_scaler.pkl')
            save_pickle(self.target_scaler, target_scaler_path)
            saved_files['target_scaler'] = target_scaler_path
        
        # Save feature columns
        feature_columns_path = os.path.join(output_dir, 'feature_columns.json')
        save_json({'feature_columns': self.feature_columns}, feature_columns_path)
        saved_files['feature_columns'] = feature_columns_path
        
        # Save preprocessing statistics
        stats_path = os.path.join(output_dir, 'preprocessing_stats.json')
        save_json(self.preprocessing_stats, stats_path)
        saved_files['preprocessing_stats'] = stats_path
        
        # Save configuration
        config_path = os.path.join(output_dir, 'preprocessing_config.json')
        save_json(self.config, config_path)
        saved_files['preprocessing_config'] = config_path
        
        self.logger.info(f"Saved {len(saved_files)} files")
        return saved_files
    
    def load_preprocessed_data(self, input_dir: str) -> None:
        """
        Load preprocessed data and preprocessing objects.
        
        Args:
            input_dir: Directory containing processed data
        """
        self.logger.info(f"Loading preprocessed data from {input_dir}")
        
        # Load scaler
        scaler_path = os.path.join(input_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = load_pickle(scaler_path)
        
        # Load target scaler
        target_scaler_path = os.path.join(input_dir, 'target_scaler.pkl')
        if os.path.exists(target_scaler_path):
            self.target_scaler = load_pickle(target_scaler_path)
        
        # Load feature columns
        feature_columns_path = os.path.join(input_dir, 'feature_columns.json')
        if os.path.exists(feature_columns_path):
            feature_data = load_json(feature_columns_path)
            self.feature_columns = feature_data.get('feature_columns', [])
        
        # Load preprocessing statistics
        stats_path = os.path.join(input_dir, 'preprocessing_stats.json')
        if os.path.exists(stats_path):
            self.preprocessing_stats = load_json(stats_path)
        
        # Load configuration
        config_path = os.path.join(input_dir, 'preprocessing_config.json')
        if os.path.exists(config_path):
            self.config = load_json(config_path)
        
        self.logger.info("Preprocessed data loaded successfully")
    
    def preprocess_pipeline(self, data_path: str, 
                           output_dir: Optional[str] = None) -> Tuple[np.ndarray, ...]:
        """
        Complete preprocessing pipeline from raw data to model-ready sequences.
        
        Args:
            data_path: Path to raw data file
            output_dir: Optional directory to save processed data
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        self.logger.info("Starting complete preprocessing pipeline")
        
        # Load data
        df = self.load_data_from_json(data_path)
        
        # Validate data format
        required_columns = ['hargaJual', 'hargaBeli', 'lastUpdate']
        is_valid, errors = validate_data_format(df, required_columns, self.date_column)
        
        if not is_valid:
            self.logger.error(f"Data validation failed: {errors}")
            raise ValueError(f"Data validation failed: {errors}")
        
        # Clean data
        df = self.clean_data(df)
        
        # Create time features
        df = self.create_time_features(df)
        
        # Create lag features
        df = self.create_lag_features(df)
        
        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Create derived features
        df = self.create_derived_features(df)
        
        # Remove rows with NaN values (from lag features and technical indicators)
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        
        self.logger.info(f"Removed {initial_rows - final_rows} rows with NaN values")
        
        if len(df) == 0:
            raise ValueError("No data remaining after preprocessing")
        
        # Store processed data
        self.processed_data = df.copy()
        
        # Normalize data
        df_normalized = self.normalize_data(df, fit_scaler=True)
        
        # Create sequences
        X, y = self.create_sequences(df_normalized)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Save processed data if output directory specified
        if output_dir:
            self.save_preprocessed_data(output_dir)
        
        self.logger.info("Preprocessing pipeline completed successfully")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def transform_new_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessors for prediction.
        
        Args:
            df: New data to transform
        
        Returns:
            Transformed data ready for model prediction
        """
        self.logger.info("Transforming new data for prediction")
        
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Run preprocessing pipeline first.")
        
        # Apply same transformations as training data
        df_processed = df.copy()
        
        # Clean data (without fitting)
        df_processed = self.clean_data(df_processed, remove_duplicates=False)
        
        # Create features
        df_processed = self.create_time_features(df_processed)
        df_processed = self.create_lag_features(df_processed)
        df_processed = self.calculate_technical_indicators(df_processed)
        df_processed = self.create_derived_features(df_processed)
        
        # Handle missing values
        df_processed = df_processed.dropna()
        
        # Normalize using fitted scaler
        df_normalized = self.normalize_data(df_processed, fit_scaler=False)
        
        # Create sequences
        X, _ = self.create_sequences(df_normalized)
        
        self.logger.info(f"Transformed new data: {X.shape}")
        
        return X
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Inverse transform predictions to original scale.
        
        Args:
            predictions: Normalized predictions
        
        Returns:
            Predictions in original scale
        """
        if self.scaler is None:
            self.logger.warning("No scaler available for inverse transformation")
            return predictions
        
        # Get target column index
        target_idx = self.feature_columns.index(self.target_column)
        
        # Create dummy array for inverse transformation
        dummy_data = np.zeros((len(predictions), len(self.feature_columns)))
        dummy_data[:, target_idx] = predictions.flatten()
        
        # Inverse transform
        inverse_transformed = self.scaler.inverse_transform(dummy_data)
        
        return inverse_transformed[:, target_idx]
    
    def get_preprocessing_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive preprocessing report.
        
        Returns:
            Dictionary containing preprocessing statistics and information
        """
        report = {
            'config': self.config,
            'feature_columns': self.feature_columns,
            'preprocessing_stats': self.preprocessing_stats,
            'data_info': {}
        }
        
        if self.processed_data is not None:
            report['data_info'] = {
                'shape': self.processed_data.shape,
                'columns': list(self.processed_data.columns),
                'dtypes': self.processed_data.dtypes.to_dict(),
                'memory_usage': self.processed_data.memory_usage(deep=True).sum(),
                'date_range': {
                    'start': str(self.processed_data[self.date_column].min()) if self.date_column in self.processed_data.columns else None,
                    'end': str(self.processed_data[self.date_column].max()) if self.date_column in self.processed_data.columns else None
                }
            }
        
        return report
