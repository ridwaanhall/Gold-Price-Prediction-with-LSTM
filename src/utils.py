"""
Utility functions and helpers for Gold Price Prediction LSTM Model
Author: ridwaanhall
Date: 2025-06-08
"""

import os
import json
import logging
import pickle
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 log_format: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        log_format: Optional custom log format
    
    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logger
    logger = logging.getLogger('gold_price_predictor')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_directories(directories: List[str]) -> None:
    """
    Create multiple directories if they don't exist.
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save the JSON file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Path to the JSON file
    
    Returns:
        Loaded data as dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: str) -> None:
    """
    Save data to pickle file.
    
    Args:
        data: Data to save
        filepath: Path to save the pickle file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: str) -> Any:
    """
    Load data from pickle file.
    
    Args:
        filepath: Path to the pickle file
    
    Returns:
        Loaded data
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def calculate_date_features(date_series: pd.Series) -> pd.DataFrame:
    """
    Calculate time-based features from date series.
    
    Args:
        date_series: Pandas Series containing datetime values
    
    Returns:
        DataFrame with time-based features
    """
    features = pd.DataFrame(index=date_series.index)
    
    # Basic time features
    features['year'] = date_series.dt.year
    features['month'] = date_series.dt.month
    features['day'] = date_series.dt.day
    features['weekday'] = date_series.dt.weekday
    features['quarter'] = date_series.dt.quarter
    features['day_of_year'] = date_series.dt.dayofyear
    features['week_of_year'] = date_series.dt.isocalendar().week
    
    # Cyclical features
    features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
    features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
    features['day_sin'] = np.sin(2 * np.pi * features['day'] / 31)
    features['day_cos'] = np.cos(2 * np.pi * features['day'] / 31)
    features['weekday_sin'] = np.sin(2 * np.pi * features['weekday'] / 7)
    features['weekday_cos'] = np.cos(2 * np.pi * features['weekday'] / 7)
    
    # Business day indicator
    features['is_business_day'] = date_series.dt.weekday < 5
    
    # Weekend indicator
    features['is_weekend'] = date_series.dt.weekday >= 5
    
    # Month start/end indicators
    features['is_month_start'] = date_series.dt.is_month_start
    features['is_month_end'] = date_series.dt.is_month_end
    
    # Quarter start/end indicators
    features['is_quarter_start'] = date_series.dt.is_quarter_start
    features['is_quarter_end'] = date_series.dt.is_quarter_end
    
    return features


def validate_data_format(data: pd.DataFrame, 
                        required_columns: List[str],
                        date_column: str = 'lastUpdate') -> Tuple[bool, List[str]]:
    """
    Validate data format and required columns.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        date_column: Name of the date column
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check if required columns exist
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    # Check if date column exists and has valid format
    if date_column not in data.columns:
        errors.append(f"Missing date column: {date_column}")
    else:
        try:
            pd.to_datetime(data[date_column])
        except Exception as e:
            errors.append(f"Invalid date format in column {date_column}: {str(e)}")
    
    # Check for empty data
    if data.empty:
        errors.append("Data is empty")
    
    # Check for duplicate dates
    if date_column in data.columns:
        try:
            date_col = pd.to_datetime(data[date_column])
            duplicates = date_col.duplicated().sum()
            if duplicates > 0:
                errors.append(f"Found {duplicates} duplicate dates")
        except:
            pass
    
    # Check for numeric columns
    numeric_columns = [col for col in required_columns if col != date_column]
    for col in numeric_columns:
        if col in data.columns:
            try:
                pd.to_numeric(data[col])
            except Exception as e:
                errors.append(f"Column {col} contains non-numeric values: {str(e)}")
    
    return len(errors) == 0, errors


def generate_model_id(config: Dict[str, Any]) -> str:
    """
    Generate unique model ID based on configuration.
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        Unique model ID string
    """
    # Create a string representation of the config
    config_str = json.dumps(config, sort_keys=True)
    
    # Generate hash
    model_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"lstm_gold_model_{timestamp}_{model_hash}"


def calculate_technical_indicators(df: pd.DataFrame, 
                                 price_column: str = 'hargaJual',
                                 periods: List[int] = None) -> pd.DataFrame:
    """
    Calculate technical indicators for price data.
    
    Args:
        df: DataFrame containing price data
        price_column: Name of the price column
        periods: List of periods for moving averages
    
    Returns:
        DataFrame with technical indicators
    """
    if periods is None:
        periods = [5, 10, 20, 50]
    
    result_df = df.copy()
    prices = df[price_column].astype(float)
    
    # Simple Moving Averages
    for period in periods:
        result_df[f'sma_{period}'] = prices.rolling(window=period).mean()
    
    # Exponential Moving Averages
    for period in periods:
        result_df[f'ema_{period}'] = prices.ewm(span=period).mean()
    
    # RSI (Relative Strength Index)
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    result_df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = prices.ewm(span=12).mean()
    ema_26 = prices.ewm(span=26).mean()
    result_df['macd'] = ema_12 - ema_26
    result_df['macd_signal'] = result_df['macd'].ewm(span=9).mean()
    result_df['macd_histogram'] = result_df['macd'] - result_df['macd_signal']
    
    # Bollinger Bands
    sma_20 = prices.rolling(window=20).mean()
    std_20 = prices.rolling(window=20).std()
    result_df['bollinger_upper'] = sma_20 + (std_20 * 2)
    result_df['bollinger_lower'] = sma_20 - (std_20 * 2)
    result_df['bollinger_width'] = result_df['bollinger_upper'] - result_df['bollinger_lower']
    
    # Price momentum
    result_df['momentum_1'] = prices.pct_change(1)
    result_df['momentum_3'] = prices.pct_change(3)
    result_df['momentum_7'] = prices.pct_change(7)
    
    # Volatility
    result_df['volatility_5'] = prices.rolling(window=5).std()
    result_df['volatility_10'] = prices.rolling(window=10).std()
    result_df['volatility_20'] = prices.rolling(window=20).std()
    
    return result_df


def detect_outliers(data: pd.Series, 
                   method: str = 'iqr',
                   threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers in data series.
    
    Args:
        data: Data series to check for outliers
        method: Method to use ('iqr', 'zscore', 'isolation_forest')
        threshold: Threshold for outlier detection
    
    Returns:
        Boolean series indicating outliers
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (data < lower_bound) | (data > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
    
    elif method == 'isolation_forest':
        try:
            from sklearn.ensemble import IsolationForest
            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = isolation_forest.fit_predict(data.values.reshape(-1, 1))
            return pd.Series(outlier_labels == -1, index=data.index)
        except ImportError:
            print("scikit-learn not available, falling back to IQR method")
            return detect_outliers(data, method='iqr', threshold=threshold)
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def handle_missing_values(df: pd.DataFrame, 
                         strategy: str = 'forward_fill',
                         columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: DataFrame with missing values
        strategy: Strategy to handle missing values
        columns: Specific columns to process
    
    Returns:
        DataFrame with missing values handled
    """
    result_df = df.copy()
    
    if columns is None:
        columns = df.columns.tolist()
    
    for col in columns:
        if col in result_df.columns:
            if strategy == 'forward_fill':
                result_df[col] = result_df[col].fillna(method='ffill')
            elif strategy == 'backward_fill':
                result_df[col] = result_df[col].fillna(method='bfill')
            elif strategy == 'interpolate':
                result_df[col] = result_df[col].interpolate()
            elif strategy == 'mean':
                result_df[col] = result_df[col].fillna(result_df[col].mean())
            elif strategy == 'median':
                result_df[col] = result_df[col].fillna(result_df[col].median())
            elif strategy == 'drop':
                result_df = result_df.dropna(subset=[col])
    
    return result_df


def split_time_series_data(data: pd.DataFrame,
                          train_ratio: float = 0.8,
                          val_ratio: float = 0.1,
                          test_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data while maintaining temporal order.
    
    Args:
        data: DataFrame to split
        train_ratio: Ratio for training data
        val_ratio: Ratio for validation data
        test_ratio: Ratio for test data
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = data.iloc[:train_end].copy()
    val_df = data.iloc[train_end:val_end].copy()
    test_df = data.iloc[val_end:].copy()
    
    return train_df, val_df, test_df


def create_feature_lag_matrix(data: pd.DataFrame,
                             feature_columns: List[str],
                             lags: List[int]) -> pd.DataFrame:
    """
    Create lag features for specified columns.
    
    Args:
        data: DataFrame with features
        feature_columns: Columns to create lags for
        lags: List of lag periods
    
    Returns:
        DataFrame with lag features
    """
    result_df = data.copy()
    
    for col in feature_columns:
        if col in data.columns:
            for lag in lags:
                result_df[f'{col}_lag_{lag}'] = data[col].shift(lag)
    
    return result_df


def ensure_datetime_index(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Ensure DataFrame has datetime index.
    
    Args:
        df: DataFrame to process
        date_column: Name of date column
    
    Returns:
        DataFrame with datetime index
    """
    result_df = df.copy()
    
    if date_column in result_df.columns:
        result_df[date_column] = pd.to_datetime(result_df[date_column])
        result_df = result_df.set_index(date_column)
        result_df.index.name = 'date'
    
    return result_df


def get_model_summary_dict(model) -> Dict[str, Any]:
    """
    Get model summary as dictionary.
    
    Args:
        model: Keras model
    
    Returns:
        Model summary as dictionary
    """
    try:
        # Get model configuration
        config = model.get_config()
        
        # Count parameters
        total_params = model.count_params()
        trainable_params = sum([layer.count_params() for layer in model.layers if layer.trainable])
        non_trainable_params = total_params - trainable_params
        
        # Layer information
        layers_info = []
        for layer in model.layers:
            layer_info = {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'output_shape': str(layer.output_shape),
                'params': layer.count_params()
            }
            layers_info.append(layer_info)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'layers': layers_info,
            'config': config
        }
    except Exception as e:
        return {'error': str(e)}


def save_model_metadata(model, 
                       model_path: str,
                       config: Dict[str, Any],
                       training_history: Dict[str, Any],
                       evaluation_metrics: Dict[str, Any]) -> None:
    """
    Save comprehensive model metadata.
    
    Args:
        model: Trained model
        model_path: Path where model is saved
        config: Model configuration
        training_history: Training history
        evaluation_metrics: Evaluation metrics
    """
    metadata = {
        'model_path': model_path,
        'creation_date': datetime.now().isoformat(),
        'config': config,
        'model_summary': get_model_summary_dict(model),
        'training_history': training_history,
        'evaluation_metrics': evaluation_metrics,
        'version': '1.0.0'
    }
    
    metadata_path = model_path.replace('.h5', '_metadata.json').replace('.keras', '_metadata.json')
    save_json(metadata, metadata_path)


def load_model_metadata(model_path: str) -> Dict[str, Any]:
    """
    Load model metadata.
    
    Args:
        model_path: Path to the model
    
    Returns:
        Model metadata dictionary
    """
    metadata_path = model_path.replace('.h5', '_metadata.json').replace('.keras', '_metadata.json')
    
    if os.path.exists(metadata_path):
        return load_json(metadata_path)
    else:
        return {}


def format_number(number: float, decimals: int = 2) -> str:
    """
    Format number with thousands separator.
    
    Args:
        number: Number to format
        decimals: Number of decimal places
    
    Returns:
        Formatted number string
    """
    return f"{number:,.{decimals}f}"


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
    
    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


def get_file_size(filepath: str) -> str:
    """
    Get human-readable file size.
    
    Args:
        filepath: Path to file
    
    Returns:
        Formatted file size string
    """
    if not os.path.exists(filepath):
        return "File not found"
    
    size_bytes = os.path.getsize(filepath)
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} TB"


def create_backup(filepath: str, backup_dir: str = "backups") -> str:
    """
    Create backup of a file.
    
    Args:
        filepath: Path to file to backup
        backup_dir: Directory to store backups
    
    Returns:
        Path to backup file
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    os.makedirs(backup_dir, exist_ok=True)
    
    # Create backup filename with timestamp
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{name}_{timestamp}{ext}"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    # Copy file
    import shutil
    shutil.copy2(filepath, backup_path)
    
    return backup_path
