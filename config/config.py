"""
Configuration module for Gold Price Prediction LSTM Model
Author: ridwaanhall
Date: 2025-06-08
"""

import os
from dataclasses import dataclass
from typing import List


@dataclass
class DataConfig:
    """Data processing configuration parameters"""
    DATA_PATH: str = "data/"
    SEQUENCE_LENGTH: int = 60
    PREDICTION_HORIZON: int = 1
    TRAIN_RATIO: float = 0.8
    VALIDATION_RATIO: float = 0.1
    TEST_RATIO: float = 0.1
    TARGET_COLUMN: str = "hargaJual"
    DATE_COLUMN: str = "lastUpdate"
    FEATURES: List[str] = None  # Will be set in preprocessing
    
    def __post_init__(self):
        if self.FEATURES is None:
            self.FEATURES = ["hargaJual", "hargaBeli", "price_diff", "price_ratio"]


@dataclass
class ModelConfig:
    """LSTM model architecture configuration"""
    LSTM_UNITS: List[int] = None
    DROPOUT_RATE: float = 0.2
    RECURRENT_DROPOUT: float = 0.2
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    PATIENCE: int = 10
    MIN_DELTA: float = 0.0001
    VALIDATION_SPLIT: float = 0.2
    SHUFFLE: bool = False  # False for time series
    BIDIRECTIONAL: bool = False
    ATTENTION: bool = False
    
    def __post_init__(self):
        if self.LSTM_UNITS is None:
            self.LSTM_UNITS = [50, 50, 50]


@dataclass
class TrainingConfig:
    """Training pipeline configuration"""
    EARLY_STOPPING_PATIENCE: int = 10
    REDUCE_LR_PATIENCE: int = 5
    REDUCE_LR_FACTOR: float = 0.5
    MIN_LR: float = 1e-7
    MODEL_CHECKPOINT_PATH: str = "models/checkpoints/"
    SAVED_MODELS_PATH: str = "models/saved_models/"
    LOGS_PATH: str = "models/logs/"
    TENSORBOARD_LOG_DIR: str = "models/logs/tensorboard/"
    
    # Cross-validation settings
    CV_SPLITS: int = 5
    WALK_FORWARD_STEPS: int = 30
    
    # Hyperparameter tuning
    TUNING_TRIALS: int = 100
    TUNING_TIMEOUT: int = 3600  # 1 hour
    
    # Model versioning
    MODEL_VERSION_FORMAT: str = "v{major}.{minor}.{patch}"
    AUTO_VERSION: bool = True


@dataclass
class EvaluationConfig:
    """Model evaluation configuration"""
    METRICS: List[str] = None
    CONFIDENCE_LEVEL: float = 0.95
    BOOTSTRAP_SAMPLES: int = 1000
    SIGNIFICANCE_LEVEL: float = 0.05
    
    def __post_init__(self):
        if self.METRICS is None:
            self.METRICS = ["rmse", "mae", "mape", "r2", "direction_accuracy"]


@dataclass
class PredictionConfig:
    """Prediction configuration"""
    FORECAST_HORIZON: int = 7  # days
    CONFIDENCE_INTERVALS: bool = True
    PREDICTION_INTERVALS: List[float] = None
    TREND_DETECTION: bool = True
    ANOMALY_DETECTION: bool = True
    
    def __post_init__(self):
        if self.PREDICTION_INTERVALS is None:
            self.PREDICTION_INTERVALS = [0.80, 0.90, 0.95]


@dataclass
class VisualizationConfig:
    """Visualization configuration"""
    FIGURE_SIZE: tuple = (12, 8)
    DPI: int = 300
    STYLE: str = "seaborn-v0_8"
    COLOR_PALETTE: str = "husl"
    SAVE_PLOTS: bool = True
    PLOTS_DIR: str = "plots/"
    FORMAT: str = "png"


@dataclass
class LoggingConfig:
    """Logging configuration"""
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = "models/logs/gold_prediction.log"
    MAX_LOG_SIZE: int = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT: int = 5


@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_VERSION: str = "v1"
    MODEL_SERVING_BATCH_SIZE: int = 100
    MAX_REQUEST_SIZE: int = 1024 * 1024  # 1MB
    CACHE_TTL: int = 300  # 5 minutes
    MONITORING_ENABLED: bool = True
    
    # Docker settings
    DOCKER_IMAGE_NAME: str = "gold-price-predictor"
    DOCKER_TAG: str = "latest"


# Global configuration instance
class Config:
    """Main configuration class that combines all config dataclasses"""
    
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.evaluation = EvaluationConfig()
        self.prediction = PredictionConfig()
        self.visualization = VisualizationConfig()
        self.logging = LoggingConfig()
        self.deployment = DeploymentConfig()
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.data.DATA_PATH,
            self.training.MODEL_CHECKPOINT_PATH,
            self.training.SAVED_MODELS_PATH,
            self.training.LOGS_PATH,
            self.training.TENSORBOARD_LOG_DIR,
            self.visualization.PLOTS_DIR,
            os.path.dirname(self.logging.LOG_FILE)
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def update_from_dict(self, config_dict: dict):
        """Update configuration from dictionary"""
        for section, values in config_dict.items():
            if hasattr(self, section):
                config_section = getattr(self, section)
                for key, value in values.items():
                    if hasattr(config_section, key):
                        setattr(config_section, key, value)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'prediction': self.prediction.__dict__,
            'visualization': self.visualization.__dict__,
            'logging': self.logging.__dict__,
            'deployment': self.deployment.__dict__
        }


# Create global config instance
config = Config()
