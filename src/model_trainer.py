"""
Model Training Module for Gold Price Prediction LSTM Model
Author: ridwaanhall
Date: 2025-06-08
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, 
    TensorBoard, CSVLogger, LambdaCallback
)
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings
warnings.filterwarnings('ignore')

from .utils import setup_logging, save_json, load_json, generate_model_id, save_model_metadata
from .lstm_model import LSTMGoldPredictor, create_custom_loss_functions, create_custom_metrics


class ModelTrainer:
    """
    Comprehensive model training class with advanced features.
    
    This class handles model training, validation, checkpointing,
    and hyperparameter optimization for LSTM gold price prediction.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model trainer.
        
        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config or {}
        self.logger = setup_logging()
        
        # Training configuration
        self.epochs = self.config.get('epochs', 100)
        self.batch_size = self.config.get('batch_size', 32)
        self.validation_split = self.config.get('validation_split', 0.2)
        self.early_stopping_patience = self.config.get('early_stopping_patience', 10)
        self.reduce_lr_patience = self.config.get('reduce_lr_patience', 5)
        self.min_lr = self.config.get('min_lr', 1e-7)
        self.reduce_lr_factor = self.config.get('reduce_lr_factor', 0.5)
        
        # Paths
        self.checkpoint_path = self.config.get('checkpoint_path', 'models/checkpoints/')
        self.logs_path = self.config.get('logs_path', 'models/logs/')
        self.tensorboard_log_dir = self.config.get('tensorboard_log_dir', 'models/logs/tensorboard/')
        
        # Model and training state
        self.model = None
        self.training_history = None
        self.best_model_path = None
        self.callbacks_list = []
        
        # Create directories
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)
        
        self.logger.info("ModelTrainer initialized")
    
    def setup_callbacks(self, 
                       model_name: str = "lstm_model",
                       monitor: str = 'val_loss',
                       mode: str = 'min',
                       custom_callbacks: Optional[List] = None) -> List:
        """
        Setup training callbacks for model optimization and monitoring.
        
        Args:
            model_name: Name for the model files
            monitor: Metric to monitor for callbacks
            mode: Direction of optimization ('min' or 'max')
            custom_callbacks: Additional custom callbacks
        
        Returns:
            List of configured callbacks
        """
        self.logger.info(f"Setting up callbacks for model: {model_name}")
        
        callbacks = []
        
        # Early Stopping
        early_stopping = EarlyStopping(
            monitor=monitor,
            patience=self.early_stopping_patience,
            mode=mode,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Model Checkpoint
        checkpoint_filepath = os.path.join(
            self.checkpoint_path, 
            f"{model_name}_best.keras"
        )
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(model_checkpoint)
        self.best_model_path = checkpoint_filepath
        
        # Reduce Learning Rate on Plateau
        reduce_lr = ReduceLROnPlateau(
            monitor=monitor,
            factor=self.reduce_lr_factor,
            patience=self.reduce_lr_patience,
            mode=mode,
            min_lr=self.min_lr,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard
        tensorboard_log_path = os.path.join(
            self.tensorboard_log_dir,
            model_name
        )
        tensorboard = TensorBoard(
            log_dir=tensorboard_log_path,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard)
        
        # CSV Logger
        csv_log_path = os.path.join(
            self.logs_path,
            f"{model_name}_training.csv"
        )
        csv_logger = CSVLogger(csv_log_path, append=True)
        callbacks.append(csv_logger)
        
        # Custom progress callback
        def log_progress(epoch, logs):
            self.logger.info(
                f"Epoch {epoch + 1}: "
                f"loss={logs.get('loss', 0):.4f}, "
                f"val_loss={logs.get('val_loss', 0):.4f}, "
                f"lr={logs.get('lr', 0):.2e}"
            )
        
        progress_callback = LambdaCallback(
            on_epoch_end=lambda epoch, logs: log_progress(epoch, logs)
        )
        callbacks.append(progress_callback)
        
        # Add custom callbacks if provided
        if custom_callbacks:
            callbacks.extend(custom_callbacks)
        
        self.callbacks_list = callbacks
        self.logger.info(f"Configured {len(callbacks)} callbacks")
        
        return callbacks
    
    def train_model(self, 
                   model: keras.Model,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None,
                   y_val: Optional[np.ndarray] = None,
                   model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the LSTM model with validation and monitoring.
        
        Args:
            model: Compiled Keras model
            X_train: Training input data
            y_train: Training target data
            X_val: Validation input data (optional)
            y_val: Validation target data (optional)
            model_name: Name for saving the model
        
        Returns:
            Training history and metadata
        """
        if model_name is None:
            model_name = generate_model_id({})
        
        self.logger.info(f"Starting training for model: {model_name}")
        self.logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        self.model = model
        
        # Setup callbacks
        callbacks = self.setup_callbacks(model_name)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            self.logger.info(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")
        
        # Record training start time
        import time
        start_time = time.time()
        
        try:
            # Train the model
            history = model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=validation_data,
                validation_split=self.validation_split if validation_data is None else None,
                callbacks=callbacks,
                verbose=1,
                shuffle=False  # Important for time series data
            )
            
            # Record training end time
            training_time = time.time() - start_time
            
            # Store training history
            self.training_history = {
                'history': history.history,
                'epochs_trained': len(history.history['loss']),
                'training_time': training_time,
                'model_name': model_name,
                'config': self.config
            }
            
            # Log training completion
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history.get('val_loss', [0])[-1]
            
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            self.logger.info(f"Final loss: {final_loss:.6f}, Final val_loss: {final_val_loss:.6f}")
            
            # Save training history
            history_path = os.path.join(self.logs_path, f"{model_name}_history.json")
            save_json(self.training_history, history_path)
            
            return self.training_history
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
    
    def validate_model(self, 
                      model: keras.Model,
                      X_test: np.ndarray,
                      y_test: np.ndarray) -> Dict[str, float]:
        """
        Validate the trained model on test data.
        
        Args:
            model: Trained model
            X_test: Test input data
            y_test: Test target data
        
        Returns:
            Dictionary of validation metrics
        """
        self.logger.info("Validating model on test data")
        
        # Make predictions
        y_pred = model.predict(X_test, verbose=0)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Calculate additional metrics
        mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
        
        # R-squared
        ss_res = np.sum((y_test - y_pred.flatten()) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Directional accuracy
        direction_actual = np.sign(np.diff(y_test))
        direction_predicted = np.sign(np.diff(y_pred.flatten()))
        direction_accuracy = np.mean(direction_actual == direction_predicted) * 100
        
        validation_metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2),
            'direction_accuracy': float(direction_accuracy)
        }
        
        self.logger.info("Validation metrics:")
        for metric, value in validation_metrics.items():
            self.logger.info(f"  {metric}: {value:.6f}")
        
        return validation_metrics
    
    def save_training_history(self, 
                            filepath: str,
                            include_config: bool = True) -> None:
        """
        Save comprehensive training history to file.
        
        Args:
            filepath: Path to save the history
            include_config: Whether to include configuration in the saved file
        """
        if self.training_history is None:
            raise ValueError("No training history available")
        
        history_data = self.training_history.copy()
        
        if include_config:
            history_data['trainer_config'] = self.config
        
        save_json(history_data, filepath)
        self.logger.info(f"Training history saved to {filepath}")
    
    def load_best_model(self) -> keras.Model:
        """
        Load the best model from checkpoints.
        
        Returns:
            Best trained model
        """
        if self.best_model_path is None or not os.path.exists(self.best_model_path):
            raise ValueError("Best model checkpoint not found")
        
        # Load custom objects
        custom_objects = create_custom_metrics()
        custom_objects.update(create_custom_loss_functions())
        
        model = keras.models.load_model(
            self.best_model_path,
            custom_objects=custom_objects
        )
        
        self.logger.info(f"Best model loaded from {self.best_model_path}")
        return model
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary.
        
        Returns:
            Dictionary with training summary information
        """
        if self.training_history is None:
            return {"status": "No training completed"}
        
        history = self.training_history['history']
        
        summary = {
            'model_name': self.training_history.get('model_name', 'unknown'),
            'epochs_trained': self.training_history['epochs_trained'],
            'training_time': self.training_history['training_time'],
            'final_metrics': {
                'loss': history['loss'][-1],
                'val_loss': history.get('val_loss', [0])[-1]
            },
            'best_metrics': {
                'best_loss': min(history['loss']),
                'best_val_loss': min(history.get('val_loss', [float('inf')]))
            },
            'best_epoch': {
                'loss': np.argmin(history['loss']) + 1,
                'val_loss': np.argmin(history.get('val_loss', [float('inf')])) + 1
            },
            'convergence_info': {
                'early_stopped': self.training_history['epochs_trained'] < self.epochs,
                'improvement_rate': self._calculate_improvement_rate(history['val_loss'] if 'val_loss' in history else history['loss'])
            }
        }
        
        return summary
    
    def _calculate_improvement_rate(self, loss_history: List[float]) -> float:
        """Calculate the rate of improvement in loss."""
        if len(loss_history) < 2:
            return 0.0
        
        improvements = []
        for i in range(1, len(loss_history)):
            if loss_history[i] < loss_history[i-1]:
                improvements.append((loss_history[i-1] - loss_history[i]) / loss_history[i-1])
        
        return np.mean(improvements) if improvements else 0.0


class HyperparameterTuner:
    """
    Advanced hyperparameter tuning using multiple optimization strategies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize hyperparameter tuner.
        
        Args:
            config: Configuration for hyperparameter tuning
        """
        self.config = config or {}
        self.logger = setup_logging()
        
        # Tuning configuration
        self.n_trials = self.config.get('n_trials', 100)
        self.timeout = self.config.get('timeout', 3600)  # 1 hour
        self.cv_folds = self.config.get('cv_folds', 3)
        self.scoring_metric = self.config.get('scoring_metric', 'val_loss')
        
        # Results storage
        self.study = None
        self.best_params = None
        self.tuning_results = {}
        
        self.logger.info("HyperparameterTuner initialized")
    
    def grid_search(self, 
                   param_grid: Dict[str, List],
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_val: np.ndarray,
                   y_val: np.ndarray,
                   model_builder_func: Callable) -> Dict[str, Any]:
        """
        Perform grid search hyperparameter optimization.
        
        Args:
            param_grid: Dictionary of parameters to search
            X_train: Training input data
            y_train: Training target data
            X_val: Validation input data
            y_val: Validation target data
            model_builder_func: Function to build model with given parameters
        
        Returns:
            Dictionary with best parameters and results
        """
        self.logger.info(f"Starting grid search with {len(list(ParameterGrid(param_grid)))} combinations")
        
        results = []
        best_score = float('inf')
        best_params = None
        
        for i, params in enumerate(ParameterGrid(param_grid)):
            self.logger.info(f"Testing combination {i+1}: {params}")
            
            try:
                # Build and train model
                model = model_builder_func(**params)
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                
                # Quick training for evaluation
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=20,  # Reduced epochs for quick evaluation
                    batch_size=32,
                    verbose=0
                )
                
                # Get validation score
                val_loss = min(history.history['val_loss'])
                
                results.append({
                    'params': params,
                    'val_loss': val_loss,
                    'history': history.history
                })
                
                if val_loss < best_score:
                    best_score = val_loss
                    best_params = params
                
                self.logger.info(f"  Validation loss: {val_loss:.6f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to evaluate params {params}: {str(e)}")
        
        self.best_params = best_params
        self.tuning_results['grid_search'] = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }
        
        self.logger.info(f"Grid search completed. Best params: {best_params}")
        
        return self.tuning_results['grid_search']
    
    def random_search(self,
                     param_distributions: Dict[str, List],
                     X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_val: np.ndarray,
                     y_val: np.ndarray,
                     model_builder_func: Callable,
                     n_iter: int = 50) -> Dict[str, Any]:
        """
        Perform random search hyperparameter optimization.
        
        Args:
            param_distributions: Dictionary of parameter distributions
            X_train: Training input data
            y_train: Training target data
            X_val: Validation input data
            y_val: Validation target data
            model_builder_func: Function to build model with given parameters
            n_iter: Number of random iterations
        
        Returns:
            Dictionary with best parameters and results
        """
        self.logger.info(f"Starting random search with {n_iter} iterations")
        
        results = []
        best_score = float('inf')
        best_params = None
        
        for i in range(n_iter):
            # Sample random parameters
            params = {}
            for param, values in param_distributions.items():
                params[param] = np.random.choice(values)
            
            self.logger.info(f"Testing iteration {i+1}: {params}")
            
            try:
                # Build and train model
                model = model_builder_func(**params)
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                
                # Quick training for evaluation
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=20,
                    batch_size=32,
                    verbose=0
                )
                
                # Get validation score
                val_loss = min(history.history['val_loss'])
                
                results.append({
                    'params': params,
                    'val_loss': val_loss,
                    'history': history.history
                })
                
                if val_loss < best_score:
                    best_score = val_loss
                    best_params = params
                
                self.logger.info(f"  Validation loss: {val_loss:.6f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to evaluate params {params}: {str(e)}")
        
        self.best_params = best_params
        self.tuning_results['random_search'] = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }
        
        self.logger.info(f"Random search completed. Best params: {best_params}")
        
        return self.tuning_results['random_search']
    
    def bayesian_optimization(self,
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_val: np.ndarray,
                            y_val: np.ndarray,
                            model_builder_func: Callable) -> Dict[str, Any]:
        """
        Perform Bayesian optimization using Optuna.
        
        Args:
            X_train: Training input data
            y_train: Training target data
            X_val: Validation input data
            y_val: Validation target data
            model_builder_func: Function to build model with given parameters
        
        Returns:
            Dictionary with best parameters and results
        """
        self.logger.info(f"Starting Bayesian optimization with {self.n_trials} trials")
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'lstm_units': trial.suggest_categorical('lstm_units', 
                    [[32, 16], [64, 32], [128, 64], [64, 32, 16]]),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'sequence_length': trial.suggest_int('sequence_length', 30, 120)
            }
            
            try:
                # Build and train model
                model = model_builder_func(**params)
                
                # Compile with suggested learning rate
                optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
                model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
                
                # Quick training for evaluation
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=30,
                    batch_size=params['batch_size'],
                    verbose=0,
                    callbacks=[
                        EarlyStopping(patience=5, restore_best_weights=True)
                    ]
                )
                
                # Return the best validation loss
                return min(history.history['val_loss'])
                
            except Exception as e:
                self.logger.warning(f"Trial failed: {str(e)}")
                return float('inf')
        
        # Create and run study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.study = study
        self.best_params = study.best_params
        
        self.tuning_results['bayesian_optimization'] = {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'study_summary': {
                'best_trial': study.best_trial.number,
                'best_value': study.best_value
            }
        }
        
        self.logger.info(f"Bayesian optimization completed. Best params: {study.best_params}")
        
        return self.tuning_results['bayesian_optimization']
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive optimization summary.
        
        Returns:
            Dictionary with optimization results summary
        """
        summary = {
            'methods_used': list(self.tuning_results.keys()),
            'best_overall_params': self.best_params,
            'optimization_results': self.tuning_results
        }
        
        if self.study:
            summary['optuna_study_info'] = {
                'n_trials': len(self.study.trials),
                'best_value': self.study.best_value,
                'best_trial_number': self.study.best_trial.number
            }
        
        return summary
    
    def save_tuning_results(self, filepath: str) -> None:
        """
        Save hyperparameter tuning results.
        
        Args:
            filepath: Path to save the results
        """
        summary = self.get_optimization_summary()
        save_json(summary, filepath)
        self.logger.info(f"Tuning results saved to {filepath}")


def create_advanced_callbacks():
    """
    Create advanced custom callbacks for specialized training needs.
    
    Returns:
        Dictionary of advanced callbacks
    """
    
    class GradientLoggingCallback(keras.callbacks.Callback):
        """Log gradient norms during training."""
        
        def __init__(self, log_frequency=10):
            super().__init__()
            self.log_frequency = log_frequency
        
        def on_batch_end(self, batch, logs=None):
            if batch % self.log_frequency == 0:
                gradients = []
                for layer in self.model.layers:
                    if hasattr(layer, 'kernel'):
                        gradient = tf.keras.backend.gradients(
                            self.model.total_loss, layer.kernel
                        )[0]
                        if gradient is not None:
                            grad_norm = tf.norm(gradient)
                            gradients.append(grad_norm)
                
                if gradients:
                    avg_grad_norm = tf.reduce_mean(gradients)
                    logs = logs or {}
                    logs['avg_gradient_norm'] = float(avg_grad_norm)
    
    class LearningRateScheduler(keras.callbacks.Callback):
        """Custom learning rate scheduler."""
        
        def __init__(self, schedule_func):
            super().__init__()
            self.schedule_func = schedule_func
        
        def on_epoch_begin(self, epoch, logs=None):
            lr = self.schedule_func(epoch)
            keras.backend.set_value(self.model.optimizer.lr, lr)
    
    def warmup_cosine_decay(epoch, warmup_epochs=10, total_epochs=100, 
                           initial_lr=1e-3, min_lr=1e-6):
        """Warmup followed by cosine decay learning rate schedule."""
        if epoch < warmup_epochs:
            return initial_lr * (epoch + 1) / warmup_epochs
        else:
            cosine_decay = 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
            return min_lr + (initial_lr - min_lr) * cosine_decay
    
    return {
        'gradient_logging': GradientLoggingCallback,
        'lr_scheduler': LearningRateScheduler,
        'warmup_cosine_schedule': warmup_cosine_decay
    }
