"""
Unit tests for model trainer module.
"""

import pytest
import numpy as np
import tensorflow as tf
import os
import sys
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_trainer import ModelTrainer, HyperparameterTuner
from lstm_model import LSTMGoldPredictor
from test_utils import (
    TestConfigFactory, TestDataGenerator,
    preprocessed_data, test_configs, trained_model
)


class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    def test_init(self, test_configs):
        """Test trainer initialization."""
        config = TestConfigFactory.create_training_config()
        trainer = ModelTrainer(config)
        
        assert trainer.config == config
        assert trainer.callbacks == []
        assert trainer.best_model is None
        assert trainer.training_history is None
    
    def test_setup_callbacks(self, test_configs, temp_dir):
        """Test callback setup."""
        config = TestConfigFactory.create_training_config(
            early_stopping=True,
            reduce_lr=True,
            patience=5,
            lr_patience=3
        )
        
        trainer = ModelTrainer(config)
        callbacks = trainer.setup_callbacks(
            checkpoint_dir=temp_dir,
            log_dir=temp_dir
        )
        
        assert len(callbacks) > 0
        
        # Check for specific callback types
        callback_types = [type(cb).__name__ for cb in callbacks]
        
        if config.early_stopping:
            assert 'EarlyStopping' in callback_types
        
        if config.reduce_lr:
            assert 'ReduceLROnPlateau' in callback_types
        
        # Should have ModelCheckpoint and CSVLogger
        assert 'ModelCheckpoint' in callback_types
        assert 'CSVLogger' in callback_types
    
    def test_create_custom_callbacks(self, test_configs):
        """Test custom callback creation."""
        config = TestConfigFactory.create_training_config()
        trainer = ModelTrainer(config)
        
        # Create learning rate scheduler
        lr_callback = trainer.create_lr_scheduler()
        assert lr_callback is not None
        assert isinstance(lr_callback, tf.keras.callbacks.LearningRateScheduler)
        
        # Create metrics logger
        metrics_callback = trainer.create_metrics_logger()
        assert metrics_callback is not None
    
    def test_train_model(self, preprocessed_data, test_configs, temp_dir):
        """Test model training."""
        X, y = preprocessed_data
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Create model
        model_config = test_configs['model']
        predictor = LSTMGoldPredictor(model_config)
        model = predictor.build_model(input_shape=X.shape[1:])
        
        # Create trainer
        training_config = TestConfigFactory.create_training_config(
            epochs=2,
            batch_size=16,
            verbose=0
        )
        trainer = ModelTrainer(training_config)
        
        # Train model
        history = trainer.train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            checkpoint_dir=temp_dir
        )
        
        assert history is not None
        assert 'loss' in history.history
        assert 'val_loss' in history.history
        assert trainer.training_history is not None
    
    def test_train_with_validation_split(self, preprocessed_data, test_configs, temp_dir):
        """Test training with validation split."""
        X, y = preprocessed_data
        
        # Create model
        model_config = test_configs['model']
        predictor = LSTMGoldPredictor(model_config)
        model = predictor.build_model(input_shape=X.shape[1:])
        
        # Create trainer
        training_config = TestConfigFactory.create_training_config(
            epochs=2,
            validation_split=0.2,
            verbose=0
        )
        trainer = ModelTrainer(training_config)
        
        # Train model with validation split
        history = trainer.train_model(
            model=model,
            X_train=X,
            y_train=y,
            checkpoint_dir=temp_dir
        )
        
        assert history is not None
        assert 'val_loss' in history.history
    
    def test_cross_validation(self, preprocessed_data, test_configs):
        """Test cross-validation training."""
        X, y = preprocessed_data
        
        # Create model config
        model_config = test_configs['model']
        
        # Create trainer
        training_config = TestConfigFactory.create_training_config(
            epochs=1,  # Keep low for testing
            verbose=0
        )
        trainer = ModelTrainer(training_config)
        
        # Perform cross-validation
        cv_scores = trainer.cross_validate(
            model_config=model_config,
            X=X,
            y=y,
            cv_folds=3,
            input_shape=X.shape[1:]
        )
        
        assert cv_scores is not None
        assert len(cv_scores) == 3  # Should have scores for each fold
        assert all(isinstance(score, dict) for score in cv_scores)
    
    def test_save_training_results(self, preprocessed_data, test_configs, temp_dir):
        """Test saving training results."""
        X, y = preprocessed_data
        
        # Create and train model
        model_config = test_configs['model']
        predictor = LSTMGoldPredictor(model_config)
        model = predictor.build_model(input_shape=X.shape[1:])
        
        training_config = TestConfigFactory.create_training_config(
            epochs=2,
            verbose=0
        )
        trainer = ModelTrainer(training_config)
        
        history = trainer.train_model(
            model=model,
            X_train=X,
            y_train=y,
            validation_split=0.2,
            checkpoint_dir=temp_dir
        )
        
        # Save results
        results_path = os.path.join(temp_dir, 'training_results.json')
        trainer.save_training_results(results_path, model_config.__dict__)
        
        assert os.path.exists(results_path)
        
        # Load and verify results
        import json
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        assert 'training_config' in results
        assert 'model_config' in results
        assert 'history' in results
        assert 'final_metrics' in results


class TestHyperparameterTuner:
    """Test cases for HyperparameterTuner class."""
    
    def test_init(self):
        """Test tuner initialization."""
        tuner = HyperparameterTuner()
        
        assert tuner.best_params is None
        assert tuner.best_score is None
        assert tuner.tuning_results == []
    
    def test_define_search_space(self):
        """Test search space definition."""
        tuner = HyperparameterTuner()
        
        search_space = tuner.define_search_space()
        
        assert isinstance(search_space, dict)
        assert 'lstm_units' in search_space
        assert 'dense_units' in search_space
        assert 'dropout_rate' in search_space
        assert 'learning_rate' in search_space
        assert 'batch_size' in search_space
    
    def test_create_model_from_params(self, preprocessed_data):
        """Test model creation from parameters."""
        X, y = preprocessed_data
        
        tuner = HyperparameterTuner()
        
        params = {
            'model_type': 'simple_lstm',
            'lstm_units': [64],
            'dense_units': [32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'loss_function': 'mse'
        }
        
        model = tuner.create_model_from_params(params, input_shape=X.shape[1:])
        
        assert model is not None
        assert isinstance(model, tf.keras.Model)
        assert model.input_shape[1:] == X.shape[1:]
    
    def test_evaluate_params(self, preprocessed_data):
        """Test parameter evaluation."""
        X, y = preprocessed_data
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        tuner = HyperparameterTuner()
        
        params = {
            'model_type': 'simple_lstm',
            'lstm_units': [32],
            'dense_units': [16],
            'dropout_rate': 0.1,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'loss_function': 'mse',
            'batch_size': 16,
            'epochs': 2
        }
        
        score = tuner.evaluate_params(
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            input_shape=X.shape[1:]
        )
        
        assert score is not None
        assert isinstance(score, (int, float))
        assert not np.isnan(score)
    
    def test_grid_search(self, preprocessed_data):
        """Test grid search hyperparameter tuning."""
        X, y = preprocessed_data
        
        # Use small dataset for testing
        X_small, y_small = X[:30], y[:30]
        
        tuner = HyperparameterTuner()
        
        # Define small search space for testing
        param_grid = {
            'lstm_units': [[32], [64]],
            'dropout_rate': [0.1, 0.2],
            'learning_rate': [0.001, 0.01]
        }
        
        best_params, best_score = tuner.grid_search(
            param_grid=param_grid,
            X_train=X_small,
            y_train=y_small,
            X_val=X_small,
            y_val=y_small,
            input_shape=X.shape[1:],
            epochs=1,
            verbose=0
        )
        
        assert best_params is not None
        assert best_score is not None
        assert isinstance(best_params, dict)
        assert 'lstm_units' in best_params
    
    def test_random_search(self, preprocessed_data):
        """Test random search hyperparameter tuning."""
        X, y = preprocessed_data
        
        # Use small dataset for testing
        X_small, y_small = X[:30], y[:30]
        
        tuner = HyperparameterTuner()
        
        # Define search space
        search_space = {
            'lstm_units': [[32], [64], [128]],
            'dropout_rate': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.01, 0.1]
        }
        
        best_params, best_score = tuner.random_search(
            search_space=search_space,
            X_train=X_small,
            y_train=y_small,
            X_val=X_small,
            y_val=y_small,
            input_shape=X.shape[1:],
            n_trials=3,
            epochs=1,
            verbose=0
        )
        
        assert best_params is not None
        assert best_score is not None
        assert len(tuner.tuning_results) == 3
    
    @patch('optuna.create_study')
    def test_bayesian_optimization(self, mock_study, preprocessed_data):
        """Test Bayesian optimization hyperparameter tuning."""
        X, y = preprocessed_data
        X_small, y_small = X[:20], y[:20]
        
        # Mock optuna study
        mock_trial = MagicMock()
        mock_trial.suggest_categorical.return_value = [64]
        mock_trial.suggest_float.return_value = 0.001
        mock_trial.suggest_int.return_value = 16
        
        mock_study_instance = MagicMock()
        mock_study_instance.best_params = {'lstm_units': [64], 'learning_rate': 0.001}
        mock_study_instance.best_value = 0.1
        mock_study.return_value = mock_study_instance
        
        tuner = HyperparameterTuner()
        
        # This should work with mocked optuna
        try:
            best_params, best_score = tuner.bayesian_optimization(
                X_train=X_small,
                y_train=y_small,
                X_val=X_small,
                y_val=y_small,
                input_shape=X.shape[1:],
                n_trials=2,
                epochs=1
            )
            
            assert best_params is not None
            assert best_score is not None
        except ImportError:
            # Optuna not installed, skip test
            pytest.skip("Optuna not installed")
    
    def test_save_tuning_results(self, temp_dir):
        """Test saving tuning results."""
        tuner = HyperparameterTuner()
        
        # Add some dummy results
        tuner.tuning_results = [
            {'params': {'lstm_units': [32]}, 'score': 0.1},
            {'params': {'lstm_units': [64]}, 'score': 0.05}
        ]
        tuner.best_params = {'lstm_units': [64]}
        tuner.best_score = 0.05
        
        results_path = os.path.join(temp_dir, 'tuning_results.json')
        tuner.save_results(results_path)
        
        assert os.path.exists(results_path)
        
        # Load and verify
        import json
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        assert 'best_params' in results
        assert 'best_score' in results
        assert 'all_results' in results
        assert len(results['all_results']) == 2
    
    def test_load_tuning_results(self, temp_dir):
        """Test loading tuning results."""
        tuner = HyperparameterTuner()
        
        # Create test results file
        results = {
            'best_params': {'lstm_units': [64]},
            'best_score': 0.05,
            'all_results': [
                {'params': {'lstm_units': [32]}, 'score': 0.1}
            ]
        }
        
        results_path = os.path.join(temp_dir, 'tuning_results.json')
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f)
        
        # Load results
        tuner.load_results(results_path)
        
        assert tuner.best_params == results['best_params']
        assert tuner.best_score == results['best_score']
        assert tuner.tuning_results == results['all_results']
    
    def test_get_best_model(self, preprocessed_data):
        """Test getting best model from tuning results."""
        X, y = preprocessed_data
        
        tuner = HyperparameterTuner()
        tuner.best_params = {
            'model_type': 'simple_lstm',
            'lstm_units': [64],
            'dense_units': [32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'loss_function': 'mse'
        }
        
        model = tuner.get_best_model(input_shape=X.shape[1:])
        
        assert model is not None
        assert isinstance(model, tf.keras.Model)
    
    def test_generate_tuning_report(self):
        """Test tuning report generation."""
        tuner = HyperparameterTuner()
        
        # Add dummy results
        tuner.tuning_results = [
            {'params': {'lstm_units': [32], 'learning_rate': 0.01}, 'score': 0.1},
            {'params': {'lstm_units': [64], 'learning_rate': 0.001}, 'score': 0.05},
            {'params': {'lstm_units': [128], 'learning_rate': 0.1}, 'score': 0.15}
        ]
        tuner.best_params = {'lstm_units': [64], 'learning_rate': 0.001}
        tuner.best_score = 0.05
        
        report = tuner.generate_report()
        
        assert isinstance(report, dict)
        assert 'best_params' in report
        assert 'best_score' in report
        assert 'total_trials' in report
        assert 'parameter_importance' in report
        
        assert report['total_trials'] == 3
        assert report['best_score'] == 0.05
    
    @pytest.mark.parametrize("method", ['grid', 'random'])
    def test_tuning_methods(self, preprocessed_data, method):
        """Test different tuning methods."""
        X, y = preprocessed_data
        X_small, y_small = X[:20], y[:20]
        
        tuner = HyperparameterTuner()
        
        if method == 'grid':
            param_space = {
                'lstm_units': [[32], [64]],
                'dropout_rate': [0.1, 0.2]
            }
            best_params, best_score = tuner.grid_search(
                param_grid=param_space,
                X_train=X_small,
                y_train=y_small,
                X_val=X_small,
                y_val=y_small,
                input_shape=X.shape[1:],
                epochs=1,
                verbose=0
            )
        else:  # random
            search_space = {
                'lstm_units': [[32], [64], [128]],
                'dropout_rate': [0.1, 0.2, 0.3]
            }
            best_params, best_score = tuner.random_search(
                search_space=search_space,
                X_train=X_small,
                y_train=y_small,
                X_val=X_small,
                y_val=y_small,
                input_shape=X.shape[1:],
                n_trials=2,
                epochs=1,
                verbose=0
            )
        
        assert best_params is not None
        assert best_score is not None
        assert isinstance(best_params, dict)
