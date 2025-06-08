"""
Integration tests for the complete gold price prediction pipeline.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import shutil
from unittest.mock import patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import GoldDataPreprocessor
from lstm_model import LSTMGoldPredictor
from model_trainer import ModelTrainer
from evaluation import ModelEvaluator
from prediction import GoldPricePredictor
from visualization import Visualizer
from test_utils import TestDataGenerator, TestConfigFactory


@pytest.mark.integration
class TestCompletePipeline:
    """Integration tests for the complete ML pipeline."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for integration tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_dataset(self, temp_workspace):
        """Create sample dataset for integration testing."""
        # Generate larger dataset for integration tests
        data = TestDataGenerator.create_sample_gold_data(n_samples=200)
        
        # Save to JSON file
        data_path = os.path.join(temp_workspace, 'gold_data.json')
        data.to_json(data_path, orient='records', date_format='iso')
        
        return data_path
    
    def test_end_to_end_pipeline(self, temp_workspace, sample_dataset):
        """Test complete end-to-end pipeline."""
        # 1. Data Preprocessing
        data_config = TestConfigFactory.create_data_config(
            data_path=sample_dataset,
            sequence_length=20,
            feature_engineering=True
        )
        
        preprocessor = GoldDataPreprocessor(data_config)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess()
        
        # Verify preprocessing outputs
        assert X_train is not None and len(X_train) > 0
        assert X_val is not None and len(X_val) > 0
        assert X_test is not None and len(X_test) > 0
        
        # 2. Model Building and Training
        model_config = TestConfigFactory.create_model_config(
            model_type='simple_lstm',
            lstm_units=[64, 32],
            dense_units=[16]
        )
        
        predictor = LSTMGoldPredictor(model_config)
        model = predictor.build_model(input_shape=X_train.shape[1:])
        
        training_config = TestConfigFactory.create_training_config(
            epochs=5,
            batch_size=32,
            early_stopping=True,
            patience=3
        )
        
        trainer = ModelTrainer(training_config)
        history = trainer.train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            checkpoint_dir=temp_workspace
        )
        
        # Verify training completed
        assert history is not None
        assert 'loss' in history.history
        
        # 3. Model Evaluation
        evaluator = ModelEvaluator(data_config)
        
        # Make predictions on test set
        test_predictions = model.predict(X_test)
        
        # Inverse transform if needed
        if preprocessor.scaler is not None:
            y_test_original = preprocessor.inverse_transform_target(y_test)
            test_predictions_original = preprocessor.inverse_transform_target(
                test_predictions.flatten()
            )
        else:
            y_test_original = y_test
            test_predictions_original = test_predictions.flatten()
        
        # Evaluate predictions
        eval_results = evaluator.evaluate_predictions(
            y_test_original, 
            test_predictions_original
        )
        
        # Verify evaluation results
        assert 'basic_metrics' in eval_results
        assert 'performance_classification' in eval_results
        
        # 4. Future Predictions
        prediction_config = TestConfigFactory.create_data_config()
        price_predictor = GoldPricePredictor(model, preprocessor, prediction_config)
        
        # Make future predictions
        future_predictions = price_predictor.predict_future(
            n_steps=5,
            confidence_interval=0.95
        )
        
        # Verify predictions
        assert 'predictions' in future_predictions
        assert len(future_predictions['predictions']) == 5
        
        # 5. Visualization (basic test)
        viz_config = TestConfigFactory.create_data_config()
        visualizer = Visualizer(viz_config)
        
        # Test that visualization methods can be called without errors
        try:
            fig = visualizer.plot_predictions(
                y_test_original[:20], 
                test_predictions_original[:20]
            )
            assert fig is not None
        except Exception as e:
            pytest.fail(f"Visualization failed: {str(e)}")
    
    def test_model_persistence(self, temp_workspace, sample_dataset):
        """Test model saving and loading in pipeline."""
        # Train a model
        data_config = TestConfigFactory.create_data_config(
            data_path=sample_dataset,
            sequence_length=15
        )
        
        preprocessor = GoldDataPreprocessor(data_config)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess()
        
        model_config = TestConfigFactory.create_model_config()
        predictor = LSTMGoldPredictor(model_config)
        model = predictor.build_model(input_shape=X_train.shape[1:])
        
        # Train briefly
        model.fit(X_train, y_train, epochs=2, verbose=0)
        
        # Save model and preprocessor
        model_path = os.path.join(temp_workspace, 'model.h5')
        scaler_path = os.path.join(temp_workspace, 'scaler.pkl')
        
        predictor.save_model(model_path)
        preprocessor.save_scaler(scaler_path)
        
        # Load model and preprocessor
        new_predictor = LSTMGoldPredictor(model_config)
        new_predictor.load_model(model_path)
        
        new_preprocessor = GoldDataPreprocessor(data_config)
        new_preprocessor.load_scaler(scaler_path)
        
        # Test that loaded model produces same predictions
        original_preds = model.predict(X_test[:5])
        loaded_preds = new_predictor.predict(X_test[:5])
        
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)
    
    @pytest.mark.slow
    def test_hyperparameter_optimization_integration(self, temp_workspace, sample_dataset):
        """Test hyperparameter optimization in full pipeline."""
        from model_trainer import HyperparameterTuner
        
        # Prepare data
        data_config = TestConfigFactory.create_data_config(
            data_path=sample_dataset,
            sequence_length=10
        )
        
        preprocessor = GoldDataPreprocessor(data_config)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess()
        
        # Use small search space for testing
        tuner = HyperparameterTuner()
        
        param_grid = {
            'lstm_units': [[32], [64]],
            'dropout_rate': [0.1, 0.2],
            'learning_rate': [0.001, 0.01]
        }
        
        # Perform grid search
        best_params, best_score = tuner.grid_search(
            param_grid=param_grid,
            X_train=X_train[:50],  # Use subset for speed
            y_train=y_train[:50],
            X_val=X_val[:20],
            y_val=y_val[:20],
            input_shape=X_train.shape[1:],
            epochs=2,
            verbose=0
        )
        
        assert best_params is not None
        assert best_score is not None
        
        # Build best model
        best_model = tuner.get_best_model(input_shape=X_train.shape[1:])
        assert best_model is not None
        
        # Train best model
        best_model.fit(X_train, y_train, epochs=2, verbose=0)
        
        # Evaluate best model
        test_preds = best_model.predict(X_test)
        
        evaluator = ModelEvaluator(data_config)
        results = evaluator.evaluate_predictions(y_test, test_preds.flatten())
        
        assert 'basic_metrics' in results
    
    def test_cross_validation_integration(self, temp_workspace, sample_dataset):
        """Test cross-validation in full pipeline."""
        # Prepare data
        data_config = TestConfigFactory.create_data_config(
            data_path=sample_dataset,
            sequence_length=15
        )
        
        preprocessor = GoldDataPreprocessor(data_config)
        data = preprocessor.load_data()
        cleaned_data = preprocessor.clean_data()
        
        if data_config.feature_engineering:
            # Convert date column for feature engineering
            cleaned_data['tanggal'] = pd.to_datetime(cleaned_data['tanggal'])
            cleaned_data = cleaned_data.sort_values('tanggal').reset_index(drop=True)
            featured_data = preprocessor.engineer_features()
        else:
            featured_data = cleaned_data
        
        # Normalize and create sequences
        normalized_data, scaler = preprocessor.normalize_data()
        X, y = preprocessor.create_sequences(normalized_data)
        
        # Perform cross-validation
        model_config = TestConfigFactory.create_model_config()
        training_config = TestConfigFactory.create_training_config(epochs=2)
        
        trainer = ModelTrainer(training_config)
        cv_results = trainer.cross_validate(
            model_config=model_config,
            X=X,
            y=y,
            cv_folds=3,
            input_shape=X.shape[1:]
        )
        
        assert cv_results is not None
        assert len(cv_results) == 3
        
        # Each fold should have evaluation results
        for fold_result in cv_results:
            assert isinstance(fold_result, dict)
            assert 'loss' in fold_result or 'mse' in fold_result
    
    def test_prediction_pipeline_with_confidence_intervals(self, temp_workspace, sample_dataset):
        """Test prediction pipeline with confidence intervals."""
        # Train model
        data_config = TestConfigFactory.create_data_config(
            data_path=sample_dataset,
            sequence_length=10
        )
        
        preprocessor = GoldDataPreprocessor(data_config)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess()
        
        model_config = TestConfigFactory.create_model_config()
        predictor = LSTMGoldPredictor(model_config)
        model = predictor.build_model(input_shape=X_train.shape[1:])
        
        # Quick training
        model.fit(X_train, y_train, epochs=3, verbose=0)
        
        # Create price predictor
        price_predictor = GoldPricePredictor(model, preprocessor, data_config)
        
        # Test single step prediction with confidence
        single_pred = price_predictor.predict_single_step(
            last_sequence=X_test[0],
            confidence_interval=0.95
        )
        
        assert 'prediction' in single_pred
        assert 'confidence_interval' in single_pred
        assert 'lower_bound' in single_pred['confidence_interval']
        assert 'upper_bound' in single_pred['confidence_interval']
        
        # Test multi-step prediction
        multi_pred = price_predictor.predict_future(
            n_steps=3,
            confidence_interval=0.90
        )
        
        assert 'predictions' in multi_pred
        assert len(multi_pred['predictions']) == 3
        assert 'confidence_intervals' in multi_pred
    
    def test_model_comparison_pipeline(self, temp_workspace, sample_dataset):
        """Test comparing different models in pipeline."""
        # Prepare data
        data_config = TestConfigFactory.create_data_config(
            data_path=sample_dataset,
            sequence_length=12
        )
        
        preprocessor = GoldDataPreprocessor(data_config)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess()
        
        # Train different models
        model_types = ['simple_lstm', 'stacked_lstm']
        model_results = {}
        
        for model_type in model_types:
            model_config = TestConfigFactory.create_model_config(
                model_type=model_type,
                lstm_units=[32] if model_type == 'simple_lstm' else [32, 16]
            )
            
            predictor = LSTMGoldPredictor(model_config)
            model = predictor.build_model(input_shape=X_train.shape[1:])
            
            # Train model
            model.fit(X_train, y_train, epochs=3, verbose=0)
            
            # Evaluate model
            test_preds = model.predict(X_test)
            
            evaluator = ModelEvaluator(data_config)
            results = evaluator.evaluate_predictions(y_test, test_preds.flatten())
            
            model_results[model_type] = results['basic_metrics']
        
        # Compare models
        evaluator = ModelEvaluator(data_config)
        comparison = evaluator.compare_models(model_results)
        
        assert 'ranking' in comparison
        assert 'best_model' in comparison
        assert len(comparison['ranking']) == 2
    
    def test_data_pipeline_robustness(self, temp_workspace):
        """Test data pipeline with various data quality issues."""
        # Create problematic dataset
        problematic_data = TestDataGenerator.create_sample_gold_data(n_samples=100)
        
        # Add data quality issues
        problematic_data.loc[10:15, 'hargaJual'] = np.nan  # Missing values
        problematic_data.loc[20, 'hargaJual'] = -1000  # Negative value
        problematic_data.loc[25, 'hargaJual'] = 5000000  # Extreme outlier
        problematic_data = pd.concat([problematic_data, problematic_data.iloc[0:5]], ignore_index=True)  # Duplicates
        
        # Save problematic data
        data_path = os.path.join(temp_workspace, 'problematic_data.json')
        problematic_data.to_json(data_path, orient='records', date_format='iso')
        
        # Test preprocessing handles issues gracefully
        data_config = TestConfigFactory.create_data_config(
            data_path=data_path,
            handle_missing='interpolate',
            outlier_method='iqr'
        )
        
        preprocessor = GoldDataPreprocessor(data_config)
        
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess()
            
            # Verify data is cleaned
            assert X_train is not None
            assert not np.isnan(X_train).any()
            assert not np.isnan(y_train).any()
            
            # Verify no extreme values remain
            assert np.all(y_train > 0)  # No negative prices
            
        except Exception as e:
            pytest.fail(f"Data preprocessing failed with problematic data: {str(e)}")
    
    @pytest.mark.slow
    def test_full_training_pipeline_with_callbacks(self, temp_workspace, sample_dataset):
        """Test complete training pipeline with all callbacks."""
        # Prepare data
        data_config = TestConfigFactory.create_data_config(
            data_path=sample_dataset
        )
        
        preprocessor = GoldDataPreprocessor(data_config)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess()
        
        # Create model
        model_config = TestConfigFactory.create_model_config()
        predictor = LSTMGoldPredictor(model_config)
        model = predictor.build_model(input_shape=X_train.shape[1:])
        
        # Create trainer with all callbacks
        training_config = TestConfigFactory.create_training_config(
            epochs=10,
            early_stopping=True,
            patience=3,
            reduce_lr=True,
            lr_patience=2,
            save_best_only=True
        )
        
        trainer = ModelTrainer(training_config)
        
        # Train with callbacks
        history = trainer.train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            checkpoint_dir=temp_workspace,
            log_dir=temp_workspace
        )
        
        # Verify training completed and callbacks worked
        assert history is not None
        assert len(history.history['loss']) <= 10  # May stop early
        
        # Check that model checkpoint was saved
        checkpoint_files = [f for f in os.listdir(temp_workspace) if f.endswith('.h5')]
        assert len(checkpoint_files) > 0
        
        # Check that CSV log was created
        csv_files = [f for f in os.listdir(temp_workspace) if f.endswith('.csv')]
        assert len(csv_files) > 0
