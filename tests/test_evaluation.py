"""
Unit tests for evaluation module.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys
from unittest.mock import patch, MagicMock
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation import ModelEvaluator
from test_utils import (
    TestConfigFactory, TestDataGenerator, PERFORMANCE_THRESHOLDS,
    assert_metrics_in_range, preprocessed_data, test_configs, trained_model
)


class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""
    
    def test_init(self):
        """Test evaluator initialization."""
        config = TestConfigFactory.create_data_config()
        evaluator = ModelEvaluator(config)
        
        assert evaluator.config == config
        assert evaluator.results == {}
    
    def test_calculate_basic_metrics(self):
        """Test basic metrics calculation."""
        evaluator = ModelEvaluator(TestConfigFactory.create_data_config())
        
        # Create test data
        y_true = np.array([100, 200, 150, 300, 250])
        y_pred = np.array([95, 210, 140, 295, 260])
        
        metrics = evaluator.calculate_basic_metrics(y_true, y_pred)
        
        assert isinstance(metrics, dict)
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'mape' in metrics
        assert 'r2' in metrics
        
        # Check that metrics are reasonable
        assert metrics['mse'] > 0
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
        assert metrics['mape'] >= 0
        assert -1 <= metrics['r2'] <= 1
    
    def test_calculate_advanced_metrics(self):
        """Test advanced metrics calculation."""
        evaluator = ModelEvaluator(TestConfigFactory.create_data_config())
        
        y_true = np.array([100, 200, 150, 300, 250])
        y_pred = np.array([95, 210, 140, 295, 260])
        
        metrics = evaluator.calculate_advanced_metrics(y_true, y_pred)
        
        assert isinstance(metrics, dict)
        assert 'direction_accuracy' in metrics
        assert 'max_error' in metrics
        assert 'explained_variance' in metrics
        assert 'median_absolute_error' in metrics
        
        # Check ranges
        assert 0 <= metrics['direction_accuracy'] <= 1
        assert metrics['max_error'] >= 0
    
    def test_directional_accuracy(self):
        """Test directional accuracy calculation."""
        evaluator = ModelEvaluator(TestConfigFactory.create_data_config())
        
        # Perfect directional accuracy
        y_true = np.array([100, 150, 200, 180, 220])
        y_pred = np.array([95, 145, 195, 175, 215])
        
        accuracy = evaluator.calculate_directional_accuracy(y_true, y_pred)
        assert accuracy == 1.0
        
        # No directional accuracy
        y_true = np.array([100, 150, 200, 180, 220])
        y_pred = np.array([105, 140, 190, 185, 210])
        
        accuracy = evaluator.calculate_directional_accuracy(y_true, y_pred)
        assert accuracy == 0.0
    
    def test_calculate_residuals(self):
        """Test residual calculation and analysis."""
        evaluator = ModelEvaluator(TestConfigFactory.create_data_config())
        
        y_true = np.array([100, 200, 150, 300, 250])
        y_pred = np.array([95, 210, 140, 295, 260])
        
        residuals = evaluator.calculate_residuals(y_true, y_pred)
        
        assert isinstance(residuals, dict)
        assert 'residuals' in residuals
        assert 'standardized_residuals' in residuals
        assert 'residual_stats' in residuals
        
        assert len(residuals['residuals']) == len(y_true)
        assert len(residuals['standardized_residuals']) == len(y_true)
        
        # Check residual stats
        stats = residuals['residual_stats']
        assert 'mean' in stats
        assert 'std' in stats
        assert 'skewness' in stats
        assert 'kurtosis' in stats
    
    def test_perform_statistical_tests(self):
        """Test statistical significance tests."""
        evaluator = ModelEvaluator(TestConfigFactory.create_data_config())
        
        y_true = np.random.normal(100, 10, 50)
        y_pred = y_true + np.random.normal(0, 5, 50)  # Add some noise
        
        test_results = evaluator.perform_statistical_tests(y_true, y_pred)
        
        assert isinstance(test_results, dict)
        assert 'normality_test' in test_results
        assert 'autocorrelation_test' in test_results
        assert 'heteroscedasticity_test' in test_results
        
        # Each test should have statistic and p-value
        for test_name, test_result in test_results.items():
            assert 'statistic' in test_result
            assert 'p_value' in test_result
    
    def test_classification_performance(self):
        """Test performance classification."""
        evaluator = ModelEvaluator(TestConfigFactory.create_data_config())
        
        metrics = {
            'mape': 2.0,  # Excellent
            'r2': 0.85,   # Good
            'direction_accuracy': 0.75  # Good
        }
        
        classification = evaluator.classify_performance(metrics)
        
        assert isinstance(classification, dict)
        assert 'overall' in classification
        assert 'mape_class' in classification
        assert 'r2_class' in classification
        assert 'direction_class' in classification
        
        assert classification['mape_class'] == 'Excellent'
        assert classification['r2_class'] == 'Good'
    
    def test_cross_validation_evaluation(self, preprocessed_data):
        """Test cross-validation evaluation."""
        X, y = preprocessed_data
        
        evaluator = ModelEvaluator(TestConfigFactory.create_data_config())
        
        # Mock model that returns predictions
        def mock_model_func(X_train, y_train, X_val):
            # Simple model that returns mean of training data
            return np.full(len(X_val), np.mean(y_train))
        
        cv_results = evaluator.cross_validate_model(
            X=X,
            y=y,
            model_func=mock_model_func,
            cv_folds=3
        )
        
        assert isinstance(cv_results, dict)
        assert 'fold_scores' in cv_results
        assert 'mean_scores' in cv_results
        assert 'std_scores' in cv_results
        
        assert len(cv_results['fold_scores']) == 3
        
        # Check that mean scores contain expected metrics
        mean_scores = cv_results['mean_scores']
        assert 'mse' in mean_scores
        assert 'mae' in mean_scores
        assert 'r2' in mean_scores
    
    def test_walk_forward_validation(self, preprocessed_data):
        """Test walk-forward validation."""
        X, y = preprocessed_data
        
        evaluator = ModelEvaluator(TestConfigFactory.create_data_config())
        
        def mock_model_func(X_train, y_train, X_test):
            return np.full(len(X_test), np.mean(y_train))
        
        wf_results = evaluator.walk_forward_validation(
            X=X,
            y=y,
            model_func=mock_model_func,
            initial_train_size=30,
            step_size=5
        )
        
        assert isinstance(wf_results, dict)
        assert 'step_scores' in wf_results
        assert 'overall_scores' in wf_results
        
        assert len(wf_results['step_scores']) > 0
        
        # Each step should have predictions and metrics
        for step in wf_results['step_scores']:
            assert 'predictions' in step
            assert 'actual' in step
            assert 'metrics' in step
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        evaluator = ModelEvaluator(TestConfigFactory.create_data_config())
        
        # Create mock model results
        model_results = {
            'LSTM_Simple': {
                'mse': 100,
                'mae': 8,
                'r2': 0.85,
                'mape': 2.5
            },
            'LSTM_Stacked': {
                'mse': 80,
                'mae': 7,
                'r2': 0.88,
                'mape': 2.2
            },
            'LSTM_Attention': {
                'mse': 75,
                'mae': 6.5,
                'r2': 0.90,
                'mape': 2.0
            }
        }
        
        comparison = evaluator.compare_models(model_results)
        
        assert isinstance(comparison, dict)
        assert 'ranking' in comparison
        assert 'best_model' in comparison
        assert 'improvement_analysis' in comparison
        
        # Best model should be the one with lowest MAPE
        assert comparison['best_model'] == 'LSTM_Attention'
        
        # Ranking should be in order of performance
        ranking = comparison['ranking']
        assert ranking[0]['model'] == 'LSTM_Attention'
        assert ranking[-1]['model'] == 'LSTM_Simple'
    
    def test_evaluate_predictions(self):
        """Test comprehensive prediction evaluation."""
        evaluator = ModelEvaluator(TestConfigFactory.create_data_config())
        
        # Create realistic test data
        np.random.seed(42)
        y_true = 750000 + np.random.normal(0, 25000, 100)  # Gold prices around 750k
        y_pred = y_true + np.random.normal(0, 15000, 100)  # Add prediction error
        
        results = evaluator.evaluate_predictions(y_true, y_pred)
        
        assert isinstance(results, dict)
        assert 'basic_metrics' in results
        assert 'advanced_metrics' in results
        assert 'residual_analysis' in results
        assert 'statistical_tests' in results
        assert 'performance_classification' in results
        
        # Check that results are stored
        assert evaluator.results == results
    
    def test_generate_evaluation_report(self):
        """Test evaluation report generation."""
        evaluator = ModelEvaluator(TestConfigFactory.create_data_config())
        
        # Set some mock results
        evaluator.results = {
            'basic_metrics': {
                'mse': 100,
                'rmse': 10,
                'mae': 8,
                'mape': 2.5,
                'r2': 0.85
            },
            'advanced_metrics': {
                'direction_accuracy': 0.75,
                'max_error': 20
            },
            'performance_classification': {
                'overall': 'Good',
                'mape_class': 'Excellent'
            }
        }
        
        report = evaluator.generate_report()
        
        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'detailed_metrics' in report
        assert 'recommendations' in report
        
        summary = report['summary']
        assert 'model_performance' in summary
        assert 'key_metrics' in summary
    
    def test_save_and_load_results(self, temp_dir):
        """Test saving and loading evaluation results."""
        evaluator = ModelEvaluator(TestConfigFactory.create_data_config())
        
        # Set some test results
        test_results = {
            'basic_metrics': {'mse': 100, 'mae': 8},
            'advanced_metrics': {'direction_accuracy': 0.75}
        }
        evaluator.results = test_results
        
        # Save results
        results_path = os.path.join(temp_dir, 'evaluation_results.json')
        evaluator.save_results(results_path)
        
        assert os.path.exists(results_path)
        
        # Load results into new evaluator
        new_evaluator = ModelEvaluator(TestConfigFactory.create_data_config())
        new_evaluator.load_results(results_path)
        
        assert new_evaluator.results == test_results
    
    def test_error_metrics_edge_cases(self):
        """Test error metrics with edge cases."""
        evaluator = ModelEvaluator(TestConfigFactory.create_data_config())
        
        # Test with perfect predictions
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200, 300])
        
        metrics = evaluator.calculate_basic_metrics(y_true, y_pred)
        
        assert metrics['mse'] == 0
        assert metrics['rmse'] == 0
        assert metrics['mae'] == 0
        assert metrics['r2'] == 1.0
        
        # Test with zero true values (should handle MAPE gracefully)
        y_true = np.array([0, 100, 200])
        y_pred = np.array([5, 95, 205])
        
        metrics = evaluator.calculate_basic_metrics(y_true, y_pred)
        
        # MAPE should be calculated only for non-zero values
        assert not np.isnan(metrics['mape'])
        assert not np.isinf(metrics['mape'])
    
    def test_residual_analysis_comprehensive(self):
        """Test comprehensive residual analysis."""
        evaluator = ModelEvaluator(TestConfigFactory.create_data_config())
        
        # Create data with known residual pattern
        n_points = 100
        y_true = np.linspace(100, 200, n_points)
        # Add heteroscedastic noise (increasing variance)
        noise = np.random.normal(0, np.linspace(1, 10, n_points))
        y_pred = y_true + noise
        
        residuals = evaluator.calculate_residuals(y_true, y_pred)
        
        # Check residual statistics
        stats = residuals['residual_stats']
        assert abs(stats['mean']) < 5  # Should be close to zero
        assert stats['std'] > 0
        
        # Check for standardized residuals
        std_residuals = residuals['standardized_residuals']
        assert abs(np.mean(std_residuals)) < 0.5
        assert abs(np.std(std_residuals) - 1.0) < 0.3
    
    def test_performance_thresholds(self):
        """Test performance classification thresholds."""
        evaluator = ModelEvaluator(TestConfigFactory.create_data_config())
        
        # Test excellent performance
        excellent_metrics = {
            'mape': 1.5,
            'r2': 0.95,
            'direction_accuracy': 0.85
        }
        
        classification = evaluator.classify_performance(excellent_metrics)
        assert classification['mape_class'] == 'Excellent'
        assert classification['r2_class'] == 'Excellent'
        assert classification['direction_class'] == 'Good'
        
        # Test poor performance
        poor_metrics = {
            'mape': 15.0,
            'r2': 0.3,
            'direction_accuracy': 0.45
        }
        
        classification = evaluator.classify_performance(poor_metrics)
        assert classification['mape_class'] == 'Poor'
        assert classification['r2_class'] == 'Poor'
        assert classification['direction_class'] == 'Poor'
    
    @pytest.mark.parametrize("metric_name,values", [
        ('mse', [100, 50, 25]),
        ('mae', [10, 8, 5]),
        ('mape', [5.0, 2.5, 1.0]),
        ('r2', [0.7, 0.85, 0.95])
    ])
    def test_metric_calculations(self, metric_name, values):
        """Test individual metric calculations."""
        evaluator = ModelEvaluator(TestConfigFactory.create_data_config())
        
        # Create test data for each metric value
        y_true = np.array([100, 200, 150])
        
        for target_value in values:
            if metric_name == 'mse':
                # Create y_pred to achieve target MSE
                y_pred = y_true + np.sqrt(target_value)
            elif metric_name == 'mae':
                # Create y_pred to achieve target MAE
                y_pred = y_true + target_value
            elif metric_name == 'mape':
                # Create y_pred to achieve target MAPE
                y_pred = y_true * (1 + target_value/100)
            elif metric_name == 'r2':
                # Create y_pred to achieve target R²
                if target_value >= 0.95:
                    y_pred = y_true + np.random.normal(0, 1, len(y_true))
                else:
                    y_pred = y_true + np.random.normal(0, 20, len(y_true))
            
            metrics = evaluator.calculate_basic_metrics(y_true, y_pred)
            
            # Check that calculated metric is reasonable
            if metric_name in metrics:
                calculated_value = metrics[metric_name]
                assert not np.isnan(calculated_value)
                assert not np.isinf(calculated_value)
    
    def test_batch_evaluation(self):
        """Test evaluating multiple prediction sets."""
        evaluator = ModelEvaluator(TestConfigFactory.create_data_config())
        
        # Create multiple prediction sets
        prediction_sets = {
            'train': {
                'y_true': np.array([100, 200, 150, 300]),
                'y_pred': np.array([95, 205, 145, 295])
            },
            'validation': {
                'y_true': np.array([180, 250, 220]),
                'y_pred': np.array([175, 255, 225])
            },
            'test': {
                'y_true': np.array([120, 280, 190, 240]),
                'y_pred': np.array([115, 275, 195, 245])
            }
        }
        
        batch_results = evaluator.evaluate_batch(prediction_sets)
        
        assert isinstance(batch_results, dict)
        assert 'train' in batch_results
        assert 'validation' in batch_results
        assert 'test' in batch_results
        
        # Each set should have complete evaluation
        for set_name, results in batch_results.items():
            assert 'basic_metrics' in results
            assert 'advanced_metrics' in results
            assert 'performance_classification' in results
