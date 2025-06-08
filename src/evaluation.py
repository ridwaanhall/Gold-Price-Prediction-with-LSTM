"""
Model Evaluation Module for Gold Price Prediction LSTM Model
Author: ridwaanhall
Date: 2025-06-08
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import TimeSeriesSplit, KFold
from scipy import stats
from scipy.stats import jarque_bera, shapiro, normaltest
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

from .utils import setup_logging, save_json, format_number


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAPE value as percentage
    """
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return float('inf')
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        SMAPE value as percentage
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    if not np.any(mask):
        return float('inf')
    
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy of predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Directional accuracy as percentage
    """
    if len(y_true) < 2:
        return 0.0
    
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred.flatten()))
    
    return np.mean(true_direction == pred_direction) * 100


class ModelEvaluator:
    """
    Comprehensive model evaluation class with advanced metrics and analysis.
    
    This class provides various evaluation metrics, statistical tests,
    and diagnostic tools for LSTM model assessment.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model evaluator.
        
        Args:
            config: Configuration dictionary with evaluation parameters
        """
        self.config = config or {}
        self.logger = setup_logging()
        
        # Evaluation configuration
        self.confidence_level = self.config.get('confidence_level', 0.95)
        self.bootstrap_samples = self.config.get('bootstrap_samples', 1000)
        self.significance_level = self.config.get('significance_level', 0.05)
        
        # Results storage
        self.evaluation_results = {}
        self.predictions_data = {}
        
        self.logger.info("ModelEvaluator initialized")
    
    def calculate_metrics(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         return_confidence_intervals: bool = True) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            return_confidence_intervals: Whether to calculate confidence intervals
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Calculating evaluation metrics")
        
        # Flatten predictions if needed
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()
        
        # Basic metrics
        metrics = {
            'rmse': calculate_rmse(y_true, y_pred),
            'mae': calculate_mae(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'mape': calculate_mape(y_true, y_pred),
            'smape': calculate_smape(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'direction_accuracy': direction_accuracy(y_true, y_pred)
        }
        
        # Additional statistical metrics
        residuals = y_true - y_pred
        metrics.update({
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'min_residual': np.min(residuals),
            'max_residual': np.max(residuals),
            'median_absolute_error': np.median(np.abs(residuals)),
            'explained_variance': self._explained_variance_score(y_true, y_pred),
            'max_error': np.max(np.abs(residuals))
        })
        
        # Relative metrics
        if np.mean(y_true) != 0:
            metrics['relative_rmse'] = metrics['rmse'] / np.mean(y_true) * 100
            metrics['relative_mae'] = metrics['mae'] / np.mean(y_true) * 100
        
        # Theil's U statistic (for time series)
        metrics['theil_u'] = self._calculate_theil_u(y_true, y_pred)
        
        # Calculate confidence intervals if requested
        if return_confidence_intervals:
            ci_results = self._bootstrap_confidence_intervals(y_true, y_pred)
            metrics['confidence_intervals'] = ci_results
        
        self.logger.info(f"Calculated {len(metrics)} evaluation metrics")
        
        return metrics
    
    def _explained_variance_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate explained variance score."""
        y_true_var = np.var(y_true)
        if y_true_var == 0:
            return 0.0
        
        residual_var = np.var(y_true - y_pred)
        return 1 - (residual_var / y_true_var)
    
    def _calculate_theil_u(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Theil's U statistic."""
        if len(y_true) < 2:
            return float('inf')
        
        # Theil's U = sqrt(MSE) / sqrt(mean((y_true[1:] - y_true[:-1])**2))
        mse = mean_squared_error(y_true, y_pred)
        naive_forecast_mse = np.mean((y_true[1:] - y_true[:-1]) ** 2)
        
        if naive_forecast_mse == 0:
            return float('inf')
        
        return np.sqrt(mse) / np.sqrt(naive_forecast_mse)
    
    def _bootstrap_confidence_intervals(self, 
                                      y_true: np.ndarray, 
                                      y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Calculate bootstrap confidence intervals for metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            Dictionary with confidence intervals for each metric
        """
        n_samples = len(y_true)
        bootstrap_metrics = {
            'rmse': [], 'mae': [], 'mape': [], 'r2': [], 'direction_accuracy': []
        }
        
        for _ in range(self.bootstrap_samples):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Calculate metrics for bootstrap sample
            bootstrap_metrics['rmse'].append(calculate_rmse(y_true_boot, y_pred_boot))
            bootstrap_metrics['mae'].append(calculate_mae(y_true_boot, y_pred_boot))
            bootstrap_metrics['mape'].append(calculate_mape(y_true_boot, y_pred_boot))
            bootstrap_metrics['r2'].append(r2_score(y_true_boot, y_pred_boot))
            bootstrap_metrics['direction_accuracy'].append(direction_accuracy(y_true_boot, y_pred_boot))
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        confidence_intervals = {}
        
        for metric, values in bootstrap_metrics.items():
            lower = np.percentile(values, (alpha/2) * 100)
            upper = np.percentile(values, (1 - alpha/2) * 100)
            confidence_intervals[metric] = {
                'lower': lower,
                'upper': upper,
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        return confidence_intervals
    
    def residual_analysis(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         save_plots: bool = True,
                         output_dir: str = "plots/") -> Dict[str, Any]:
        """
        Perform comprehensive residual analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_plots: Whether to save diagnostic plots
            output_dir: Directory to save plots
        
        Returns:
            Dictionary with residual analysis results
        """
        self.logger.info("Performing residual analysis")
        
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()
        
        residuals = y_true - y_pred
        standardized_residuals = residuals / np.std(residuals)
        
        # Statistical tests
        analysis_results = {
            'residual_statistics': {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'min': np.min(residuals),
                'max': np.max(residuals),
                'q25': np.percentile(residuals, 25),
                'q50': np.percentile(residuals, 50),
                'q75': np.percentile(residuals, 75),
                'skewness': stats.skew(residuals),
                'kurtosis': stats.kurtosis(residuals)
            }
        }
        
        # Normality tests
        try:
            shapiro_stat, shapiro_p = shapiro(residuals)
            jb_stat, jb_p = jarque_bera(residuals)
            ks_stat, ks_p = normaltest(residuals)
            
            analysis_results['normality_tests'] = {
                'shapiro_wilk': {'statistic': shapiro_stat, 'p_value': shapiro_p},
                'jarque_bera': {'statistic': jb_stat, 'p_value': jb_p},
                'kolmogorov_smirnov': {'statistic': ks_stat, 'p_value': ks_p}
            }
        except Exception as e:
            self.logger.warning(f"Error in normality tests: {str(e)}")
            analysis_results['normality_tests'] = {'error': str(e)}
        
        # Autocorrelation test (Durbin-Watson)
        try:
            dw_statistic = self._durbin_watson_test(residuals)
            analysis_results['autocorrelation'] = {
                'durbin_watson': dw_statistic,
                'interpretation': self._interpret_durbin_watson(dw_statistic)
            }
        except Exception as e:
            self.logger.warning(f"Error in autocorrelation test: {str(e)}")
        
        # Heteroscedasticity analysis
        analysis_results['heteroscedasticity'] = self._test_heteroscedasticity(y_pred, residuals)
        
        # Outlier detection
        analysis_results['outliers'] = self._detect_outliers_in_residuals(residuals, standardized_residuals)
        
        # Create diagnostic plots if requested
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            self._create_residual_plots(y_true, y_pred, residuals, output_dir)
        
        return analysis_results
    
    def _durbin_watson_test(self, residuals: np.ndarray) -> float:
        """Calculate Durbin-Watson test statistic."""
        diff_residuals = np.diff(residuals)
        return np.sum(diff_residuals**2) / np.sum(residuals**2)
    
    def _interpret_durbin_watson(self, dw_stat: float) -> str:
        """Interpret Durbin-Watson test result."""
        if dw_stat < 1.5:
            return "Positive autocorrelation"
        elif dw_stat > 2.5:
            return "Negative autocorrelation"
        else:
            return "No significant autocorrelation"
    
    def _test_heteroscedasticity(self, fitted_values: np.ndarray, residuals: np.ndarray) -> Dict[str, Any]:
        """Test for heteroscedasticity in residuals."""
        try:
            # Breusch-Pagan test (simplified)
            residuals_squared = residuals ** 2
            correlation = np.corrcoef(fitted_values, residuals_squared)[0, 1]
            
            # White's test approximation
            fitted_squared = fitted_values ** 2
            white_correlation = np.corrcoef(fitted_squared, residuals_squared)[0, 1]
            
            return {
                'breusch_pagan_correlation': correlation,
                'white_test_correlation': white_correlation,
                'interpretation': 'Heteroscedasticity detected' if abs(correlation) > 0.3 else 'Homoscedasticity'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _detect_outliers_in_residuals(self, 
                                    residuals: np.ndarray, 
                                    standardized_residuals: np.ndarray) -> Dict[str, Any]:
        """Detect outliers in residuals."""
        # Z-score method (standardized residuals > 3)
        z_outliers = np.where(np.abs(standardized_residuals) > 3)[0]
        
        # IQR method
        q1, q3 = np.percentile(residuals, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        iqr_outliers = np.where((residuals < lower_bound) | (residuals > upper_bound))[0]
        
        return {
            'z_score_outliers': {
                'indices': z_outliers.tolist(),
                'count': len(z_outliers),
                'percentage': len(z_outliers) / len(residuals) * 100
            },
            'iqr_outliers': {
                'indices': iqr_outliers.tolist(),
                'count': len(iqr_outliers),
                'percentage': len(iqr_outliers) / len(residuals) * 100
            }
        }
    
    def _create_residual_plots(self, 
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             residuals: np.ndarray,
                             output_dir: str) -> None:
        """Create comprehensive residual diagnostic plots."""
        plt.style.use('seaborn-v0_8')
        
        # 1. Residuals vs Fitted
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.scatter(y_pred, residuals, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Fitted Values')
        
        # 2. Q-Q plot
        plt.subplot(2, 2, 2)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals')
        
        # 3. Histogram of residuals
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30, density=True, alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Density')
        plt.title('Distribution of Residuals')
        
        # 4. Residuals over time
        plt.subplot(2, 2, 4)
        plt.plot(residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Residuals')
        plt.title('Residuals Over Time')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'residual_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Residual plots saved to {output_dir}")
    
    def cross_validation(self, 
                        model_builder_func,
                        X: np.ndarray,
                        y: np.ndarray,
                        cv_folds: int = 5,
                        time_series_split: bool = True) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.
        
        Args:
            model_builder_func: Function to build and compile model
            X: Input features
            y: Target values
            cv_folds: Number of CV folds
            time_series_split: Whether to use time series split
        
        Returns:
            Cross-validation results
        """
        self.logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        if time_series_split:
            cv = TimeSeriesSplit(n_splits=cv_folds)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_scores = {
            'rmse': [], 'mae': [], 'mape': [], 'r2': [], 'direction_accuracy': []
        }
        
        fold_predictions = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            self.logger.info(f"Processing fold {fold + 1}/{cv_folds}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            try:
                # Build and train model
                model = model_builder_func()
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                
                history = model.fit(
                    X_train_fold, y_train_fold,
                    validation_data=(X_val_fold, y_val_fold),
                    epochs=50,
                    batch_size=32,
                    verbose=0,
                    callbacks=[
                        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                    ]
                )
                
                # Make predictions
                y_pred_fold = model.predict(X_val_fold, verbose=0).flatten()
                
                # Calculate metrics
                cv_scores['rmse'].append(calculate_rmse(y_val_fold, y_pred_fold))
                cv_scores['mae'].append(calculate_mae(y_val_fold, y_pred_fold))
                cv_scores['mape'].append(calculate_mape(y_val_fold, y_pred_fold))
                cv_scores['r2'].append(r2_score(y_val_fold, y_pred_fold))
                cv_scores['direction_accuracy'].append(direction_accuracy(y_val_fold, y_pred_fold))
                
                fold_predictions.append({
                    'fold': fold + 1,
                    'true_values': y_val_fold,
                    'predictions': y_pred_fold,
                    'indices': val_idx
                })
                
            except Exception as e:
                self.logger.warning(f"Fold {fold + 1} failed: {str(e)}")
        
        # Calculate summary statistics
        cv_results = {}
        for metric, scores in cv_scores.items():
            if scores:  # Only if we have valid scores
                cv_results[metric] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'scores': scores
                }
        
        cv_results['fold_predictions'] = fold_predictions
        cv_results['n_folds'] = len(fold_predictions)
        
        self.logger.info("Cross-validation completed")
        
        return cv_results
    
    def generate_evaluation_report(self, 
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 model_name: str = "LSTM_Model",
                                 output_dir: str = "reports/") -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            output_dir: Directory to save the report
        
        Returns:
            Complete evaluation report
        """
        self.logger.info(f"Generating evaluation report for {model_name}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate all metrics
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # Residual analysis
        residual_results = self.residual_analysis(y_true, y_pred, 
                                                save_plots=True, 
                                                output_dir=output_dir)
        
        # Performance classification
        performance_class = self._classify_model_performance(metrics)
        
        # Create comprehensive report
        report = {
            'model_name': model_name,
            'evaluation_date': pd.Timestamp.now().isoformat(),
            'data_summary': {
                'n_samples': len(y_true),
                'target_mean': float(np.mean(y_true)),
                'target_std': float(np.std(y_true)),
                'target_range': [float(np.min(y_true)), float(np.max(y_true))],
                'prediction_mean': float(np.mean(y_pred)),
                'prediction_std': float(np.std(y_pred))
            },
            'performance_metrics': metrics,
            'residual_analysis': residual_results,
            'performance_classification': performance_class,
            'recommendations': self._generate_recommendations(metrics, residual_results)
        }
        
        # Save report
        report_path = os.path.join(output_dir, f"{model_name}_evaluation_report.json")
        save_json(report, report_path)
        
        # Create summary text report
        self._create_text_report(report, output_dir, model_name)
        
        self.logger.info(f"Evaluation report saved to {output_dir}")
        
        return report
    
    def _classify_model_performance(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Classify model performance based on metrics."""
        classification = {}
        
        # MAPE classification
        mape = metrics.get('mape', float('inf'))
        if mape < 1:
            classification['mape_class'] = 'Excellent'
        elif mape < 3:
            classification['mape_class'] = 'Very Good'
        elif mape < 5:
            classification['mape_class'] = 'Good'
        elif mape < 10:
            classification['mape_class'] = 'Fair'
        else:
            classification['mape_class'] = 'Poor'
        
        # R² classification
        r2 = metrics.get('r2', -float('inf'))
        if r2 > 0.9:
            classification['r2_class'] = 'Excellent'
        elif r2 > 0.8:
            classification['r2_class'] = 'Very Good'
        elif r2 > 0.7:
            classification['r2_class'] = 'Good'
        elif r2 > 0.5:
            classification['r2_class'] = 'Fair'
        else:
            classification['r2_class'] = 'Poor'
        
        # Direction accuracy classification
        dir_acc = metrics.get('direction_accuracy', 0)
        if dir_acc > 70:
            classification['direction_class'] = 'Excellent'
        elif dir_acc > 60:
            classification['direction_class'] = 'Good'
        elif dir_acc > 50:
            classification['direction_class'] = 'Fair'
        else:
            classification['direction_class'] = 'Poor'
        
        # Overall classification (worst of the three)
        classes = [classification['mape_class'], classification['r2_class'], classification['direction_class']]
        class_order = ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor']
        
        worst_class_idx = max([class_order.index(cls) for cls in classes])
        classification['overall'] = class_order[worst_class_idx]
        
        return classification
    
    def _generate_recommendations(self, 
                                metrics: Dict[str, Any], 
                                residual_analysis: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on evaluation results."""
        recommendations = []
        
        # Performance-based recommendations
        mape = metrics.get('mape', float('inf'))
        r2 = metrics.get('r2', -float('inf'))
        dir_acc = metrics.get('direction_accuracy', 0)
        
        if mape > 5:
            recommendations.append("Consider feature engineering or increasing model complexity to reduce MAPE")
        
        if r2 < 0.8:
            recommendations.append("Model explains less than 80% of variance - consider adding more features or regularization")
        
        if dir_acc < 60:
            recommendations.append("Poor directional accuracy - consider using directional loss function")
        
        # Residual-based recommendations
        normality_tests = residual_analysis.get('normality_tests', {})
        if any(test.get('p_value', 1) < 0.05 for test in normality_tests.values() if isinstance(test, dict)):
            recommendations.append("Residuals are not normally distributed - consider data transformation")
        
        heteroscedasticity = residual_analysis.get('heteroscedasticity', {})
        if 'Heteroscedasticity detected' in str(heteroscedasticity.get('interpretation', '')):
            recommendations.append("Heteroscedasticity detected - consider robust regression or data transformation")
        
        autocorr = residual_analysis.get('autocorrelation', {})
        if 'autocorrelation' in str(autocorr.get('interpretation', '')).lower():
            recommendations.append("Autocorrelation in residuals - consider AR/MA components or longer sequences")
        
        outliers = residual_analysis.get('outliers', {})
        outlier_pct = outliers.get('z_score_outliers', {}).get('percentage', 0)
        if outlier_pct > 5:
            recommendations.append("High percentage of outliers - consider outlier removal or robust loss functions")
        
        if not recommendations:
            recommendations.append("Model performance is satisfactory - consider ensemble methods for further improvement")
        
        return recommendations
    
    def _create_text_report(self, 
                          report: Dict[str, Any], 
                          output_dir: str, 
                          model_name: str) -> None:
        """Create human-readable text report."""
        text_report_path = os.path.join(output_dir, f"{model_name}_evaluation_summary.txt")
        
        with open(text_report_path, 'w') as f:
            f.write(f"Model Evaluation Report: {model_name}\n")
            f.write("=" * 50 + "\n\n")
            
            # Performance summary
            metrics = report['performance_metrics']
            f.write("Performance Metrics:\n")
            f.write(f"  RMSE: {format_number(metrics['rmse'])}\n")
            f.write(f"  MAE: {format_number(metrics['mae'])}\n")
            f.write(f"  MAPE: {format_number(metrics['mape'])}%\n")
            f.write(f"  R²: {format_number(metrics['r2'])}\n")
            f.write(f"  Direction Accuracy: {format_number(metrics['direction_accuracy'])}%\n\n")
            
            # Performance classification
            classification = report['performance_classification']
            f.write("Performance Classification:\n")
            f.write(f"  Overall: {classification['overall']}\n")
            f.write(f"  MAPE: {classification['mape_class']}\n")
            f.write(f"  R²: {classification['r2_class']}\n")
            f.write(f"  Direction: {classification['direction_class']}\n\n")
            
            # Recommendations
            f.write("Recommendations:\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"  {i}. {rec}\n")
        
        self.logger.info(f"Text report saved to {text_report_path}")


class WalkForwardValidator:
    """
    Walk-forward validation for time series models.
    """
    
    def __init__(self, 
                 initial_window: int = 100,
                 step_size: int = 1,
                 forecast_horizon: int = 1):
        """
        Initialize walk-forward validator.
        
        Args:
            initial_window: Initial training window size
            step_size: Number of steps to move window
            forecast_horizon: Number of steps to forecast
        """
        self.initial_window = initial_window
        self.step_size = step_size
        self.forecast_horizon = forecast_horizon
        self.logger = setup_logging()
    
    def validate(self, 
                model_builder_func,
                X: np.ndarray,
                y: np.ndarray) -> Dict[str, Any]:
        """
        Perform walk-forward validation.
        
        Args:
            model_builder_func: Function to build model
            X: Input features
            y: Target values
        
        Returns:
            Walk-forward validation results
        """
        self.logger.info("Starting walk-forward validation")
        
        n_samples = len(X)
        predictions = []
        actuals = []
        fold_metrics = []
        
        start_idx = self.initial_window
        
        while start_idx + self.forecast_horizon < n_samples:
            # Define training window
            train_start = max(0, start_idx - self.initial_window)
            train_end = start_idx
            
            # Define prediction window
            pred_start = start_idx
            pred_end = min(start_idx + self.forecast_horizon, n_samples)
            
            # Extract data
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_pred = X[pred_start:pred_end]
            y_actual = y[pred_start:pred_end]
            
            try:
                # Build and train model
                model = model_builder_func()
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                
                model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
                
                # Make prediction
                y_pred = model.predict(X_pred, verbose=0).flatten()
                
                # Store results
                predictions.extend(y_pred)
                actuals.extend(y_actual)
                
                # Calculate fold metrics
                fold_rmse = calculate_rmse(y_actual, y_pred)
                fold_mae = calculate_mae(y_actual, y_pred)
                fold_mape = calculate_mape(y_actual, y_pred)
                
                fold_metrics.append({
                    'fold': len(fold_metrics) + 1,
                    'train_start': train_start,
                    'train_end': train_end,
                    'pred_start': pred_start,
                    'pred_end': pred_end,
                    'rmse': fold_rmse,
                    'mae': fold_mae,
                    'mape': fold_mape
                })
                
                self.logger.info(f"Fold {len(fold_metrics)}: RMSE={fold_rmse:.4f}, MAE={fold_mae:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Fold failed: {str(e)}")
            
            start_idx += self.step_size
        
        # Calculate overall metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        overall_metrics = {
            'rmse': calculate_rmse(actuals, predictions),
            'mae': calculate_mae(actuals, predictions),
            'mape': calculate_mape(actuals, predictions),
            'r2': r2_score(actuals, predictions),
            'direction_accuracy': direction_accuracy(actuals, predictions)
        }
        
        results = {
            'overall_metrics': overall_metrics,
            'fold_metrics': fold_metrics,
            'predictions': predictions.tolist(),
            'actuals': actuals.tolist(),
            'n_folds': len(fold_metrics)
        }
        
        self.logger.info("Walk-forward validation completed")
        
        return results
