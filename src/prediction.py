"""
Gold Price Prediction Module

This module provides functionality for making predictions using trained LSTM models,
including single-step and multi-step forecasting with confidence intervals.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import joblib
from datetime import datetime, timedelta
import warnings

from .utils import setup_logging, save_json, load_json
from .data_preprocessing import GoldDataPreprocessor
from .lstm_model import LSTMGoldPredictor

warnings.filterwarnings('ignore')


class GoldPricePredictor:
    """
    Gold Price Predictor for making forecasts using trained LSTM models.
    
    This class handles:
    - Loading trained models and preprocessors
    - Single-step and multi-step predictions
    - Confidence interval estimation
    - Prediction uncertainty quantification
    - Scenario analysis
    """
    
    def __init__(self, model_path: str, preprocessor_path: str, config: Dict):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to saved model
            preprocessor_path: Path to saved preprocessor
            config: Configuration dictionary
        """
        self.config = config
        self.logger = setup_logging('GoldPricePredictor')
        
        # Load model and preprocessor
        self.model = self._load_model(model_path)
        self.preprocessor = self._load_preprocessor(preprocessor_path)
        
        # Prediction settings
        self.sequence_length = config.get('sequence_length', 30)
        self.confidence_level = config.get('confidence_level', 0.95)
        
        self.logger.info(f"Predictor initialized with model: {model_path}")
    
    def _load_model(self, model_path: str) -> tf.keras.Model:
        """Load trained model."""
        try:
            model = tf.keras.models.load_model(model_path)
            self.logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def _load_preprocessor(self, preprocessor_path: str) -> GoldDataPreprocessor:
        """Load trained preprocessor."""
        try:
            preprocessor = joblib.load(preprocessor_path)
            self.logger.info(f"Preprocessor loaded from {preprocessor_path}")
            return preprocessor
        except Exception as e:
            self.logger.error(f"Error loading preprocessor: {e}")
            raise
    
    def predict_single_step(self, 
                           input_data: Union[pd.DataFrame, np.ndarray],
                           return_confidence: bool = True) -> Dict:
        """
        Make single-step prediction.
        
        Args:
            input_data: Input data for prediction
            return_confidence: Whether to return confidence intervals
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess input data
            if isinstance(input_data, pd.DataFrame):
                processed_data = self.preprocessor.prepare_prediction_data(input_data)
            else:
                processed_data = input_data
            
            # Ensure correct shape
            if len(processed_data.shape) == 2:
                processed_data = processed_data.reshape(1, *processed_data.shape)
            
            # Make prediction
            prediction = self.model.predict(processed_data, verbose=0)
            
            # Inverse transform if needed
            if hasattr(self.preprocessor, 'target_scaler'):
                prediction = self.preprocessor.target_scaler.inverse_transform(
                    prediction.reshape(-1, 1)
                ).flatten()
            
            result = {
                'prediction': float(prediction[0]),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add confidence intervals if requested
            if return_confidence:
                confidence_interval = self._calculate_confidence_interval(
                    processed_data, prediction[0]
                )
                result.update(confidence_interval)
            
            self.logger.info(f"Single-step prediction: {result['prediction']:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in single-step prediction: {e}")
            raise
    
    def predict_multi_step(self, 
                          input_data: Union[pd.DataFrame, np.ndarray],
                          steps: int = 7,
                          method: str = 'recursive') -> Dict:
        """
        Make multi-step predictions.
        
        Args:
            input_data: Input data for prediction
            steps: Number of steps to predict
            method: Prediction method ('recursive' or 'direct')
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if method == 'recursive':
                return self._predict_recursive(input_data, steps)
            elif method == 'direct':
                return self._predict_direct(input_data, steps)
            else:
                raise ValueError(f"Unknown prediction method: {method}")
                
        except Exception as e:
            self.logger.error(f"Error in multi-step prediction: {e}")
            raise
    
    def _predict_recursive(self, input_data: Union[pd.DataFrame, np.ndarray], 
                          steps: int) -> Dict:
        """Recursive multi-step prediction."""
        try:
            # Preprocess input data
            if isinstance(input_data, pd.DataFrame):
                processed_data = self.preprocessor.prepare_prediction_data(input_data)
            else:
                processed_data = input_data.copy()
            
            predictions = []
            current_input = processed_data[-self.sequence_length:].copy()
            
            for step in range(steps):
                # Reshape for model input
                model_input = current_input.reshape(1, *current_input.shape)
                
                # Make prediction
                pred = self.model.predict(model_input, verbose=0)
                predictions.append(float(pred[0]))
                
                # Update input for next prediction
                # Remove first element and add prediction
                current_input = np.roll(current_input, -1, axis=0)
                current_input[-1] = pred[0]  # Assuming single feature output
            
            # Inverse transform predictions if needed
            if hasattr(self.preprocessor, 'target_scaler'):
                predictions = self.preprocessor.target_scaler.inverse_transform(
                    np.array(predictions).reshape(-1, 1)
                ).flatten()
            
            # Generate timestamps
            timestamps = self._generate_future_timestamps(steps)
            
            result = {
                'predictions': predictions.tolist(),
                'timestamps': timestamps,
                'method': 'recursive',
                'steps': steps,
                'generated_at': datetime.now().isoformat()
            }
            
            self.logger.info(f"Recursive prediction completed for {steps} steps")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in recursive prediction: {e}")
            raise
    
    def _predict_direct(self, input_data: Union[pd.DataFrame, np.ndarray], 
                       steps: int) -> Dict:
        """Direct multi-step prediction (requires specially trained model)."""
        try:
            # This would require a model trained for multi-output
            # For now, fall back to recursive method
            self.logger.warning("Direct multi-step prediction not implemented, using recursive")
            return self._predict_recursive(input_data, steps)
            
        except Exception as e:
            self.logger.error(f"Error in direct prediction: {e}")
            raise
    
    def _calculate_confidence_interval(self, input_data: np.ndarray, 
                                     prediction: float) -> Dict:
        """Calculate prediction confidence interval using Monte Carlo dropout."""
        try:
            n_samples = 100
            predictions = []
            
            # Enable dropout during inference for uncertainty estimation
            for _ in range(n_samples):
                pred = self.model(input_data, training=True)
                if hasattr(self.preprocessor, 'target_scaler'):
                    pred = self.preprocessor.target_scaler.inverse_transform(
                        pred.numpy().reshape(-1, 1)
                    ).flatten()
                predictions.append(float(pred[0]))
            
            predictions = np.array(predictions)
            
            # Calculate confidence interval
            alpha = 1 - self.confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(predictions, lower_percentile)
            upper_bound = np.percentile(predictions, upper_percentile)
            std_dev = np.std(predictions)
            
            return {
                'confidence_interval': {
                    'lower': float(lower_bound),
                    'upper': float(upper_bound),
                    'level': self.confidence_level
                },
                'uncertainty': {
                    'std_dev': float(std_dev),
                    'variance': float(np.var(predictions))
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Could not calculate confidence interval: {e}")
            return {}
    
    def _generate_future_timestamps(self, steps: int) -> List[str]:
        """Generate future timestamps for predictions."""
        try:
            # Assume daily predictions
            base_time = datetime.now()
            timestamps = []
            
            for i in range(1, steps + 1):
                future_time = base_time + timedelta(days=i)
                timestamps.append(future_time.isoformat())
            
            return timestamps
            
        except Exception as e:
            self.logger.error(f"Error generating timestamps: {e}")
            return [f"step_{i}" for i in range(1, steps + 1)]
    
    def predict_scenarios(self, 
                         base_data: Union[pd.DataFrame, np.ndarray],
                         scenarios: Dict[str, Dict]) -> Dict:
        """
        Predict under different scenarios.
        
        Args:
            base_data: Base input data
            scenarios: Dictionary of scenario modifications
            
        Returns:
            Dictionary with predictions for each scenario
        """
        try:
            results = {}
            
            for scenario_name, modifications in scenarios.items():
                self.logger.info(f"Running scenario: {scenario_name}")
                
                # Apply scenario modifications
                modified_data = self._apply_scenario_modifications(
                    base_data.copy(), modifications
                )
                
                # Make prediction
                prediction = self.predict_single_step(modified_data)
                results[scenario_name] = prediction
            
            return {
                'scenarios': results,
                'base_scenario': self.predict_single_step(base_data),
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in scenario prediction: {e}")
            raise
    
    def _apply_scenario_modifications(self, 
                                    data: Union[pd.DataFrame, np.ndarray],
                                    modifications: Dict) -> Union[pd.DataFrame, np.ndarray]:
        """Apply scenario modifications to data."""
        try:
            # This is a simplified implementation
            # In practice, you'd have more sophisticated scenario modeling
            
            if isinstance(data, pd.DataFrame):
                for column, change in modifications.items():
                    if column in data.columns:
                        if isinstance(change, (int, float)):
                            data[column] *= (1 + change)  # Percentage change
                        elif isinstance(change, dict):
                            if 'multiply' in change:
                                data[column] *= change['multiply']
                            elif 'add' in change:
                                data[column] += change['add']
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error applying scenario modifications: {e}")
            raise
    
    def batch_predict(self, 
                     data_list: List[Union[pd.DataFrame, np.ndarray]]) -> List[Dict]:
        """Make batch predictions."""
        try:
            results = []
            
            for i, data in enumerate(data_list):
                self.logger.info(f"Processing batch item {i+1}/{len(data_list)}")
                prediction = self.predict_single_step(data)
                results.append(prediction)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch prediction: {e}")
            raise
    
    def get_feature_importance(self, 
                              input_data: Union[pd.DataFrame, np.ndarray],
                              method: str = 'gradient') -> Dict:
        """
        Calculate feature importance for predictions.
        
        Args:
            input_data: Input data
            method: Method for calculating importance ('gradient' or 'permutation')
            
        Returns:
            Dictionary with feature importance scores
        """
        try:
            if method == 'gradient':
                return self._gradient_feature_importance(input_data)
            elif method == 'permutation':
                return self._permutation_feature_importance(input_data)
            else:
                raise ValueError(f"Unknown importance method: {method}")
                
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {e}")
            raise
    
    def _gradient_feature_importance(self, 
                                   input_data: Union[pd.DataFrame, np.ndarray]) -> Dict:
        """Calculate feature importance using gradients."""
        try:
            # Preprocess input data
            if isinstance(input_data, pd.DataFrame):
                processed_data = self.preprocessor.prepare_prediction_data(input_data)
            else:
                processed_data = input_data
            
            # Ensure correct shape
            if len(processed_data.shape) == 2:
                processed_data = processed_data.reshape(1, *processed_data.shape)
            
            # Calculate gradients
            with tf.GradientTape() as tape:
                input_tensor = tf.Variable(processed_data, dtype=tf.float32)
                tape.watch(input_tensor)
                prediction = self.model(input_tensor)
            
            gradients = tape.gradient(prediction, input_tensor)
            
            # Calculate importance scores
            importance_scores = np.abs(gradients.numpy()).mean(axis=0)
            
            # Create feature names
            feature_names = [f"feature_{i}" for i in range(importance_scores.shape[-1])]
            
            # Sort by importance
            sorted_indices = np.argsort(importance_scores.flatten())[::-1]
            
            return {
                'method': 'gradient',
                'importance_scores': {
                    feature_names[i]: float(importance_scores.flatten()[i])
                    for i in sorted_indices
                },
                'top_features': [feature_names[i] for i in sorted_indices[:5]]
            }
            
        except Exception as e:
            self.logger.error(f"Error in gradient importance calculation: {e}")
            return {}
    
    def _permutation_feature_importance(self, 
                                      input_data: Union[pd.DataFrame, np.ndarray]) -> Dict:
        """Calculate feature importance using permutation."""
        try:
            # This is a simplified implementation
            # In practice, you'd need more sophisticated permutation testing
            
            baseline_pred = self.predict_single_step(input_data, return_confidence=False)
            baseline_score = baseline_pred['prediction']
            
            importance_scores = {}
            
            # This would require implementing proper permutation testing
            # For now, return placeholder
            return {
                'method': 'permutation',
                'importance_scores': {},
                'note': 'Permutation importance not fully implemented'
            }
            
        except Exception as e:
            self.logger.error(f"Error in permutation importance calculation: {e}")
            return {}
    
    def save_predictions(self, predictions: Dict, filepath: str):
        """Save predictions to file."""
        try:
            save_json(predictions, filepath)
            self.logger.info(f"Predictions saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving predictions: {e}")
            raise


class PredictionPipeline:
    """
    Complete prediction pipeline for gold price forecasting.
    
    This class orchestrates the entire prediction process from data loading
    to result generation and saving.
    """
    
    def __init__(self, config: Dict):
        """Initialize prediction pipeline."""
        self.config = config
        self.logger = setup_logging('PredictionPipeline')
        
        # Initialize components
        self.predictor = None
        self.results = {}
    
    def setup(self, model_path: str, preprocessor_path: str):
        """Setup the prediction pipeline."""
        try:
            self.predictor = GoldPricePredictor(
                model_path=model_path,
                preprocessor_path=preprocessor_path,
                config=self.config
            )
            self.logger.info("Prediction pipeline setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up prediction pipeline: {e}")
            raise
    
    def run_prediction_suite(self, 
                           input_data: Union[pd.DataFrame, str],
                           output_dir: str) -> Dict:
        """
        Run complete prediction suite.
        
        Args:
            input_data: Input data or path to data file
            output_dir: Directory to save results
            
        Returns:
            Dictionary with all prediction results
        """
        try:
            # Load data if path provided
            if isinstance(input_data, str):
                if input_data.endswith('.json'):
                    data = pd.read_json(input_data)
                elif input_data.endswith('.csv'):
                    data = pd.read_csv(input_data)
                else:
                    raise ValueError(f"Unsupported file format: {input_data}")
            else:
                data = input_data
            
            results = {}
            
            # Single-step prediction
            self.logger.info("Running single-step prediction")
            single_step = self.predictor.predict_single_step(data)
            results['single_step'] = single_step
            
            # Multi-step prediction
            self.logger.info("Running multi-step prediction")
            multi_step = self.predictor.predict_multi_step(
                data, steps=self.config.get('forecast_horizon', 7)
            )
            results['multi_step'] = multi_step
            
            # Feature importance
            self.logger.info("Calculating feature importance")
            importance = self.predictor.get_feature_importance(data)
            results['feature_importance'] = importance
            
            # Save results
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = output_path / f"predictions_{timestamp}.json"
            
            self.predictor.save_predictions(results, str(results_file))
            
            self.results = results
            self.logger.info("Prediction suite completed successfully")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in prediction suite: {e}")
            raise
    
    def generate_report(self, output_path: str = None) -> str:
        """Generate prediction report."""
        try:
            if not self.results:
                raise ValueError("No prediction results available. Run prediction suite first.")
            
            report_lines = [
                "# Gold Price Prediction Report",
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## Single-Step Prediction",
                f"Predicted Price: ${self.results['single_step']['prediction']:.2f}",
                ""
            ]
            
            # Add confidence interval if available
            if 'confidence_interval' in self.results['single_step']:
                ci = self.results['single_step']['confidence_interval']
                report_lines.extend([
                    f"Confidence Interval ({ci['level']*100}%): "
                    f"${ci['lower']:.2f} - ${ci['upper']:.2f}",
                    ""
                ])
            
            # Add multi-step predictions
            if 'multi_step' in self.results:
                multi_step = self.results['multi_step']
                report_lines.extend([
                    "## Multi-Step Forecast",
                    f"Forecast Horizon: {multi_step['steps']} days",
                    "Predictions:"
                ])
                
                for i, (pred, timestamp) in enumerate(zip(
                    multi_step['predictions'], multi_step['timestamps']
                )):
                    report_lines.append(f"Day {i+1}: ${pred:.2f} ({timestamp[:10]})")
                
                report_lines.append("")
            
            # Add feature importance
            if 'feature_importance' in self.results and self.results['feature_importance']:
                importance = self.results['feature_importance']
                if 'top_features' in importance:
                    report_lines.extend([
                        "## Top Important Features",
                        *[f"- {feature}" for feature in importance['top_features'][:5]],
                        ""
                    ])
            
            report_content = "\n".join(report_lines)
            
            # Save report if path provided
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(report_content)
                self.logger.info(f"Report saved to {output_path}")
            
            return report_content
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            raise
