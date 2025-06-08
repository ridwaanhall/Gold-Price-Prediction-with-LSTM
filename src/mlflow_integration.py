"""
MLflow Integration for Gold Price Prediction

This module provides MLflow integration for experiment tracking,
model versioning, and deployment management.
"""

import os
import logging
import json
import pickle
from datetime import datetime
from typing import Dict, Any, Optional, List
import tempfile
import shutil

import mlflow
import mlflow.tensorflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
import tensorflow as tf

from .utils import setup_logging
from .config.config import Config


logger = logging.getLogger(__name__)


class MLflowManager:
    """
    Manages MLflow operations for the gold price prediction project
    """
    
    def __init__(self, config: Config, experiment_name: str = "gold_price_prediction"):
        """
        Initialize MLflow manager
        
        Args:
            config: Configuration object
            experiment_name: Name of the MLflow experiment
        """
        self.config = config
        self.experiment_name = experiment_name
        self.client = MlflowClient()
        
        # Set MLflow tracking URI
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except Exception:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
        
        logger.info(f"MLflow experiment: {experiment_name} (ID: {self.experiment_id})")
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Start an MLflow run
        
        Args:
            run_name: Name for the run
            tags: Tags to add to the run
            
        Returns:
            MLflow run object
        """
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        default_tags = {
            "project": "gold_price_prediction",
            "version": "1.0.0",
            "environment": os.getenv('ENVIRONMENT', 'development')
        }
        
        if tags:
            default_tags.update(tags)
        
        run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=default_tags
        )
        
        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """
        Log hyperparameters to MLflow
        
        Args:
            hyperparams: Dictionary of hyperparameters
        """
        try:
            # Convert complex types to strings
            clean_params = {}
            for key, value in hyperparams.items():
                if isinstance(value, (dict, list)):
                    clean_params[key] = json.dumps(value)
                elif isinstance(value, np.ndarray):
                    clean_params[key] = value.tolist()
                else:
                    clean_params[key] = value
            
            mlflow.log_params(clean_params)
            logger.info(f"Logged {len(clean_params)} hyperparameters")
            
        except Exception as e:
            logger.error(f"Error logging hyperparameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow
        
        Args:
            metrics: Dictionary of metrics
            step: Step number for the metrics
        """
        try:
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value, step=step)
            
            logger.info(f"Logged {len(metrics)} metrics")
            
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    def log_model(self, 
                  model: tf.keras.Model,
                  model_name: str = "lstm_gold_predictor",
                  signature: Optional[Any] = None,
                  input_example: Optional[np.ndarray] = None,
                  artifacts: Optional[Dict[str, str]] = None):
        """
        Log model to MLflow
        
        Args:
            model: Trained Keras model
            model_name: Name for the model
            signature: Model signature
            input_example: Example input for the model
            artifacts: Additional artifacts to log
        """
        try:
            # Log the model
            mlflow.tensorflow.log_model(
                model=model,
                artifact_path=model_name,
                signature=signature,
                input_example=input_example
            )
            
            # Log additional artifacts
            if artifacts:
                for artifact_name, artifact_path in artifacts.items():
                    mlflow.log_artifact(artifact_path, artifact_name)
            
            logger.info(f"Logged model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error logging model: {e}")
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """
        Log dataset information
        
        Args:
            dataset_info: Information about the dataset
        """
        try:
            # Log as parameters
            for key, value in dataset_info.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(f"data_{key}", value)
            
            # Save detailed info as artifact
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(dataset_info, f, indent=2, default=str)
                temp_path = f.name
            
            mlflow.log_artifact(temp_path, "dataset_info.json")
            os.unlink(temp_path)
            
            logger.info("Logged dataset information")
            
        except Exception as e:
            logger.error(f"Error logging dataset info: {e}")
    
    def log_training_plots(self, plots_dir: str):
        """
        Log training plots as artifacts
        
        Args:
            plots_dir: Directory containing plots
        """
        try:
            if os.path.exists(plots_dir):
                mlflow.log_artifacts(plots_dir, "plots")
                logger.info(f"Logged plots from {plots_dir}")
            
        except Exception as e:
            logger.error(f"Error logging plots: {e}")
    
    def log_predictions(self, predictions: np.ndarray, actuals: np.ndarray):
        """
        Log predictions vs actuals
        
        Args:
            predictions: Model predictions
            actuals: Actual values
        """
        try:
            # Create predictions DataFrame
            results_df = pd.DataFrame({
                'actual': actuals.flatten(),
                'predicted': predictions.flatten(),
                'error': actuals.flatten() - predictions.flatten()
            })
            
            # Save as CSV
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                results_df.to_csv(f.name, index=False)
                temp_path = f.name
            
            mlflow.log_artifact(temp_path, "predictions.csv")
            os.unlink(temp_path)
            
            logger.info("Logged predictions")
            
        except Exception as e:
            logger.error(f"Error logging predictions: {e}")
    
    def register_model(self, 
                      model_name: str,
                      run_id: Optional[str] = None,
                      model_version: Optional[str] = None,
                      description: Optional[str] = None):
        """
        Register model in MLflow Model Registry
        
        Args:
            model_name: Name of the model
            run_id: Run ID containing the model
            model_version: Version of the model
            description: Description of the model
            
        Returns:
            Registered model version
        """
        try:
            if run_id is None:
                run_id = mlflow.active_run().info.run_id
            
            model_uri = f"runs:/{run_id}/{model_name}"
            
            # Register the model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                description=description
            )
            
            logger.info(f"Registered model: {model_name} version {model_version.version}")
            return model_version
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return None
    
    def transition_model_stage(self, 
                              model_name: str, 
                              version: str, 
                              stage: str):
        """
        Transition model to a different stage
        
        Args:
            model_name: Name of the model
            version: Version of the model
            stage: Target stage (Staging, Production, Archived)
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
            
        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")
    
    def get_best_model(self, metric_name: str = "val_mape", ascending: bool = True):
        """
        Get the best model based on a metric
        
        Args:
            metric_name: Metric to optimize for
            ascending: Whether lower values are better
            
        Returns:
            Best run information
        """
        try:
            experiment = mlflow.get_experiment(self.experiment_id)
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"]
            )
            
            if len(runs) > 0:
                best_run = runs.iloc[0]
                logger.info(f"Best model: {best_run['run_id']} with {metric_name}: {best_run[f'metrics.{metric_name}']}")
                return best_run
            else:
                logger.warning("No runs found in experiment")
                return None
                
        except Exception as e:
            logger.error(f"Error getting best model: {e}")
            return None
    
    def load_model(self, model_name: str, version: Optional[str] = None, stage: Optional[str] = None):
        """
        Load model from MLflow Model Registry
        
        Args:
            model_name: Name of the model
            version: Version of the model
            stage: Stage of the model (Production, Staging)
            
        Returns:
            Loaded model
        """
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
            else:
                model_uri = f"models:/{model_name}/latest"
            
            model = mlflow.tensorflow.load_model(model_uri)
            logger.info(f"Loaded model: {model_uri}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def compare_models(self, run_ids: List[str], metrics: List[str]):
        """
        Compare multiple models
        
        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to compare
            
        Returns:
            Comparison DataFrame
        """
        try:
            comparison_data = []
            
            for run_id in run_ids:
                run = self.client.get_run(run_id)
                row = {'run_id': run_id}
                
                for metric in metrics:
                    metric_value = run.data.metrics.get(metric, None)
                    row[metric] = metric_value
                
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            logger.info(f"Compared {len(run_ids)} models")
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return None
    
    def delete_experiment(self):
        """Delete the current experiment"""
        try:
            mlflow.delete_experiment(self.experiment_id)
            logger.info(f"Deleted experiment: {self.experiment_name}")
            
        except Exception as e:
            logger.error(f"Error deleting experiment: {e}")
    
    def cleanup_old_runs(self, keep_last_n: int = 10):
        """
        Clean up old runs, keeping only the most recent ones
        
        Args:
            keep_last_n: Number of recent runs to keep
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=["start_time DESC"]
            )
            
            if len(runs) > keep_last_n:
                runs_to_delete = runs.iloc[keep_last_n:]
                
                for _, run in runs_to_delete.iterrows():
                    mlflow.delete_run(run['run_id'])
                
                logger.info(f"Deleted {len(runs_to_delete)} old runs")
            
        except Exception as e:
            logger.error(f"Error cleaning up old runs: {e}")


class MLflowCallback(tf.keras.callbacks.Callback):
    """
    Keras callback for logging metrics to MLflow during training
    """
    
    def __init__(self, log_every_n_epochs: int = 1):
        """
        Initialize MLflow callback
        
        Args:
            log_every_n_epochs: Log metrics every n epochs
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
    
    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at the end of each epoch"""
        if logs and (epoch + 1) % self.log_every_n_epochs == 0:
            for metric_name, metric_value in logs.items():
                mlflow.log_metric(metric_name, metric_value, step=epoch)
    
    def on_train_end(self, logs=None):
        """Log final metrics at the end of training"""
        if logs:
            for metric_name, metric_value in logs.items():
                mlflow.log_metric(f"final_{metric_name}", metric_value)


def create_mlflow_manager(config: Config, experiment_name: str = "gold_price_prediction") -> MLflowManager:
    """
    Factory function to create MLflow manager
    
    Args:
        config: Configuration object
        experiment_name: Name of the experiment
        
    Returns:
        MLflowManager instance
    """
    return MLflowManager(config, experiment_name)


def log_experiment_summary(manager: MLflowManager, 
                          model_performance: Dict[str, float],
                          training_time: float,
                          data_info: Dict[str, Any]):
    """
    Log a comprehensive experiment summary
    
    Args:
        manager: MLflow manager instance
        model_performance: Model performance metrics
        training_time: Training time in seconds
        data_info: Information about the training data
    """
    try:
        # Log performance metrics
        manager.log_metrics(model_performance)
        
        # Log training time
        manager.log_metrics({"training_time_seconds": training_time})
        
        # Log data information
        manager.log_dataset_info(data_info)
        
        # Log system information
        system_info = {
            "python_version": os.sys.version,
            "tensorflow_version": tf.__version__,
            "timestamp": datetime.now().isoformat()
        }
        
        manager.log_hyperparameters(system_info)
        
        logger.info("Logged comprehensive experiment summary")
        
    except Exception as e:
        logger.error(f"Error logging experiment summary: {e}")


if __name__ == "__main__":
    # Example usage
    from config.config import load_config
    
    config = load_config()
    manager = create_mlflow_manager(config)
    
    # Start a run
    with manager.start_run("test_run"):
        # Log some test metrics
        manager.log_metrics({
            "mape": 2.5,
            "rmse": 15.2,
            "mae": 12.1
        })
        
        print("MLflow integration test completed successfully!")