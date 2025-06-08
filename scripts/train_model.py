"""
Training Script for Gold Price Prediction Model

This script handles the complete training pipeline including:
- Data loading and preprocessing
- Model configuration and creation
- Training with hyperparameter optimization
- Model evaluation and validation
- Results saving and logging
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from config.config import (
    DataConfig, ModelConfig, TrainingConfig, 
    EvaluationConfig, load_config
)
from src.utils import setup_logging, save_json, ensure_directory
from src.data_preprocessing import GoldDataPreprocessor
from src.lstm_model import LSTMGoldPredictor
from src.model_trainer import ModelTrainer, HyperparameterTuner
from src.evaluation import ModelEvaluator
from src.visualization import Visualizer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Gold Price Prediction Model')
    
    parser.add_argument('--config', type=str, 
                       default='config/model_config.yaml',
                       help='Path to configuration file')
    
    parser.add_argument('--data', type=str,
                       default='data/sample_data.json',
                       help='Path to training data')
    
    parser.add_argument('--output', type=str,
                       default='models/saved_models',
                       help='Output directory for trained model')
    
    parser.add_argument('--tune-hyperparams', action='store_true',
                       help='Enable hyperparameter tuning')
    
    parser.add_argument('--optimization-method', type=str,
                       default='bayesian',
                       choices=['grid', 'random', 'bayesian'],
                       help='Hyperparameter optimization method')
    
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of optimization trials')
    
    parser.add_argument('--cross-validate', action='store_true',
                       help='Enable cross-validation')
    
    parser.add_argument('--save-plots', action='store_true',
                       help='Save visualization plots')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


def setup_directories(output_dir: str):
    """Setup necessary directories."""
    directories = [
        output_dir,
        os.path.join(output_dir, 'checkpoints'),
        os.path.join(output_dir, 'logs'),
        os.path.join(output_dir, 'plots'),
        os.path.join(output_dir, 'metrics')
    ]
    
    for directory in directories:
        ensure_directory(directory)


def load_and_preprocess_data(data_path: str, data_config: DataConfig, logger):
    """Load and preprocess training data."""
    logger.info(f"Loading data from: {data_path}")
    
    # Initialize preprocessor
    preprocessor = GoldDataPreprocessor(data_config)
    
    # Load and prepare data
    data = preprocessor.load_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_training_data(data)
    
    logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return preprocessor, (X_train, X_val, X_test, y_train, y_val, y_test)


def create_model(model_config: ModelConfig, input_shape: tuple, logger):
    """Create LSTM model."""
    logger.info("Creating LSTM model")
    
    model_creator = LSTMGoldPredictor(model_config)
    model = model_creator.build_model(input_shape)
    
    logger.info(f"Model created with architecture: {model_config.architecture}")
    logger.info(f"Total parameters: {model.count_params():,}")
    
    return model_creator, model


def train_model(model, trainer, train_data, val_data, training_config, logger):
    """Train the model."""
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    logger.info("Starting model training")
    
    # Train model
    history, best_model = trainer.train(
        model=model,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        config=training_config
    )
    
    logger.info("Model training completed")
    
    return history, best_model


def optimize_hyperparameters(trainer, train_data, val_data, 
                           training_config, method, n_trials, logger):
    """Optimize hyperparameters."""
    logger.info(f"Starting hyperparameter optimization using {method} method")
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Define hyperparameter space
    param_space = {
        'units': [32, 64, 128, 256],
        'dropout_rate': [0.1, 0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.01, 0.1],
        'batch_size': [16, 32, 64],
        'l2_reg': [0.001, 0.01, 0.1]
    }
    
    # Initialize tuner
    tuner = HyperparameterTuner(trainer)
    
    # Run optimization
    if method == 'grid':
        best_params, best_score = tuner.grid_search(
            param_space, X_train, y_train, X_val, y_val, training_config
        )
    elif method == 'random':
        best_params, best_score = tuner.random_search(
            param_space, X_train, y_train, X_val, y_val, 
            training_config, n_trials
        )
    elif method == 'bayesian':
        best_params, best_score = tuner.bayesian_optimization(
            param_space, X_train, y_train, X_val, y_val, 
            training_config, n_trials
        )
    
    logger.info(f"Best hyperparameters found: {best_params}")
    logger.info(f"Best validation score: {best_score:.4f}")
    
    return best_params, best_score


def evaluate_model(model, evaluator, test_data, preprocessor, 
                  cross_validate, logger):
    """Evaluate trained model."""
    logger.info("Evaluating model performance")
    
    X_test, y_test = test_data
    
    # Generate predictions
    y_pred = model.predict(X_test, verbose=0)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_test, y_pred)
    
    # Perform cross-validation if requested
    if cross_validate:
        logger.info("Performing cross-validation")
        cv_scores = evaluator.cross_validate(model, X_test, y_test)
        metrics['cross_validation'] = cv_scores
    
    # Classification analysis
    classification = evaluator.classify_performance(metrics)
    metrics['performance_classification'] = classification
    
    logger.info(f"Model evaluation completed. MAPE: {metrics.get('mape', 'N/A'):.4f}")
    
    return metrics, y_pred


def save_results(model, preprocessor, history, metrics, 
                best_params, output_dir, timestamp, logger):
    """Save training results."""
    logger.info("Saving training results")
    
    # Save model
    model_path = os.path.join(output_dir, f'gold_price_model_{timestamp}.h5')
    model.save(model_path)
    
    # Save preprocessor
    preprocessor_path = os.path.join(output_dir, f'preprocessor_{timestamp}.pkl')
    import joblib
    joblib.dump(preprocessor, preprocessor_path)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics', f'metrics_{timestamp}.json')
    save_json(metrics, metrics_path)
    
    # Save training history
    history_path = os.path.join(output_dir, 'metrics', f'history_{timestamp}.json')
    save_json(history.history if hasattr(history, 'history') else history, history_path)
    
    # Save hyperparameters
    if best_params:
        params_path = os.path.join(output_dir, 'metrics', f'best_params_{timestamp}.json')
        save_json(best_params, params_path)
    
    # Save metadata
    metadata = {
        'model_path': model_path,
        'preprocessor_path': preprocessor_path,
        'metrics_path': metrics_path,
        'history_path': history_path,
        'timestamp': timestamp,
        'performance_summary': {
            'mape': metrics.get('mape'),
            'r2_score': metrics.get('r2_score'),
            'direction_accuracy': metrics.get('direction_accuracy'),
            'classification': metrics.get('performance_classification')
        }
    }
    
    metadata_path = os.path.join(output_dir, f'training_metadata_{timestamp}.json')
    save_json(metadata, metadata_path)
    
    logger.info(f"Results saved to: {output_dir}")
    
    return {
        'model_path': model_path,
        'preprocessor_path': preprocessor_path,
        'metadata_path': metadata_path
    }


def create_visualizations(data, predictions, metrics, history, 
                         output_dir, timestamp, logger):
    """Create and save visualizations."""
    logger.info("Creating visualizations")
    
    try:
        visualizer = Visualizer()
        plots_dir = os.path.join(output_dir, 'plots')
        
        # Training history plot
        if history:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss plot
            if 'loss' in history:
                ax1.plot(history['loss'], label='Training Loss')
                if 'val_loss' in history:
                    ax1.plot(history['val_loss'], label='Validation Loss')
                ax1.set_title('Model Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()
                ax1.grid(True)
            
            # Metrics plot
            metric_keys = [k for k in history.keys() if 'loss' not in k and 'val_' not in k]
            if metric_keys:
                for key in metric_keys[:3]:  # Limit to 3 metrics
                    ax2.plot(history[key], label=key)
                    if f'val_{key}' in history:
                        ax2.plot(history[f'val_{key}'], label=f'val_{key}')
                
                ax2.set_title('Training Metrics')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Metric Value')
                ax2.legend()
                ax2.grid(True)
            
            plt.tight_layout()
            history_plot_path = os.path.join(plots_dir, f'training_history_{timestamp}.png')
            plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Performance metrics plot
        if metrics:
            visualizer.plot_model_performance(
                metrics,
                save_path=os.path.join(plots_dir, f'performance_metrics_{timestamp}.png')
            )
        
        logger.info(f"Visualizations saved to: {plots_dir}")
        
    except Exception as e:
        logger.warning(f"Could not create visualizations: {e}")


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging('ModelTraining', level=log_level)
    
    logger.info("Starting Gold Price Prediction Model Training")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Output: {args.output}")
    
    try:
        # Load configuration
        config = load_config(args.config)
        data_config = DataConfig(**config.get('data', {}))
        model_config = ModelConfig(**config.get('model', {}))
        training_config = TrainingConfig(**config.get('training', {}))
        evaluation_config = EvaluationConfig(**config.get('evaluation', {}))
        
        # Setup directories
        setup_directories(args.output)
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load and preprocess data
        preprocessor, (X_train, X_val, X_test, y_train, y_val, y_test) = load_and_preprocess_data(
            args.data, data_config, logger
        )
        
        # Create model
        model_creator, model = create_model(model_config, X_train.shape[1:], logger)
        
        # Initialize trainer and evaluator
        trainer = ModelTrainer(model_creator)
        evaluator = ModelEvaluator(evaluation_config)
        
        # Hyperparameter optimization
        best_params = None
        if args.tune_hyperparams:
            best_params, best_score = optimize_hyperparameters(
                trainer, (X_train, y_train), (X_val, y_val),
                training_config, args.optimization_method, args.n_trials, logger
            )
            
            # Update model config with best parameters
            for key, value in best_params.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
            
            # Recreate model with best parameters
            model_creator, model = create_model(model_config, X_train.shape[1:], logger)
        
        # Train model
        history, best_model = train_model(
            model, trainer, (X_train, y_train), (X_val, y_val),
            training_config, logger
        )
        
        # Use best model if available
        final_model = best_model if best_model is not None else model
        
        # Evaluate model
        metrics, y_pred = evaluate_model(
            final_model, evaluator, (X_test, y_test), 
            preprocessor, args.cross_validate, logger
        )
        
        # Save results
        saved_paths = save_results(
            final_model, preprocessor, history, metrics,
            best_params, args.output, timestamp, logger
        )
        
        # Create visualizations
        if args.save_plots:
            create_visualizations(
                None, y_pred, metrics, history.history if hasattr(history, 'history') else history,
                args.output, timestamp, logger
            )
        
        # Print summary
        logger.info("=" * 50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        logger.info(f"Model saved to: {saved_paths['model_path']}")
        logger.info(f"Preprocessor saved to: {saved_paths['preprocessor_path']}")
        logger.info(f"Metadata saved to: {saved_paths['metadata_path']}")
        
        if metrics:
            logger.info("\nPerformance Summary:")
            logger.info(f"MAPE: {metrics.get('mape', 'N/A'):.4f}")
            logger.info(f"R² Score: {metrics.get('r2_score', 'N/A'):.4f}")
            logger.info(f"Direction Accuracy: {metrics.get('direction_accuracy', 'N/A'):.4f}")
            logger.info(f"Performance Classification: {metrics.get('performance_classification', 'N/A')}")
        
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
