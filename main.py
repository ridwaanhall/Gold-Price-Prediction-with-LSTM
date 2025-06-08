"""
Main Entry Point for Gold Price Prediction System

This is the main entry point that provides a unified interface for all
functionalities of the gold price prediction system.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

from src.utils import setup_logging, ensure_directory
from config.config import load_config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Gold Price Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python main.py train --data data/sample_data.json --tune-hyperparams

  # Evaluate a trained model
  python main.py evaluate --model models/saved_models/model.h5 --preprocessor models/saved_models/preprocessor.pkl

  # Make predictions
  python main.py predict --model models/saved_models/model.h5 --preprocessor models/saved_models/preprocessor.pkl

  # Run complete pipeline
  python main.py pipeline --data data/sample_data.json
        """
    )
    
    # Main command
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--data', type=str, default='data/sample_data.json',
                             help='Path to training data')
    train_parser.add_argument('--config', type=str, default='config/model_config.yaml',
                             help='Path to configuration file')
    train_parser.add_argument('--output', type=str, default='models/saved_models',
                             help='Output directory for trained model')
    train_parser.add_argument('--tune-hyperparams', action='store_true',
                             help='Enable hyperparameter tuning')
    train_parser.add_argument('--optimization-method', type=str, default='bayesian',
                             choices=['grid', 'random', 'bayesian'],
                             help='Hyperparameter optimization method')
    train_parser.add_argument('--n-trials', type=int, default=50,
                             help='Number of optimization trials')
    train_parser.add_argument('--cross-validate', action='store_true',
                             help='Enable cross-validation')
    train_parser.add_argument('--save-plots', action='store_true',
                             help='Save training visualizations')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model', type=str, required=True,
                            help='Path to trained model')
    eval_parser.add_argument('--preprocessor', type=str, required=True,
                            help='Path to saved preprocessor')
    eval_parser.add_argument('--test-data', type=str, default='data/sample_data.json',
                            help='Path to test data')
    eval_parser.add_argument('--config', type=str, default='config/model_config.yaml',
                            help='Path to configuration file')
    eval_parser.add_argument('--output', type=str, default='models/evaluation',
                            help='Output directory for evaluation results')
    eval_parser.add_argument('--cross-validate', action='store_true',
                            help='Perform cross-validation')
    eval_parser.add_argument('--walk-forward', action='store_true',
                            help='Perform walk-forward validation')
    eval_parser.add_argument('--residual-analysis', action='store_true',
                            help='Perform residual analysis')
    eval_parser.add_argument('--generate-report', action='store_true',
                            help='Generate comprehensive evaluation report')
    eval_parser.add_argument('--save-plots', action='store_true',
                            help='Save evaluation plots')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model', type=str, required=True,
                               help='Path to trained model')
    predict_parser.add_argument('--preprocessor', type=str, required=True,
                               help='Path to saved preprocessor')
    predict_parser.add_argument('--input-data', type=str, default='data/sample_data.json',
                               help='Path to input data for prediction')
    predict_parser.add_argument('--config', type=str, default='config/model_config.yaml',
                               help='Path to configuration file')
    predict_parser.add_argument('--output', type=str, default='predictions',
                               help='Output directory for predictions')
    predict_parser.add_argument('--prediction-type', type=str, default='both',
                               choices=['single', 'multi', 'both'],
                               help='Type of prediction to make')
    predict_parser.add_argument('--forecast-horizon', type=int, default=7,
                               help='Number of days to forecast')
    predict_parser.add_argument('--confidence-intervals', action='store_true',
                               help='Calculate confidence intervals')
    predict_parser.add_argument('--scenarios', type=str,
                               help='Path to scenarios JSON file')
    predict_parser.add_argument('--save-plots', action='store_true',
                               help='Save prediction visualizations')
    predict_parser.add_argument('--generate-report', action='store_true',
                               help='Generate prediction report')
    predict_parser.add_argument('--feature-importance', action='store_true',
                               help='Calculate feature importance')
    
    # Pipeline command (runs complete workflow)
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete pipeline')
    pipeline_parser.add_argument('--data', type=str, default='data/sample_data.json',
                                help='Path to training data')
    pipeline_parser.add_argument('--config', type=str, default='config/model_config.yaml',
                                help='Path to configuration file')
    pipeline_parser.add_argument('--output', type=str, default='models',
                                help='Base output directory')
    pipeline_parser.add_argument('--tune-hyperparams', action='store_true',
                                help='Enable hyperparameter tuning')
    pipeline_parser.add_argument('--quick-run', action='store_true',
                                help='Run with minimal settings for quick testing')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    
    # Global options
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--log-file', type=str,
                       help='Log file path')
    
    return parser.parse_args()


def setup_logging_system(verbose: bool = False, log_file: str = None):
    """Setup logging for the main system."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory
    if log_file:
        ensure_directory(os.path.dirname(log_file))
    else:
        ensure_directory('logs')
        log_file = f"logs/gold_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Setup logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('GoldPredictionSystem')


def run_train_command(args, logger):
    """Run training command."""
    logger.info("Starting training workflow")
    
    # Import and run training script
    from scripts.train_model import main as train_main
    
    # Convert args to sys.argv format for the script
    sys.argv = ['train_model.py']
    sys.argv.extend(['--data', args.data])
    sys.argv.extend(['--config', args.config])
    sys.argv.extend(['--output', args.output])
    
    if args.tune_hyperparams:
        sys.argv.append('--tune-hyperparams')
        sys.argv.extend(['--optimization-method', args.optimization_method])
        sys.argv.extend(['--n-trials', str(args.n_trials)])
    
    if args.cross_validate:
        sys.argv.append('--cross-validate')
    
    if args.save_plots:
        sys.argv.append('--save-plots')
    
    if args.verbose:
        sys.argv.append('--verbose')
    
    try:
        train_main()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def run_evaluate_command(args, logger):
    """Run evaluation command."""
    logger.info("Starting evaluation workflow")
    
    # Import and run evaluation script
    from scripts.evaluate_model import main as eval_main
    
    # Convert args to sys.argv format for the script
    sys.argv = ['evaluate_model.py']
    sys.argv.extend(['--model', args.model])
    sys.argv.extend(['--preprocessor', args.preprocessor])
    sys.argv.extend(['--test-data', args.test_data])
    sys.argv.extend(['--config', args.config])
    sys.argv.extend(['--output', args.output])
    
    if args.cross_validate:
        sys.argv.append('--cross-validate')
    
    if args.walk_forward:
        sys.argv.append('--walk-forward')
    
    if args.residual_analysis:
        sys.argv.append('--residual-analysis')
    
    if args.generate_report:
        sys.argv.append('--generate-report')
    
    if args.save_plots:
        sys.argv.append('--save-plots')
    
    if args.verbose:
        sys.argv.append('--verbose')
    
    try:
        eval_main()
        logger.info("Evaluation completed successfully")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def run_predict_command(args, logger):
    """Run prediction command."""
    logger.info("Starting prediction workflow")
    
    # Import and run prediction script
    from scripts.predict import main as predict_main
    
    # Convert args to sys.argv format for the script
    sys.argv = ['predict.py']
    sys.argv.extend(['--model', args.model])
    sys.argv.extend(['--preprocessor', args.preprocessor])
    sys.argv.extend(['--input-data', args.input_data])
    sys.argv.extend(['--config', args.config])
    sys.argv.extend(['--output', args.output])
    sys.argv.extend(['--prediction-type', args.prediction_type])
    sys.argv.extend(['--forecast-horizon', str(args.forecast_horizon)])
    
    if args.confidence_intervals:
        sys.argv.append('--confidence-intervals')
    
    if args.scenarios:
        sys.argv.extend(['--scenarios', args.scenarios])
    
    if args.save_plots:
        sys.argv.append('--save-plots')
    
    if args.generate_report:
        sys.argv.append('--generate-report')
    
    if args.feature_importance:
        sys.argv.append('--feature-importance')
    
    if args.verbose:
        sys.argv.append('--verbose')
    
    try:
        predict_main()
        logger.info("Prediction completed successfully")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


def run_pipeline_command(args, logger):
    """Run complete pipeline."""
    logger.info("Starting complete pipeline workflow")
    
    try:
        # Setup output directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pipeline_output = os.path.join(args.output, f'pipeline_{timestamp}')
        train_output = os.path.join(pipeline_output, 'training')
        eval_output = os.path.join(pipeline_output, 'evaluation')
        pred_output = os.path.join(pipeline_output, 'predictions')
        
        ensure_directory(train_output)
        ensure_directory(eval_output)
        ensure_directory(pred_output)
        
        logger.info(f"Pipeline output directory: {pipeline_output}")
        
        # Step 1: Training
        logger.info("=" * 50)
        logger.info("STEP 1: MODEL TRAINING")
        logger.info("=" * 50)
        
        train_args = argparse.Namespace(
            data=args.data,
            config=args.config,
            output=train_output,
            tune_hyperparams=args.tune_hyperparams and not args.quick_run,
            optimization_method='bayesian',
            n_trials=10 if args.quick_run else 50,
            cross_validate=not args.quick_run,
            save_plots=True,
            verbose=args.verbose
        )
        
        run_train_command(train_args, logger)
        
        # Find the trained model and preprocessor
        import glob
        model_files = glob.glob(os.path.join(train_output, '*.h5'))
        preprocessor_files = glob.glob(os.path.join(train_output, '*.pkl'))
        
        if not model_files or not preprocessor_files:
            raise FileNotFoundError("Could not find trained model or preprocessor files")
        
        model_path = model_files[0]  # Use the most recent
        preprocessor_path = preprocessor_files[0]
        
        logger.info(f"Using trained model: {model_path}")
        logger.info(f"Using preprocessor: {preprocessor_path}")
        
        # Step 2: Evaluation
        logger.info("=" * 50)
        logger.info("STEP 2: MODEL EVALUATION")
        logger.info("=" * 50)
        
        eval_args = argparse.Namespace(
            model=model_path,
            preprocessor=preprocessor_path,
            test_data=args.data,
            config=args.config,
            output=eval_output,
            cross_validate=not args.quick_run,
            walk_forward=not args.quick_run,
            residual_analysis=True,
            generate_report=True,
            save_plots=True,
            verbose=args.verbose
        )
        
        run_evaluate_command(eval_args, logger)
        
        # Step 3: Predictions
        logger.info("=" * 50)
        logger.info("STEP 3: MAKING PREDICTIONS")
        logger.info("=" * 50)
        
        pred_args = argparse.Namespace(
            model=model_path,
            preprocessor=preprocessor_path,
            input_data=args.data,
            config=args.config,
            output=pred_output,
            prediction_type='both',
            forecast_horizon=7,
            confidence_intervals=True,
            scenarios=None,
            save_plots=True,
            generate_report=True,
            feature_importance=True,
            verbose=args.verbose
        )
        
        run_predict_command(pred_args, logger)
        
        # Generate pipeline summary
        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        logger.info(f"All results saved to: {pipeline_output}")
        logger.info(f"- Training results: {train_output}")
        logger.info(f"- Evaluation results: {eval_output}")
        logger.info(f"- Prediction results: {pred_output}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


def show_system_info(logger):
    """Show system information."""
    logger.info("Gold Price Prediction System Information")
    logger.info("=" * 50)
    
    # System info
    import platform
    import tensorflow as tf
    
    logger.info(f"Python Version: {platform.python_version()}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"TensorFlow Version: {tf.__version__}")
    
    # GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"GPU Available: {len(gpus) > 0}")
    if gpus:
        for i, gpu in enumerate(gpus):
            logger.info(f"  GPU {i}: {gpu.name}")
    
    # Project structure
    logger.info("\nProject Structure:")
    project_root = Path(__file__).parent
    
    for item in sorted(project_root.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            logger.info(f"  📁 {item.name}/")
            # Show some key files in each directory
            key_files = list(item.glob('*.py'))[:3]  # Show first 3 Python files
            for file in key_files:
                logger.info(f"    📄 {file.name}")
            if len(list(item.glob('*.py'))) > 3:
                logger.info(f"    ... and {len(list(item.glob('*.py'))) - 3} more files")
    
    # Configuration info
    config_file = project_root / 'config' / 'model_config.yaml'
    if config_file.exists():
        logger.info(f"\nConfiguration file found: {config_file}")
        try:
            config = load_config(str(config_file))
            logger.info(f"Model architecture: {config.get('model', {}).get('architecture', 'N/A')}")
            logger.info(f"Sequence length: {config.get('data', {}).get('sequence_length', 'N/A')}")
        except Exception as e:
            logger.warning(f"Could not load configuration: {e}")
    
    logger.info("=" * 50)


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging_system(args.verbose, args.log_file)
    
    # Print banner
    logger.info("=" * 60)
    logger.info("🏆 GOLD PRICE PREDICTION SYSTEM")
    logger.info("🤖 LSTM-Based Time Series Forecasting")
    logger.info("=" * 60)
    
    try:
        if not args.command:
            logger.error("No command specified. Use --help for available commands.")
            return
        
        # Route to appropriate command
        if args.command == 'train':
            run_train_command(args, logger)
        elif args.command == 'evaluate':
            run_evaluate_command(args, logger)
        elif args.command == 'predict':
            run_predict_command(args, logger)
        elif args.command == 'pipeline':
            run_pipeline_command(args, logger)
        elif args.command == 'info':
            show_system_info(logger)
        else:
            logger.error(f"Unknown command: {args.command}")
        
        logger.info("🎉 Operation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
