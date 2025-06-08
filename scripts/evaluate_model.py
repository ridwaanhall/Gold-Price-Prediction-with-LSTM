"""
Model Evaluation Script for Gold Price Prediction

This script provides comprehensive evaluation of trained models including:
- Performance metrics calculation
- Statistical analysis
- Visualization generation
- Comparison between models
- Report generation
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from config.config import EvaluationConfig, load_config
from src.utils import setup_logging, save_json, load_json, ensure_directory
from src.data_preprocessing import GoldDataPreprocessor
from src.evaluation import ModelEvaluator
from src.visualization import Visualizer, ReportGenerator
import tensorflow as tf
import joblib


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Gold Price Prediction Model')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    
    parser.add_argument('--preprocessor', type=str, required=True,
                       help='Path to saved preprocessor')
    
    parser.add_argument('--test-data', type=str,
                       default='data/sample_data.json',
                       help='Path to test data')
    
    parser.add_argument('--config', type=str,
                       default='config/model_config.yaml',
                       help='Path to configuration file')
    
    parser.add_argument('--output', type=str,
                       default='models/evaluation',
                       help='Output directory for evaluation results')
    
    parser.add_argument('--cross-validate', action='store_true',
                       help='Perform cross-validation')
    
    parser.add_argument('--walk-forward', action='store_true',
                       help='Perform walk-forward validation')
    
    parser.add_argument('--residual-analysis', action='store_true',
                       help='Perform residual analysis')
    
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate comprehensive evaluation report')
    
    parser.add_argument('--save-plots', action='store_true',
                       help='Save evaluation plots')
    
    parser.add_argument('--compare-models', type=str, nargs='+',
                       help='Paths to additional models for comparison')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


def load_model_and_preprocessor(model_path: str, preprocessor_path: str, logger):
    """Load trained model and preprocessor."""
    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Loading preprocessor from: {preprocessor_path}")
    
    try:
        # Load model
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded successfully. Parameters: {model.count_params():,}")
        
        # Load preprocessor
        preprocessor = joblib.load(preprocessor_path)
        logger.info("Preprocessor loaded successfully")
        
        return model, preprocessor
        
    except Exception as e:
        logger.error(f"Error loading model or preprocessor: {e}")
        raise


def prepare_test_data(test_data_path: str, preprocessor: GoldDataPreprocessor, logger):
    """Prepare test data for evaluation."""
    logger.info(f"Preparing test data from: {test_data_path}")
    
    try:
        # Load test data
        if test_data_path.endswith('.json'):
            data = pd.read_json(test_data_path)
        elif test_data_path.endswith('.csv'):
            data = pd.read_csv(test_data_path)
        else:
            raise ValueError(f"Unsupported file format: {test_data_path}")
        
        # Prepare data using preprocessor
        processed_data = preprocessor.prepare_prediction_data(data)
        
        # For evaluation, we need to split into sequences
        X, y = preprocessor.create_sequences(
            processed_data, 
            preprocessor.config.sequence_length
        )
        
        logger.info(f"Test data prepared. Shape: X={X.shape}, y={y.shape}")
        
        return X, y, data
        
    except Exception as e:
        logger.error(f"Error preparing test data: {e}")
        raise


def evaluate_single_model(model, X_test, y_test, evaluator, 
                         cross_validate, walk_forward, residual_analysis, logger):
    """Evaluate a single model comprehensively."""
    logger.info("Starting comprehensive model evaluation")
    
    results = {}
    
    # Generate predictions
    logger.info("Generating predictions")
    y_pred = model.predict(X_test, verbose=0)
    
    # Basic metrics
    logger.info("Calculating performance metrics")
    metrics = evaluator.calculate_metrics(y_test, y_pred)
    results['basic_metrics'] = metrics
    
    # Cross-validation
    if cross_validate:
        logger.info("Performing cross-validation")
        cv_scores = evaluator.cross_validate(model, X_test, y_test)
        results['cross_validation'] = cv_scores
        logger.info(f"CV Mean Score: {np.mean(list(cv_scores.values())):.4f}")
    
    # Walk-forward validation
    if walk_forward:
        logger.info("Performing walk-forward validation")
        wf_results = evaluator.walk_forward_validation(
            model, X_test, y_test, window_size=30
        )
        results['walk_forward'] = wf_results
        logger.info(f"Walk-forward MAPE: {wf_results.get('mape', 'N/A'):.4f}")
    
    # Residual analysis
    if residual_analysis:
        logger.info("Performing residual analysis")
        residual_stats = evaluator.analyze_residuals(y_test, y_pred)
        results['residual_analysis'] = residual_stats
    
    # Performance classification
    classification = evaluator.classify_performance(metrics)
    results['performance_classification'] = classification
    
    # Statistical significance tests
    logger.info("Performing statistical tests")
    stat_tests = evaluator.statistical_significance_tests(y_test, y_pred)
    results['statistical_tests'] = stat_tests
    
    logger.info("Model evaluation completed")
    
    return results, y_pred


def compare_models(models_info: List[Dict], X_test, y_test, evaluator, logger):
    """Compare multiple models."""
    logger.info(f"Comparing {len(models_info)} models")
    
    comparison_results = {}
    all_predictions = {}
    
    for i, model_info in enumerate(models_info):
        model_name = model_info.get('name', f'Model_{i+1}')
        model = model_info['model']
        
        logger.info(f"Evaluating {model_name}")
        
        # Generate predictions
        y_pred = model.predict(X_test, verbose=0)
        all_predictions[model_name] = y_pred
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(y_test, y_pred)
        comparison_results[model_name] = metrics
    
    # Model comparison analysis
    comparison_summary = {
        'best_model': {},
        'worst_model': {},
        'metrics_comparison': comparison_results
    }
    
    # Find best and worst models for each metric
    for metric in ['mape', 'rmse', 'r2_score', 'direction_accuracy']:
        if metric in list(comparison_results.values())[0]:
            # For error metrics (lower is better)
            if metric in ['mape', 'rmse', 'mae']:
                best_model = min(comparison_results.keys(), 
                               key=lambda x: comparison_results[x][metric])
                worst_model = max(comparison_results.keys(), 
                                key=lambda x: comparison_results[x][metric])
            # For accuracy metrics (higher is better)
            else:
                best_model = max(comparison_results.keys(), 
                               key=lambda x: comparison_results[x][metric])
                worst_model = min(comparison_results.keys(), 
                                key=lambda x: comparison_results[x][metric])
            
            comparison_summary['best_model'][metric] = best_model
            comparison_summary['worst_model'][metric] = worst_model
    
    logger.info("Model comparison completed")
    
    return comparison_summary, all_predictions


def create_evaluation_visualizations(y_test, y_pred, metrics, 
                                   output_dir, timestamp, logger,
                                   all_predictions=None):
    """Create evaluation visualizations."""
    logger.info("Creating evaluation visualizations")
    
    try:
        visualizer = Visualizer()
        plots_dir = os.path.join(output_dir, 'plots')
        ensure_directory(plots_dir)
        
        # Convert to pandas Series for plotting
        actual_series = pd.Series(y_test.flatten(), name='Actual')
        pred_series = pd.Series(y_pred.flatten(), name='Predicted')
        
        # Prediction vs Actual plot
        visualizer.plot_predictions(
            actual_series, pred_series,
            title='Model Predictions vs Actual Values',
            save_path=os.path.join(plots_dir, f'predictions_vs_actual_{timestamp}.png')
        )
        
        # Residual analysis plot
        visualizer.plot_residuals(
            y_test.flatten(), y_pred.flatten(),
            title='Residual Analysis',
            save_path=os.path.join(plots_dir, f'residual_analysis_{timestamp}.png')
        )
        
        # Performance metrics plot
        visualizer.plot_model_performance(
            metrics,
            title='Model Performance Metrics',
            save_path=os.path.join(plots_dir, f'performance_metrics_{timestamp}.png')
        )
        
        # Model comparison plot (if multiple predictions available)
        if all_predictions and len(all_predictions) > 1:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plot actual values
            ax.plot(actual_series.values, label='Actual', 
                   color='black', linewidth=2, alpha=0.8)
            
            # Plot predictions from different models
            colors = plt.cm.Set1(np.linspace(0, 1, len(all_predictions)))
            for (model_name, pred), color in zip(all_predictions.items(), colors):
                ax.plot(pred.flatten(), label=model_name, 
                       color=color, linewidth=1.5, alpha=0.7)
            
            ax.set_title('Model Comparison: Predictions vs Actual', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Time Index')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'model_comparison_{timestamp}.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to: {plots_dir}")
        
    except Exception as e:
        logger.warning(f"Could not create visualizations: {e}")


def generate_evaluation_report(results, y_test, y_pred, output_dir, 
                             timestamp, logger, comparison_results=None):
    """Generate comprehensive evaluation report."""
    logger.info("Generating evaluation report")
    
    try:
        report_lines = [
            "# Gold Price Prediction Model Evaluation Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Basic metrics summary
        if 'basic_metrics' in results:
            metrics = results['basic_metrics']
            report_lines.extend([
                "### Performance Metrics",
                f"- **MAPE (Mean Absolute Percentage Error)**: {metrics.get('mape', 'N/A'):.4f}",
                f"- **RMSE (Root Mean Square Error)**: {metrics.get('rmse', 'N/A'):.4f}",
                f"- **MAE (Mean Absolute Error)**: {metrics.get('mae', 'N/A'):.4f}",
                f"- **R² Score**: {metrics.get('r2_score', 'N/A'):.4f}",
                f"- **Direction Accuracy**: {metrics.get('direction_accuracy', 'N/A'):.4f}",
                ""
            ])
        
        # Performance classification
        if 'performance_classification' in results:
            classification = results['performance_classification']
            report_lines.extend([
                "### Performance Classification",
                f"**Overall Rating**: {classification.get('overall', 'N/A')}",
                ""
            ])
            
            for category, rating in classification.items():
                if category != 'overall':
                    report_lines.append(f"- {category.title()}: {rating}")
            report_lines.append("")
        
        # Cross-validation results
        if 'cross_validation' in results:
            cv_results = results['cross_validation']
            report_lines.extend([
                "### Cross-Validation Results",
                f"- **Mean CV Score**: {np.mean(list(cv_results.values())):.4f}",
                f"- **CV Standard Deviation**: {np.std(list(cv_results.values())):.4f}",
                ""
            ])
        
        # Walk-forward validation
        if 'walk_forward' in results:
            wf_results = results['walk_forward']
            report_lines.extend([
                "### Walk-Forward Validation",
                f"- **Walk-Forward MAPE**: {wf_results.get('mape', 'N/A'):.4f}",
                f"- **Walk-Forward R²**: {wf_results.get('r2_score', 'N/A'):.4f}",
                ""
            ])
        
        # Residual analysis
        if 'residual_analysis' in results:
            residual_stats = results['residual_analysis']
            report_lines.extend([
                "### Residual Analysis",
                f"- **Mean Residual**: {residual_stats.get('mean', 'N/A'):.4f}",
                f"- **Residual Standard Deviation**: {residual_stats.get('std', 'N/A'):.4f}",
                f"- **Normality Test p-value**: {residual_stats.get('normality_pvalue', 'N/A'):.4f}",
                ""
            ])
        
        # Model comparison (if available)
        if comparison_results:
            report_lines.extend([
                "### Model Comparison",
                "Best performing models by metric:",
                ""
            ])
            
            for metric, best_model in comparison_results.get('best_model', {}).items():
                report_lines.append(f"- **{metric.upper()}**: {best_model}")
            
            report_lines.append("")
        
        # Statistical tests
        if 'statistical_tests' in results:
            stat_tests = results['statistical_tests']
            report_lines.extend([
                "### Statistical Significance Tests",
                ""
            ])
            
            for test_name, test_result in stat_tests.items():
                if isinstance(test_result, dict):
                    p_value = test_result.get('p_value', 'N/A')
                    significant = 'Yes' if p_value < 0.05 else 'No'
                    report_lines.append(f"- **{test_name}**: p-value = {p_value:.4f}, Significant = {significant}")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            ""
        ])
        
        # Add recommendations based on performance
        if 'basic_metrics' in results:
            mape = results['basic_metrics'].get('mape', float('inf'))
            r2 = results['basic_metrics'].get('r2_score', 0)
            direction_acc = results['basic_metrics'].get('direction_accuracy', 0)
            
            if mape < 0.03:  # Less than 3%
                report_lines.append("✅ **Excellent MAPE**: Model shows high accuracy in price prediction.")
            elif mape < 0.05:  # Less than 5%
                report_lines.append("✅ **Good MAPE**: Model performance is acceptable for practical use.")
            else:
                report_lines.append("⚠️ **High MAPE**: Consider model improvements or feature engineering.")
            
            if r2 > 0.8:
                report_lines.append("✅ **Strong R² Score**: Model explains variance well.")
            elif r2 > 0.6:
                report_lines.append("✅ **Moderate R² Score**: Model has reasonable explanatory power.")
            else:
                report_lines.append("⚠️ **Low R² Score**: Model may need architectural improvements.")
            
            if direction_acc > 0.7:
                report_lines.append("✅ **Good Direction Accuracy**: Model predicts price movements well.")
            else:
                report_lines.append("⚠️ **Low Direction Accuracy**: Consider trend-focused features.")
        
        report_lines.extend([
            "",
            "## Technical Details",
            "",
            f"- **Test Set Size**: {len(y_test)} samples",
            f"- **Prediction Range**: {y_pred.min():.2f} to {y_pred.max():.2f}",
            f"- **Actual Range**: {y_test.min():.2f} to {y_test.max():.2f}",
            "",
            "---",
            "*Report generated by Gold Price Prediction Evaluation System*"
        ])
        
        # Save report
        report_content = "\n".join(report_lines)
        report_path = os.path.join(output_dir, f'evaluation_report_{timestamp}.md')
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Evaluation report saved to: {report_path}")
        
        return report_path
        
    except Exception as e:
        logger.error(f"Error generating evaluation report: {e}")
        raise


def save_evaluation_results(results, output_dir, timestamp, logger):
    """Save evaluation results to files."""
    logger.info("Saving evaluation results")
    
    try:
        # Save detailed results
        results_path = os.path.join(output_dir, f'evaluation_results_{timestamp}.json')
        save_json(results, results_path)
        
        # Save summary
        summary = {
            'timestamp': timestamp,
            'basic_metrics': results.get('basic_metrics', {}),
            'performance_classification': results.get('performance_classification', {}),
            'evaluation_summary': {
                'mape': results.get('basic_metrics', {}).get('mape'),
                'r2_score': results.get('basic_metrics', {}).get('r2_score'),
                'direction_accuracy': results.get('basic_metrics', {}).get('direction_accuracy'),
                'overall_performance': results.get('performance_classification', {}).get('overall')
            }
        }
        
        summary_path = os.path.join(output_dir, f'evaluation_summary_{timestamp}.json')
        save_json(summary, summary_path)
        
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"Summary saved to: {summary_path}")
        
        return results_path, summary_path
        
    except Exception as e:
        logger.error(f"Error saving evaluation results: {e}")
        raise


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging('ModelEvaluation', level=log_level)
    
    logger.info("Starting Gold Price Prediction Model Evaluation")
    logger.info(f"Model: {args.model}")
    logger.info(f"Preprocessor: {args.preprocessor}")
    logger.info(f"Test Data: {args.test_data}")
    
    try:
        # Setup output directory
        ensure_directory(args.output)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load configuration
        config = load_config(args.config) if os.path.exists(args.config) else {}
        evaluation_config = EvaluationConfig(**config.get('evaluation', {}))
        
        # Load model and preprocessor
        model, preprocessor = load_model_and_preprocessor(
            args.model, args.preprocessor, logger
        )
        
        # Prepare test data
        X_test, y_test, original_data = prepare_test_data(
            args.test_data, preprocessor, logger
        )
        
        # Initialize evaluator
        evaluator = ModelEvaluator(evaluation_config)
        
        # Evaluate main model
        results, y_pred = evaluate_single_model(
            model, X_test, y_test, evaluator,
            args.cross_validate, args.walk_forward, 
            args.residual_analysis, logger
        )
        
        # Model comparison (if additional models provided)
        comparison_results = None
        all_predictions = None
        
        if args.compare_models:
            logger.info("Loading additional models for comparison")
            
            models_info = [{'name': 'Main Model', 'model': model}]
            
            for i, model_path in enumerate(args.compare_models):
                try:
                    comp_model = tf.keras.models.load_model(model_path)
                    models_info.append({
                        'name': f'Model_{i+2}',
                        'model': comp_model
                    })
                    logger.info(f"Loaded comparison model: {model_path}")
                except Exception as e:
                    logger.warning(f"Could not load model {model_path}: {e}")
            
            if len(models_info) > 1:
                comparison_results, all_predictions = compare_models(
                    models_info, X_test, y_test, evaluator, logger
                )
        
        # Create visualizations
        if args.save_plots:
            create_evaluation_visualizations(
                y_test, y_pred, results.get('basic_metrics', {}),
                args.output, timestamp, logger, all_predictions
            )
        
        # Generate report
        if args.generate_report:
            report_path = generate_evaluation_report(
                results, y_test, y_pred, args.output, timestamp, logger,
                comparison_results
            )
        
        # Save results
        results_path, summary_path = save_evaluation_results(
            results, args.output, timestamp, logger
        )
        
        # Print summary
        logger.info("=" * 50)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        
        if 'basic_metrics' in results:
            metrics = results['basic_metrics']
            logger.info("\nPerformance Summary:")
            logger.info(f"MAPE: {metrics.get('mape', 'N/A'):.4f}")
            logger.info(f"RMSE: {metrics.get('rmse', 'N/A'):.4f}")
            logger.info(f"R² Score: {metrics.get('r2_score', 'N/A'):.4f}")
            logger.info(f"Direction Accuracy: {metrics.get('direction_accuracy', 'N/A'):.4f}")
        
        if 'performance_classification' in results:
            classification = results['performance_classification']
            logger.info(f"Overall Performance: {classification.get('overall', 'N/A')}")
        
        logger.info(f"\nResults saved to: {args.output}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
