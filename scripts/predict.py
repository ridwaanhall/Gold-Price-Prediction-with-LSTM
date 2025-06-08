"""
Prediction Script for Gold Price Forecasting

This script provides functionality to make predictions using trained models:
- Single-step predictions
- Multi-step forecasting
- Scenario analysis
- Batch predictions
- Confidence intervals
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from config.config import load_config
from src.utils import setup_logging, save_json, load_json, ensure_directory
from src.prediction import GoldPricePredictor, PredictionPipeline
from src.visualization import Visualizer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Gold Price Prediction')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    
    parser.add_argument('--preprocessor', type=str, required=True,
                       help='Path to saved preprocessor')
    
    parser.add_argument('--input-data', type=str,
                       default='data/sample_data.json',
                       help='Path to input data for prediction')
    
    parser.add_argument('--config', type=str,
                       default='config/model_config.yaml',
                       help='Path to configuration file')
    
    parser.add_argument('--output', type=str,
                       default='predictions',
                       help='Output directory for predictions')
    
    parser.add_argument('--prediction-type', type=str,
                       default='single',
                       choices=['single', 'multi', 'both'],
                       help='Type of prediction to make')
    
    parser.add_argument('--forecast-horizon', type=int, default=7,
                       help='Number of days to forecast (for multi-step)')
    
    parser.add_argument('--confidence-intervals', action='store_true',
                       help='Calculate confidence intervals')
    
    parser.add_argument('--scenarios', type=str,
                       help='Path to scenarios JSON file')
    
    parser.add_argument('--batch-predictions', action='store_true',
                       help='Make batch predictions for multiple inputs')
    
    parser.add_argument('--save-plots', action='store_true',
                       help='Save prediction visualizations')
    
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate prediction report')
    
    parser.add_argument('--feature-importance', action='store_true',
                       help='Calculate feature importance')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


def load_input_data(data_path: str, logger):
    """Load input data for prediction."""
    logger.info(f"Loading input data from: {data_path}")
    
    try:
        if data_path.endswith('.json'):
            data = pd.read_json(data_path)
        elif data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        logger.info(f"Input data loaded. Shape: {data.shape}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading input data: {e}")
        raise


def load_scenarios(scenarios_path: str, logger):
    """Load prediction scenarios."""
    if not scenarios_path or not os.path.exists(scenarios_path):
        return None
    
    logger.info(f"Loading scenarios from: {scenarios_path}")
    
    try:
        scenarios = load_json(scenarios_path)
        logger.info(f"Loaded {len(scenarios)} scenarios")
        return scenarios
        
    except Exception as e:
        logger.warning(f"Could not load scenarios: {e}")
        return None


def make_single_step_prediction(predictor: GoldPricePredictor, 
                              input_data: pd.DataFrame,
                              confidence_intervals: bool,
                              logger):
    """Make single-step prediction."""
    logger.info("Making single-step prediction")
    
    try:
        prediction = predictor.predict_single_step(
            input_data, 
            return_confidence=confidence_intervals
        )
        
        logger.info(f"Single-step prediction: ${prediction['prediction']:.2f}")
        
        if confidence_intervals and 'confidence_interval' in prediction:
            ci = prediction['confidence_interval']
            logger.info(f"Confidence interval ({ci['level']*100}%): "
                       f"${ci['lower']:.2f} - ${ci['upper']:.2f}")
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error in single-step prediction: {e}")
        raise


def make_multi_step_prediction(predictor: GoldPricePredictor,
                             input_data: pd.DataFrame,
                             forecast_horizon: int,
                             logger):
    """Make multi-step prediction."""
    logger.info(f"Making multi-step prediction for {forecast_horizon} days")
    
    try:
        prediction = predictor.predict_multi_step(
            input_data,
            steps=forecast_horizon,
            method='recursive'
        )
        
        logger.info(f"Multi-step forecast completed:")
        for i, (pred, timestamp) in enumerate(zip(
            prediction['predictions'], prediction['timestamps']
        )):
            logger.info(f"  Day {i+1} ({timestamp[:10]}): ${pred:.2f}")
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error in multi-step prediction: {e}")
        raise


def run_scenario_analysis(predictor: GoldPricePredictor,
                         input_data: pd.DataFrame,
                         scenarios: dict,
                         logger):
    """Run scenario analysis."""
    logger.info(f"Running scenario analysis with {len(scenarios)} scenarios")
    
    try:
        scenario_results = predictor.predict_scenarios(input_data, scenarios)
        
        logger.info("Scenario analysis results:")
        base_prediction = scenario_results['base_scenario']['prediction']
        logger.info(f"  Base scenario: ${base_prediction:.2f}")
        
        for scenario_name, result in scenario_results['scenarios'].items():
            prediction = result['prediction']
            change = ((prediction - base_prediction) / base_prediction) * 100
            logger.info(f"  {scenario_name}: ${prediction:.2f} ({change:+.2f}%)")
        
        return scenario_results
        
    except Exception as e:
        logger.error(f"Error in scenario analysis: {e}")
        raise


def calculate_feature_importance(predictor: GoldPricePredictor,
                               input_data: pd.DataFrame,
                               logger):
    """Calculate feature importance."""
    logger.info("Calculating feature importance")
    
    try:
        importance = predictor.get_feature_importance(input_data, method='gradient')
        
        if importance and 'top_features' in importance:
            logger.info("Top important features:")
            for i, feature in enumerate(importance['top_features'][:5]):
                score = importance['importance_scores'].get(feature, 0)
                logger.info(f"  {i+1}. {feature}: {score:.4f}")
        
        return importance
        
    except Exception as e:
        logger.warning(f"Could not calculate feature importance: {e}")
        return {}


def create_prediction_visualizations(predictions: dict,
                                   input_data: pd.DataFrame,
                                   output_dir: str,
                                   timestamp: str,
                                   logger):
    """Create prediction visualizations."""
    logger.info("Creating prediction visualizations")
    
    try:
        visualizer = Visualizer()
        plots_dir = os.path.join(output_dir, 'plots')
        ensure_directory(plots_dir)
        
        # Historical data plot
        price_cols = [col for col in input_data.columns 
                     if 'price' in col.lower() or 'harga' in col.lower()]
        
        if price_cols:
            visualizer.plot_time_series(
                input_data[price_cols],
                title='Historical Gold Prices',
                save_path=os.path.join(plots_dir, f'historical_prices_{timestamp}.png')
            )
        
        # Prediction plots
        if 'single_step' in predictions:
            import matplotlib.pyplot as plt
            
            # Single prediction visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot recent historical data
            if price_cols:
                recent_data = input_data[price_cols[0]].tail(30)
                ax.plot(range(len(recent_data)), recent_data.values, 
                       'b-', label='Historical', linewidth=2)
                
                # Add prediction point
                pred_value = predictions['single_step']['prediction']
                ax.plot(len(recent_data), pred_value, 
                       'ro', markersize=10, label='Prediction')
                
                # Add confidence interval if available
                if 'confidence_interval' in predictions['single_step']:
                    ci = predictions['single_step']['confidence_interval']
                    ax.errorbar(len(recent_data), pred_value,
                              yerr=[[pred_value - ci['lower']], 
                                   [ci['upper'] - pred_value]],
                              capsize=5, capthick=2, 
                              color='red', alpha=0.7)
            
            ax.set_title('Gold Price Prediction', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Index')
            ax.set_ylabel('Price (IDR)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'single_prediction_{timestamp}.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Multi-step forecast plot
        if 'multi_step' in predictions:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot historical data
            if price_cols:
                historical = input_data[price_cols[0]].tail(30)
                ax.plot(range(len(historical)), historical.values,
                       'b-', label='Historical', linewidth=2)
                
                # Plot forecast
                forecast_data = predictions['multi_step']['predictions']
                forecast_x = range(len(historical), len(historical) + len(forecast_data))
                ax.plot(forecast_x, forecast_data,
                       'r--', label='Forecast', linewidth=2, marker='o')
            
            ax.set_title('Multi-Step Gold Price Forecast', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Index')
            ax.set_ylabel('Price (IDR)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'multi_step_forecast_{timestamp}.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Scenario comparison plot
        if 'scenarios' in predictions:
            import matplotlib.pyplot as plt
            
            scenario_results = predictions['scenarios']
            base_pred = predictions.get('base_scenario', {}).get('prediction', 0)
            
            scenarios = list(scenario_results['scenarios'].keys())
            pred_values = [scenario_results['scenarios'][s]['prediction'] for s in scenarios]
            
            # Add base scenario
            scenarios.insert(0, 'Base')
            pred_values.insert(0, base_pred)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bars = ax.bar(scenarios, pred_values, 
                         color=['blue'] + ['orange'] * (len(scenarios)-1),
                         alpha=0.7)
            
            # Add value labels on bars
            for bar, value in zip(bars, pred_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${value:.2f}', ha='center', va='bottom')
            
            ax.set_title('Scenario Analysis Results', fontsize=14, fontweight='bold')
            ax.set_ylabel('Predicted Price (IDR)')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'scenario_analysis_{timestamp}.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to: {plots_dir}")
        
    except Exception as e:
        logger.warning(f"Could not create visualizations: {e}")


def generate_prediction_report(predictions: dict,
                             input_data: pd.DataFrame,
                             output_dir: str,
                             timestamp: str,
                             logger):
    """Generate prediction report."""
    logger.info("Generating prediction report")
    
    try:
        report_lines = [
            "# Gold Price Prediction Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Single-step prediction
        if 'single_step' in predictions:
            single_pred = predictions['single_step']
            report_lines.extend([
                "### Current Price Prediction",
                f"**Predicted Price**: ${single_pred['prediction']:.2f}",
                ""
            ])
            
            if 'confidence_interval' in single_pred:
                ci = single_pred['confidence_interval']
                report_lines.extend([
                    f"**Confidence Interval ({ci['level']*100}%)**: "
                    f"${ci['lower']:.2f} - ${ci['upper']:.2f}",
                    f"**Prediction Uncertainty**: ±${(ci['upper'] - ci['lower'])/2:.2f}",
                    ""
                ])
        
        # Multi-step forecast
        if 'multi_step' in predictions:
            multi_pred = predictions['multi_step']
            report_lines.extend([
                f"### {multi_pred['steps']}-Day Forecast",
                ""
            ])
            
            for i, (pred, timestamp_str) in enumerate(zip(
                multi_pred['predictions'], multi_pred['timestamps']
            )):
                date_str = timestamp_str[:10] if 'T' in timestamp_str else timestamp_str
                report_lines.append(f"- **Day {i+1}** ({date_str}): ${pred:.2f}")
            
            report_lines.extend([
                "",
                f"**Average Daily Change**: ${np.mean(np.diff(multi_pred['predictions'])):.2f}",
                f"**Total Forecast Range**: ${min(multi_pred['predictions']):.2f} - ${max(multi_pred['predictions']):.2f}",
                ""
            ])
        
        # Scenario analysis
        if 'scenarios' in predictions:
            scenario_results = predictions['scenarios']
            base_pred = predictions.get('base_scenario', {}).get('prediction', 0)
            
            report_lines.extend([
                "### Scenario Analysis",
                f"**Base Scenario**: ${base_pred:.2f}",
                ""
            ])
            
            for scenario_name, result in scenario_results['scenarios'].items():
                pred_value = result['prediction']
                change_pct = ((pred_value - base_pred) / base_pred) * 100
                impact = "📈" if change_pct > 0 else "📉" if change_pct < 0 else "➡️"
                
                report_lines.append(
                    f"- **{scenario_name}**: ${pred_value:.2f} "
                    f"({change_pct:+.2f}%) {impact}"
                )
            
            report_lines.append("")
        
        # Feature importance
        if 'feature_importance' in predictions:
            importance = predictions['feature_importance']
            if 'top_features' in importance:
                report_lines.extend([
                    "### Key Price Drivers",
                    ""
                ])
                
                for i, feature in enumerate(importance['top_features'][:5]):
                    score = importance['importance_scores'].get(feature, 0)
                    report_lines.append(f"{i+1}. **{feature}**: {score:.4f}")
                
                report_lines.append("")
        
        # Market insights
        price_cols = [col for col in input_data.columns 
                     if 'price' in col.lower() or 'harga' in col.lower()]
        
        if price_cols:
            current_price = input_data[price_cols[0]].iloc[-1]
            prev_price = input_data[price_cols[0]].iloc[-2] if len(input_data) > 1 else current_price
            recent_change = ((current_price - prev_price) / prev_price) * 100
            
            trend = "🔴 Declining" if recent_change < -1 else "🟢 Rising" if recent_change > 1 else "🟡 Stable"
            
            report_lines.extend([
                "### Market Context",
                f"**Current Market Price**: ${current_price:.2f}",
                f"**Recent Price Movement**: {recent_change:+.2f}% ({trend})",
                ""
            ])
        
        # Recommendations
        report_lines.extend([
            "## Investment Recommendations",
            ""
        ])
        
        if 'single_step' in predictions:
            pred_price = predictions['single_step']['prediction']
            current_price = input_data[price_cols[0]].iloc[-1] if price_cols else pred_price
            
            price_change = ((pred_price - current_price) / current_price) * 100
            
            if price_change > 2:
                report_lines.append("📈 **Buy Signal**: Model predicts significant price increase")
            elif price_change < -2:
                report_lines.append("📉 **Sell Signal**: Model predicts significant price decrease")
            else:
                report_lines.append("⏸️ **Hold**: Model predicts stable prices")
            
            report_lines.extend([
                "",
                f"**Expected Return**: {price_change:+.2f}%",
                ""
            ])
        
        # Risk assessment
        if 'single_step' in predictions and 'uncertainty' in predictions['single_step']:
            uncertainty = predictions['single_step']['uncertainty']
            std_dev = uncertainty['std_dev']
            
            risk_level = "High" if std_dev > current_price * 0.05 else "Medium" if std_dev > current_price * 0.02 else "Low"
            
            report_lines.extend([
                "### Risk Assessment",
                f"**Prediction Uncertainty**: ${std_dev:.2f}",
                f"**Risk Level**: {risk_level}",
                ""
            ])
        
        report_lines.extend([
            "## Disclaimer",
            "",
            "*This report is generated by an AI model and should not be considered as financial advice. "
            "Gold prices are subject to market volatility and various economic factors. "
            "Please consult with a financial advisor before making investment decisions.*",
            "",
            "---",
            "*Report generated by Gold Price Prediction System*"
        ])
        
        # Save report
        report_content = "\n".join(report_lines)
        report_path = os.path.join(output_dir, f'prediction_report_{timestamp}.md')
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Prediction report saved to: {report_path}")
        
        return report_path
        
    except Exception as e:
        logger.error(f"Error generating prediction report: {e}")
        raise


def save_prediction_results(predictions: dict, output_dir: str, timestamp: str, logger):
    """Save prediction results."""
    logger.info("Saving prediction results")
    
    try:
        # Save detailed predictions
        predictions_path = os.path.join(output_dir, f'predictions_{timestamp}.json')
        save_json(predictions, predictions_path)
        
        # Save summary
        summary = {
            'timestamp': timestamp,
            'prediction_types': list(predictions.keys()),
            'generated_at': datetime.now().isoformat()
        }
        
        if 'single_step' in predictions:
            summary['single_step_prediction'] = predictions['single_step']['prediction']
        
        if 'multi_step' in predictions:
            summary['forecast_horizon'] = predictions['multi_step']['steps']
            summary['forecast_range'] = {
                'min': min(predictions['multi_step']['predictions']),
                'max': max(predictions['multi_step']['predictions'])
            }
        
        summary_path = os.path.join(output_dir, f'prediction_summary_{timestamp}.json')
        save_json(summary, summary_path)
        
        logger.info(f"Predictions saved to: {predictions_path}")
        logger.info(f"Summary saved to: {summary_path}")
        
        return predictions_path, summary_path
        
    except Exception as e:
        logger.error(f"Error saving prediction results: {e}")
        raise


def main():
    """Main prediction function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging('GoldPricePrediction', level=log_level)
    
    logger.info("Starting Gold Price Prediction")
    logger.info(f"Model: {args.model}")
    logger.info(f"Preprocessor: {args.preprocessor}")
    logger.info(f"Input Data: {args.input_data}")
    
    try:
        # Setup output directory
        ensure_directory(args.output)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load configuration
        config = load_config(args.config) if os.path.exists(args.config) else {}
        prediction_config = config.get('prediction', {})
        prediction_config.update({
            'sequence_length': config.get('data', {}).get('sequence_length', 30),
            'confidence_level': 0.95,
            'forecast_horizon': args.forecast_horizon
        })
        
        # Initialize predictor
        predictor = GoldPricePredictor(
            model_path=args.model,
            preprocessor_path=args.preprocessor,
            config=prediction_config
        )
        
        # Load input data
        input_data = load_input_data(args.input_data, logger)
        
        # Load scenarios
        scenarios = load_scenarios(args.scenarios, logger)
        
        # Container for all predictions
        all_predictions = {}
        
        # Single-step prediction
        if args.prediction_type in ['single', 'both']:
            single_pred = make_single_step_prediction(
                predictor, input_data, args.confidence_intervals, logger
            )
            all_predictions['single_step'] = single_pred
        
        # Multi-step prediction
        if args.prediction_type in ['multi', 'both']:
            multi_pred = make_multi_step_prediction(
                predictor, input_data, args.forecast_horizon, logger
            )
            all_predictions['multi_step'] = multi_pred
        
        # Scenario analysis
        if scenarios:
            scenario_results = run_scenario_analysis(
                predictor, input_data, scenarios, logger
            )
            all_predictions['scenarios'] = scenario_results
        
        # Feature importance
        if args.feature_importance:
            importance = calculate_feature_importance(predictor, input_data, logger)
            all_predictions['feature_importance'] = importance
        
        # Create visualizations
        if args.save_plots:
            create_prediction_visualizations(
                all_predictions, input_data, args.output, timestamp, logger
            )
        
        # Generate report
        if args.generate_report:
            report_path = generate_prediction_report(
                all_predictions, input_data, args.output, timestamp, logger
            )
        
        # Save results
        predictions_path, summary_path = save_prediction_results(
            all_predictions, args.output, timestamp, logger
        )
        
        # Print summary
        logger.info("=" * 50)
        logger.info("PREDICTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        
        if 'single_step' in all_predictions:
            pred = all_predictions['single_step']['prediction']
            logger.info(f"Single-Step Prediction: ${pred:.2f}")
        
        if 'multi_step' in all_predictions:
            forecast = all_predictions['multi_step']['predictions']
            logger.info(f"Multi-Step Forecast ({len(forecast)} days):")
            logger.info(f"  Range: ${min(forecast):.2f} - ${max(forecast):.2f}")
            logger.info(f"  Average: ${np.mean(forecast):.2f}")
        
        logger.info(f"\nResults saved to: {args.output}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


if __name__ == "__main__":
    main()
