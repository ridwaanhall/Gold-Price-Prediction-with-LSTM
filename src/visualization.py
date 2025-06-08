"""
Visualization Module for Gold Price Prediction

Provides comprehensive visualization capabilities including:
- Time series plots
- Prediction visualizations
- Model performance charts
- Interactive dashboards
- Statistical plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
from datetime import datetime, timedelta
import warnings

from .utils import setup_logging, ensure_directory

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class Visualizer:
    """
    Comprehensive visualization class for gold price prediction analysis.
    
    Provides both static (matplotlib/seaborn) and interactive (plotly) visualizations
    for data exploration, model performance, and prediction results.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = setup_logging('Visualizer')
        
        # Visualization settings
        self.figsize = self.config.get('figsize', (12, 8))
        self.dpi = self.config.get('dpi', 100)
        self.style = self.config.get('style', 'whitegrid')
        
        # Color schemes
        self.colors = {
            'actual': '#2E86AB',
            'predicted': '#A23B72',
            'forecast': '#F18F01',
            'confidence': '#C73E1D',
            'trend': '#FFE66D',
            'support': '#06D6A0',
            'resistance': '#EF476F'
        }
        
        # Setup seaborn style
        sns.set_style(self.style)
        
        self.logger.info("Visualizer initialized")
    
    def plot_time_series(self, 
                        data: pd.DataFrame,
                        columns: List[str] = None,
                        title: str = "Gold Price Time Series",
                        save_path: str = None,
                        interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Plot time series data.
        
        Args:
            data: DataFrame with time series data
            columns: Columns to plot
            title: Plot title
            save_path: Path to save plot
            interactive: Whether to create interactive plot
            
        Returns:
            Figure object
        """
        try:
            if columns is None:
                columns = [col for col in data.columns if 'price' in col.lower() or 'harga' in col.lower()]
            
            if interactive:
                return self._plot_interactive_time_series(data, columns, title, save_path)
            else:
                return self._plot_static_time_series(data, columns, title, save_path)
                
        except Exception as e:
            self.logger.error(f"Error plotting time series: {e}")
            raise
    
    def _plot_static_time_series(self, 
                               data: pd.DataFrame,
                               columns: List[str],
                               title: str,
                               save_path: str = None) -> plt.Figure:
        """Create static time series plot."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        for i, col in enumerate(columns):
            if col in data.columns:
                ax.plot(data.index, data[col], 
                       label=col, linewidth=2, alpha=0.8)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (IDR)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Static time series plot saved to {save_path}")
        
        return fig
    
    def _plot_interactive_time_series(self, 
                                    data: pd.DataFrame,
                                    columns: List[str],
                                    title: str,
                                    save_path: str = None) -> go.Figure:
        """Create interactive time series plot."""
        fig = go.Figure()
        
        for col in columns:
            if col in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[col],
                    mode='lines',
                    name=col,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price (IDR)',
            hovermode='x unified',
            template='plotly_white',
            width=1200,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Interactive time series plot saved to {save_path}")
        
        return fig
    
    def plot_predictions(self, 
                        actual: pd.Series,
                        predicted: pd.Series,
                        forecast: pd.Series = None,
                        confidence_intervals: Dict = None,
                        title: str = "Prediction Results",
                        save_path: str = None,
                        interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Plot prediction results with actual vs predicted values.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            forecast: Future forecast values
            confidence_intervals: Confidence interval data
            title: Plot title
            save_path: Path to save plot
            interactive: Whether to create interactive plot
            
        Returns:
            Figure object
        """
        try:
            if interactive:
                return self._plot_interactive_predictions(
                    actual, predicted, forecast, confidence_intervals, title, save_path
                )
            else:
                return self._plot_static_predictions(
                    actual, predicted, forecast, confidence_intervals, title, save_path
                )
                
        except Exception as e:
            self.logger.error(f"Error plotting predictions: {e}")
            raise
    
    def _plot_static_predictions(self, 
                               actual: pd.Series,
                               predicted: pd.Series,
                               forecast: pd.Series = None,
                               confidence_intervals: Dict = None,
                               title: str = "Prediction Results",
                               save_path: str = None) -> plt.Figure:
        """Create static prediction plot."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot actual values
        ax.plot(actual.index, actual.values, 
               color=self.colors['actual'], label='Actual', 
               linewidth=2, alpha=0.8)
        
        # Plot predictions
        ax.plot(predicted.index, predicted.values, 
               color=self.colors['predicted'], label='Predicted', 
               linewidth=2, alpha=0.8)
        
        # Plot forecast if provided
        if forecast is not None:
            ax.plot(forecast.index, forecast.values, 
                   color=self.colors['forecast'], label='Forecast', 
                   linewidth=2, linestyle='--', alpha=0.8)
        
        # Plot confidence intervals if provided
        if confidence_intervals:
            if 'upper' in confidence_intervals and 'lower' in confidence_intervals:
                ax.fill_between(
                    forecast.index if forecast is not None else predicted.index,
                    confidence_intervals['lower'],
                    confidence_intervals['upper'],
                    color=self.colors['confidence'],
                    alpha=0.2,
                    label=f"Confidence Interval ({confidence_intervals.get('level', 0.95)*100:.0f}%)"
                )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (IDR)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Static prediction plot saved to {save_path}")
        
        return fig
    
    def _plot_interactive_predictions(self, 
                                    actual: pd.Series,
                                    predicted: pd.Series,
                                    forecast: pd.Series = None,
                                    confidence_intervals: Dict = None,
                                    title: str = "Prediction Results",
                                    save_path: str = None) -> go.Figure:
        """Create interactive prediction plot."""
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=actual.index,
            y=actual.values,
            mode='lines',
            name='Actual',
            line=dict(color=self.colors['actual'], width=2)
        ))
        
        # Add predictions
        fig.add_trace(go.Scatter(
            x=predicted.index,
            y=predicted.values,
            mode='lines',
            name='Predicted',
            line=dict(color=self.colors['predicted'], width=2)
        ))
        
        # Add forecast if provided
        if forecast is not None:
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast.values,
                mode='lines',
                name='Forecast',
                line=dict(color=self.colors['forecast'], width=2, dash='dash')
            ))
        
        # Add confidence intervals if provided
        if confidence_intervals and 'upper' in confidence_intervals and 'lower' in confidence_intervals:
            x_data = forecast.index if forecast is not None else predicted.index
            
            fig.add_trace(go.Scatter(
                x=x_data,
                y=confidence_intervals['upper'],
                mode='lines',
                name='Upper CI',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=x_data,
                y=confidence_intervals['lower'],
                mode='lines',
                name=f"Confidence Interval ({confidence_intervals.get('level', 0.95)*100:.0f}%)",
                line=dict(width=0),
                fill='tonexty',
                fillcolor=f"rgba{(*plt.colors.to_rgb(self.colors['confidence']), 0.2)}"
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price (IDR)',
            hovermode='x unified',
            template='plotly_white',
            width=1200,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Interactive prediction plot saved to {save_path}")
        
        return fig
    
    def plot_model_performance(self, 
                             metrics: Dict,
                             title: str = "Model Performance Metrics",
                             save_path: str = None) -> plt.Figure:
        """
        Plot model performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Figure object
        """
        try:
            # Create subplots for different metric categories
            fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            # Error metrics
            error_metrics = {k: v for k, v in metrics.items() 
                           if any(x in k.lower() for x in ['mae', 'mse', 'rmse', 'mape'])}
            
            if error_metrics:
                ax = axes[0, 0]
                bars = ax.bar(error_metrics.keys(), error_metrics.values(), 
                             color=sns.color_palette("husl", len(error_metrics)))
                ax.set_title('Error Metrics')
                ax.set_ylabel('Error Value')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}', ha='center', va='bottom')
            
            # Accuracy metrics
            accuracy_metrics = {k: v for k, v in metrics.items() 
                              if any(x in k.lower() for x in ['r2', 'accuracy', 'correlation'])}
            
            if accuracy_metrics:
                ax = axes[0, 1]
                bars = ax.bar(accuracy_metrics.keys(), accuracy_metrics.values(), 
                             color=sns.color_palette("viridis", len(accuracy_metrics)))
                ax.set_title('Accuracy Metrics')
                ax.set_ylabel('Score')
                ax.set_ylim(0, 1.1)
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}', ha='center', va='bottom')
            
            # Trading metrics
            trading_metrics = {k: v for k, v in metrics.items() 
                             if any(x in k.lower() for x in ['direction', 'trend', 'signal'])}
            
            if trading_metrics:
                ax = axes[1, 0]
                bars = ax.bar(trading_metrics.keys(), trading_metrics.values(), 
                             color=sns.color_palette("Set2", len(trading_metrics)))
                ax.set_title('Trading Metrics')
                ax.set_ylabel('Score')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}', ha='center', va='bottom')
            
            # Loss history (if available)
            if 'training_history' in metrics:
                ax = axes[1, 1]
                history = metrics['training_history']
                
                if 'loss' in history:
                    ax.plot(history['loss'], label='Training Loss', color=self.colors['actual'])
                if 'val_loss' in history:
                    ax.plot(history['val_loss'], label='Validation Loss', color=self.colors['predicted'])
                
                ax.set_title('Training History')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.info(f"Performance plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting model performance: {e}")
            raise
    
    def plot_residuals(self, 
                      actual: np.ndarray,
                      predicted: np.ndarray,
                      title: str = "Residual Analysis",
                      save_path: str = None) -> plt.Figure:
        """
        Plot residual analysis.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Figure object
        """
        try:
            residuals = actual - predicted
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            # Residuals vs Predicted
            axes[0, 0].scatter(predicted, residuals, alpha=0.6, color=self.colors['actual'])
            axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.8)
            axes[0, 0].set_xlabel('Predicted Values')
            axes[0, 0].set_ylabel('Residuals')
            axes[0, 0].set_title('Residuals vs Predicted')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Residuals histogram
            axes[0, 1].hist(residuals, bins=30, alpha=0.7, color=self.colors['predicted'])
            axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.8)
            axes[0, 1].set_xlabel('Residuals')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Residuals Distribution')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Residuals over time
            axes[1, 1].plot(range(len(residuals)), residuals, 
                           color=self.colors['forecast'], alpha=0.7)
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.8)
            axes[1, 1].set_xlabel('Time Index')
            axes[1, 1].set_ylabel('Residuals')
            axes[1, 1].set_title('Residuals Over Time')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.info(f"Residuals plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting residuals: {e}")
            raise
    
    def plot_feature_importance(self, 
                              importance_data: Dict,
                              title: str = "Feature Importance",
                              top_n: int = 15,
                              save_path: str = None) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            importance_data: Dictionary with feature importance scores
            title: Plot title
            top_n: Number of top features to show
            save_path: Path to save plot
            
        Returns:
            Figure object
        """
        try:
            if 'importance_scores' not in importance_data:
                raise ValueError("No importance scores found in data")
            
            scores = importance_data['importance_scores']
            
            # Sort features by importance
            sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            features, importances = zip(*sorted_features[:top_n])
            
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            bars = ax.barh(range(len(features)), importances, 
                          color=sns.color_palette("viridis", len(features)))
            
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel('Importance Score')
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2,
                       f'{width:.4f}', ha='left', va='center')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.info(f"Feature importance plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting feature importance: {e}")
            raise
    
    def create_dashboard(self, 
                        data: pd.DataFrame,
                        predictions: Dict,
                        metrics: Dict,
                        save_path: str = None) -> go.Figure:
        """
        Create comprehensive interactive dashboard.
        
        Args:
            data: Historical data
            predictions: Prediction results
            metrics: Model performance metrics
            save_path: Path to save dashboard
            
        Returns:
            Plotly figure object
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Price History', 'Predictions vs Actual',
                              'Performance Metrics', 'Residual Analysis',
                              'Feature Importance', 'Forecast'),
                specs=[[{"secondary_y": True}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]],
                vertical_spacing=0.08
            )
            
            # Price history
            price_cols = [col for col in data.columns if 'price' in col.lower() or 'harga' in col.lower()]
            if price_cols:
                fig.add_trace(
                    go.Scatter(x=data.index, y=data[price_cols[0]], 
                             name='Historical Price', line=dict(color=self.colors['actual'])),
                    row=1, col=1
                )
            
            # Predictions vs Actual (if available)
            if 'single_step' in predictions:
                pred_value = predictions['single_step']['prediction']
                fig.add_trace(
                    go.Scatter(x=[data.index[-1]], y=[pred_value], 
                             mode='markers', name='Prediction',
                             marker=dict(size=10, color=self.colors['predicted'])),
                    row=1, col=2
                )
            
            # Performance metrics
            if metrics:
                metric_names = list(metrics.keys())[:6]  # Top 6 metrics
                metric_values = [metrics[name] for name in metric_names]
                
                fig.add_trace(
                    go.Bar(x=metric_names, y=metric_values, 
                          name='Metrics', marker_color=self.colors['trend']),
                    row=2, col=1
                )
            
            # Multi-step forecast
            if 'multi_step' in predictions:
                forecast_data = predictions['multi_step']
                timestamps = pd.to_datetime(forecast_data['timestamps'])
                
                fig.add_trace(
                    go.Scatter(x=timestamps, y=forecast_data['predictions'],
                             mode='lines+markers', name='Forecast',
                             line=dict(color=self.colors['forecast'], dash='dash')),
                    row=3, col=2
                )
            
            # Update layout
            fig.update_layout(
                title_text="Gold Price Prediction Dashboard",
                title_x=0.5,
                showlegend=True,
                height=1200,
                template='plotly_white'
            )
            
            if save_path:
                fig.write_html(save_path)
                self.logger.info(f"Dashboard saved to {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating dashboard: {e}")
            raise
    
    def plot_correlation_matrix(self, 
                              data: pd.DataFrame,
                              title: str = "Feature Correlation Matrix",
                              save_path: str = None) -> plt.Figure:
        """
        Plot correlation matrix of features.
        
        Args:
            data: DataFrame with features
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Figure object
        """
        try:
            # Calculate correlation matrix
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            corr_matrix = data[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(12, 10), dpi=self.dpi)
            
            # Create heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
                       center=0, square=True, ax=ax, cbar_kws={"shrink": .8})
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.info(f"Correlation matrix saved to {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting correlation matrix: {e}")
            raise
    
    def save_all_plots(self, 
                      data: pd.DataFrame,
                      predictions: Dict,
                      metrics: Dict,
                      output_dir: str):
        """
        Save all visualization plots to directory.
        
        Args:
            data: Input data
            predictions: Prediction results
            metrics: Performance metrics
            output_dir: Output directory
        """
        try:
            output_path = Path(output_dir)
            ensure_directory(str(output_path))
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Time series plot
            self.plot_time_series(
                data, 
                save_path=str(output_path / f"time_series_{timestamp}.png")
            )
            
            # Performance metrics
            if metrics:
                self.plot_model_performance(
                    metrics,
                    save_path=str(output_path / f"performance_{timestamp}.png")
                )
            
            # Correlation matrix
            self.plot_correlation_matrix(
                data,
                save_path=str(output_path / f"correlation_{timestamp}.png")
            )
            
            # Interactive dashboard
            self.create_dashboard(
                data, predictions, metrics,
                save_path=str(output_path / f"dashboard_{timestamp}.html")
            )
            
            self.logger.info(f"All plots saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving plots: {e}")
            raise


class ReportGenerator:
    """
    Generate comprehensive analysis reports with visualizations.
    """
    
    def __init__(self, visualizer: Visualizer):
        """Initialize report generator."""
        self.visualizer = visualizer
        self.logger = setup_logging('ReportGenerator')
    
    def generate_analysis_report(self, 
                               data: pd.DataFrame,
                               predictions: Dict,
                               metrics: Dict,
                               output_path: str):
        """
        Generate comprehensive analysis report.
        
        Args:
            data: Input data
            predictions: Prediction results
            metrics: Performance metrics
            output_path: Output file path
        """
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            
            with PdfPages(output_path) as pdf:
                # Page 1: Time series analysis
                fig1 = self.visualizer.plot_time_series(data, title="Gold Price Analysis")
                pdf.savefig(fig1, bbox_inches='tight')
                plt.close(fig1)
                
                # Page 2: Performance metrics
                if metrics:
                    fig2 = self.visualizer.plot_model_performance(metrics)
                    pdf.savefig(fig2, bbox_inches='tight')
                    plt.close(fig2)
                
                # Page 3: Correlation analysis
                fig3 = self.visualizer.plot_correlation_matrix(data)
                pdf.savefig(fig3, bbox_inches='tight')
                plt.close(fig3)
                
                # Add metadata
                d = pdf.infodict()
                d['Title'] = 'Gold Price Prediction Analysis Report'
                d['Author'] = 'LSTM Gold Price Predictor'
                d['Subject'] = 'Time Series Analysis and Prediction'
                d['Keywords'] = 'Gold Price, LSTM, Machine Learning, Forecasting'
                d['Creator'] = 'Python Visualization Module'
                d['Producer'] = 'Matplotlib PDF Backend'
            
            self.logger.info(f"Analysis report saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating analysis report: {e}")
            raise
