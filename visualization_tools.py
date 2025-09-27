#!/usr/bin/env python3
"""
Visualization Tools Module for Oil Price Prediction
Handles all plotting and visualization functions - Fixed Version
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

warnings.filterwarnings('ignore')

# Set plotting style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn-darkgrid')

sns.set_palette("husl")


class PredictionVisualizer:
    """
    Comprehensive visualization tools for oil price prediction analysis
    """

    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize the visualizer

        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.colors = {
            'historical': '#2E86C1',
            'predicted': '#E74C3C',
            'trend': '#28B463',
            'confidence': '#F39C12',
            'support': '#8E44AD',
            'resistance': '#D35400'
        }

    def plot_price_history_with_predictions(self, dates: pd.DatetimeIndex,
                                            historical_prices: np.ndarray,
                                            predictions: np.ndarray,
                                            confidence_intervals: Optional[Dict] = None,
                                            n_history: int = 200,
                                            title: str = "Oil Price Prediction") -> None:
        """
        Plot historical prices with future predictions

        Args:
            dates: Historical date index
            historical_prices: Historical price data
            predictions: Future price predictions
            confidence_intervals: Optional confidence intervals
            n_history: Number of historical points to show
            title: Plot title
        """
        try:
            fig, ax = plt.subplots(figsize=self.figsize)

            # Validate inputs
            if len(dates) == 0 or len(historical_prices) == 0:
                ax.text(0.5, 0.5, 'No historical data available',
                        transform=ax.transAxes, ha='center', va='center', fontsize=16)
                plt.title(title, fontsize=16, fontweight='bold')
                plt.show()
                return

            if len(predictions) == 0:
                ax.text(0.5, 0.5, 'No prediction data available',
                        transform=ax.transAxes, ha='center', va='center', fontsize=16)
                plt.title(title, fontsize=16, fontweight='bold')
                plt.show()
                return

            # Prepare data
            recent_dates = dates[-n_history:] if len(dates) > n_history else dates
            recent_prices = historical_prices[-n_history:] if len(historical_prices) > n_history else historical_prices

            # Ensure we have matching lengths
            min_len = min(len(recent_dates), len(recent_prices))
            recent_dates = recent_dates[:min_len]
            recent_prices = recent_prices[:min_len]

            # Future dates
            last_date = dates[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                         periods=len(predictions), freq='D')

            # Plot historical data
            ax.plot(recent_dates, recent_prices, color=self.colors['historical'],
                    linewidth=2, label='Historical Prices', alpha=0.8)

            # Plot predictions
            ax.plot(future_dates, predictions, color=self.colors['predicted'],
                    linewidth=2, linestyle='--', label='Predicted Prices', alpha=0.8)

            # Add confidence intervals if available
            if confidence_intervals and isinstance(confidence_intervals, dict):
                ci_95 = confidence_intervals.get('95%', {})
                if ci_95 and 'lower_bound' in ci_95 and 'upper_bound' in ci_95:
                    lower = ci_95['lower_bound']
                    upper = ci_95['upper_bound']

                    # Handle case where bounds are scalars
                    if np.isscalar(lower):
                        lower = [lower] * len(future_dates)
                    if np.isscalar(upper):
                        upper = [upper] * len(future_dates)

                    # Ensure matching lengths
                    min_len = min(len(future_dates), len(lower), len(upper))
                    ax.fill_between(future_dates[:min_len], lower[:min_len], upper[:min_len],
                                    color=self.colors['confidence'], alpha=0.3,
                                    label='95% Confidence Interval')

            # Mark transition point
            ax.axvline(x=dates[-1], color='gray', linestyle=':', alpha=0.7, label='Today')

            # Formatting
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Oil Price ($)', fontsize=12)
            ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3)

            # Format x-axis
            try:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                plt.xticks(rotation=45)
            except:
                # Fallback if date formatting fails
                pass

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error plotting price history: {str(e)}")
            # Create simple fallback plot
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, f'Error creating plot: {str(e)[:50]}...',
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
            plt.title(title, fontsize=16, fontweight='bold')
            plt.show()

    def plot_wavelet_decomposition(self, signal: np.ndarray,
                                   components: Dict[str, np.ndarray],
                                   dates: Optional[pd.DatetimeIndex] = None,
                                   title: str = "Wavelet Decomposition") -> None:
        """
        Plot wavelet decomposition results

        Args:
            signal: Original signal
            components: Wavelet components
            dates: Optional datetime index
            title: Plot title
        """
        try:
            if len(components) == 0:
                print("No wavelet components to plot")
                return

            n_components = len(components)
            fig, axes = plt.subplots(n_components + 1, 1, figsize=(self.figsize[0], 4 * (n_components + 1)))

            # Handle single subplot case
            if n_components == 0:
                axes = [axes]

            if dates is None:
                dates = range(len(signal))

            # Original signal
            axes[0].plot(dates, signal, color='black', linewidth=2)
            axes[0].set_title(f'{title} - Original Signal', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Price ($)', fontsize=10)
            axes[0].grid(True, alpha=0.3)

            # Add price statistics to original signal
            if len(signal) > 0:
                mean_price = np.mean(signal)
                axes[0].axhline(mean_price, color='red', linestyle='--', alpha=0.7,
                                label=f'Mean: ${mean_price:.2f}')
                axes[0].legend()

            # Components
            colors = plt.cm.tab10(np.linspace(0, 1, n_components))

            for i, (name, component) in enumerate(components.items()):
                if i + 1 >= len(axes):
                    break

                ax = axes[i + 1]

                # Ensure component and dates have compatible lengths
                plot_dates = dates[:len(component)] if hasattr(dates, '__len__') else range(len(component))

                ax.plot(plot_dates, component, color=colors[i], linewidth=1.5)

                # Calculate component statistics safely
                try:
                    mean_val = np.mean(component) if len(component) > 0 else 0
                    std_val = np.std(component) if len(component) > 0 else 0
                    variance = np.var(component) if len(component) > 0 else 0
                    total_variance = np.var(signal) if len(signal) > 0 else 1
                    contribution = (variance / total_variance * 100) if total_variance > 0 else 0
                except:
                    mean_val = std_val = variance = contribution = 0

                # Title with statistics
                if name == 'trend':
                    component_title = f'{name.title()} Component (μ=${mean_val:.2f}, σ=${std_val:.2f})'
                else:
                    component_title = f'{name.replace("_", " ").title()} Component (σ={std_val:.2f}, {contribution:.1f}% var)'

                ax.set_title(component_title, fontsize=12)
                ax.set_ylabel('Amplitude', fontsize=10)
                ax.grid(True, alpha=0.3)

                # Add zero line for detail components
                if name != 'trend':
                    ax.axhline(0, color='red', linestyle='--', alpha=0.5)

                # Add component statistics as text
                ax.text(0.02, 0.95, f'Contribution: {contribution:.1f}%',
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Format x-axis for last subplot
            if isinstance(dates, pd.DatetimeIndex):
                try:
                    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                    plt.xticks(rotation=45)
                except:
                    pass

            axes[-1].set_xlabel('Date' if isinstance(dates, pd.DatetimeIndex) else 'Time', fontsize=12)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error plotting wavelet decomposition: {str(e)}")

    def plot_training_history(self, histories: Dict[str, Any],
                              title: str = "Model Training History") -> None:
        """
        Plot training history for all models

        Args:
            histories: Dictionary of training histories
            title: Plot title
        """
        try:
            if not histories:
                print("No training histories to plot")
                return

            n_models = len(histories)
            fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))

            # Handle single model case
            if n_models == 1:
                axes = axes.reshape(2, 1)

            colors = plt.cm.tab10(np.linspace(0, 1, n_models))

            for i, (name, history) in enumerate(histories.items()):
                if i >= axes.shape[1]:
                    break

                color = colors[i]

                # Check if history has the expected structure
                if not hasattr(history, 'history') or not isinstance(history.history, dict):
                    axes[0, i].text(0.5, 0.5, f'Invalid history\nfor {name}',
                                    transform=axes[0, i].transAxes, ha='center', va='center')
                    axes[1, i].text(0.5, 0.5, f'No data available',
                                    transform=axes[1, i].transAxes, ha='center', va='center')
                    continue

                # Loss plot
                if 'loss' in history.history and 'val_loss' in history.history:
                    epochs = range(1, len(history.history['loss']) + 1)
                    axes[0, i].plot(epochs, history.history['loss'], color=color,
                                    linewidth=2, label='Training Loss')
                    axes[0, i].plot(epochs, history.history['val_loss'], color=color,
                                    linewidth=2, linestyle='--', label='Validation Loss')

                    axes[0, i].set_title(f'{name.replace("_", " ").title()}\nLoss',
                                         fontweight='bold', fontsize=12)
                    axes[0, i].set_ylabel('Loss (MSE)')
                    axes[0, i].legend()
                    axes[0, i].grid(True, alpha=0.3)
                    axes[0, i].set_yscale('log')  # Log scale for better visualization

                    # Add training summary text
                    try:
                        final_loss = history.history['val_loss'][-1]
                        best_loss = min(history.history['val_loss'])
                        summary_text = f'Final: {final_loss:.4f}\nBest: {best_loss:.4f}'
                        axes[0, i].text(0.02, 0.98, summary_text, transform=axes[0, i].transAxes,
                                        verticalalignment='top', bbox=dict(boxstyle='round',
                                                                           facecolor='white', alpha=0.8), fontsize=8)
                    except:
                        pass

                # MAE plot
                if 'mae' in history.history and 'val_mae' in history.history:
                    epochs = range(1, len(history.history['mae']) + 1)
                    axes[1, i].plot(epochs, history.history['mae'], color=color,
                                    linewidth=2, label='Training MAE')
                    axes[1, i].plot(epochs, history.history['val_mae'], color=color,
                                    linewidth=2, linestyle='--', label='Validation MAE')

                    axes[1, i].set_title('Mean Absolute Error', fontweight='bold', fontsize=12)
                    axes[1, i].set_ylabel('MAE')
                    axes[1, i].legend()
                    axes[1, i].grid(True, alpha=0.3)
                else:
                    axes[1, i].text(0.5, 0.5, 'MAE data\nnot available',
                                    transform=axes[1, i].transAxes, ha='center', va='center')

                axes[1, i].set_xlabel('Epoch')

            plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error plotting training history: {str(e)}")

    def plot_risk_analysis_dashboard(self, risk_report: Dict[str, Any],
                                     prices: np.ndarray,
                                     predictions: np.ndarray,
                                     title: str = "Risk Analysis Dashboard") -> None:
        """
        Plot comprehensive risk analysis dashboard

        Args:
            risk_report: Risk analysis report
            prices: Historical prices
            predictions: Future predictions
            title: Plot title
        """
        try:
            fig = plt.figure(figsize=(18, 12))
            gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.25)

            # 1. Price History with VaR
            ax1 = fig.add_subplot(gs[0, :2])

            if len(prices) > 0:
                recent_prices = prices[-60:] if len(prices) > 60 else prices
                ax1.plot(range(len(recent_prices)), recent_prices, color=self.colors['historical'],
                         linewidth=2, label='Price History')

                current_price = risk_report.get('current_price', prices[-1] if len(prices) > 0 else 0)
                ax1.axhline(current_price, color='black', linestyle='-', alpha=0.7,
                            label=f'Current: ${current_price:.2f}')

                # Add VaR if available
                var_1d = risk_report.get('var_metrics', {}).get('1_day', {})
                if var_1d and 'var_historical_dollar' in var_1d:
                    var_dollar = var_1d['var_historical_dollar']
                    ax1.axhline(current_price - var_dollar, color='red', linestyle='--',
                                alpha=0.7, label=f'VaR: ${var_dollar:.2f}')
                    ax1.fill_between(range(len(recent_prices)), current_price - var_dollar,
                                     current_price, alpha=0.2, color='red')
            else:
                ax1.text(0.5, 0.5, 'No price data available', transform=ax1.transAxes,
                         ha='center', va='center')

            ax1.set_title('Price History with VaR', fontweight='bold', fontsize=12)
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Prediction Distribution
            ax2 = fig.add_subplot(gs[0, 2:])

            if len(predictions) > 0:
                current_price = risk_report.get('current_price', 100)  # Fallback value
                if current_price > 0:
                    pred_returns = (predictions - current_price) / current_price * 100

                    n, bins, patches = ax2.hist(pred_returns, bins=20, alpha=0.7,
                                                color=self.colors['predicted'], edgecolor='black', density=True)

                    # Color bars based on returns
                    for i, p in enumerate(patches):
                        if i < len(bins) - 1:
                            if bins[i] < -2:
                                p.set_facecolor('red')
                            elif bins[i] > 2:
                                p.set_facecolor('green')
                            else:
                                p.set_facecolor('orange')

                    ax2.axvline(0, color='black', linestyle='--', alpha=0.7, label='No Change')
                    mean_return = np.mean(pred_returns)
                    ax2.axvline(mean_return, color='blue', linestyle='-', alpha=0.7,
                                label=f'Mean: {mean_return:.1f}%')
                else:
                    ax2.text(0.5, 0.5, 'Invalid current price', transform=ax2.transAxes,
                             ha='center', va='center')
            else:
                ax2.text(0.5, 0.5, 'No prediction data', transform=ax2.transAxes,
                         ha='center', va='center')

            ax2.set_title('Prediction Returns Distribution', fontweight='bold', fontsize=12)
            ax2.set_xlabel('Expected Return (%)')
            ax2.set_ylabel('Density')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 3. Risk Summary Panel
            ax3 = fig.add_subplot(gs[1, :])
            ax3.axis('off')

            # Create risk summary text
            risk_text = []
            risk_text.append("📊 RISK ANALYSIS SUMMARY")
            risk_text.append("=" * 60)

            overall_risk = risk_report.get('overall_risk', 'UNKNOWN')
            risk_score = risk_report.get('risk_score', 0)
            risk_text.append(f"Overall Risk Level: {overall_risk}")
            risk_text.append(f"Risk Score: {risk_score:.1f}/100")

            # Volatility metrics
            vol_metrics = risk_report.get('volatility_metrics', {})
            if vol_metrics:
                annual_vol = vol_metrics.get('annual_volatility', 0) * 100
                risk_text.append(f"Annual Volatility: {annual_vol:.1f}%")

            # VaR metrics
            var_1d = risk_report.get('var_metrics', {}).get('1_day', {})
            if var_1d:
                var_amount = var_1d.get('var_historical_dollar', 0)
                risk_text.append(f"1-Day VaR: ${var_amount:.2f}")

            # Prediction risk
            pred_risk = risk_report.get('prediction_risk', {})
            if pred_risk:
                pred_vol = pred_risk.get('volatility_prediction', 0) * 100
                prob_positive = pred_risk.get('prob_positive', 0) * 100
                risk_text.append(f"Prediction Volatility: {pred_vol:.1f}%")
                risk_text.append(f"Probability of Gain: {prob_positive:.0f}%")

            # Risk factors
            risk_factors = risk_report.get('risk_factors', [])
            if risk_factors:
                risk_text.append("\nKey Risk Factors:")
                for factor in risk_factors[:3]:
                    risk_text.append(f"  • {factor}")

            ax3.text(0.05, 0.95, '\n'.join(risk_text), transform=ax3.transAxes,
                     verticalalignment='top', fontfamily='monospace', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))

            # 4. Simple Volatility Chart
            ax4 = fig.add_subplot(gs[2, :])

            if vol_metrics:
                vol_data = {
                    '30D': vol_metrics.get('volatility_30d', 0) * 100,
                    '60D': vol_metrics.get('volatility_60d', 0) * 100,
                    '90D': vol_metrics.get('volatility_90d', 0) * 100,
                    'Annual': vol_metrics.get('annual_volatility', 0) * 100
                }

                if any(vol_data.values()):
                    bars = ax4.bar(vol_data.keys(), vol_data.values(),
                                   color=['lightblue', 'skyblue', 'steelblue', 'navy'], alpha=0.8)

                    for bar, (label, value) in zip(bars, vol_data.items()):
                        height = bar.get_height()
                        if height > 0:
                            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                                     f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

                    ax4.set_title('Volatility Analysis', fontweight='bold', fontsize=12)
                    ax4.set_ylabel('Volatility (%)')
                    ax4.grid(True, alpha=0.3)
                else:
                    ax4.text(0.5, 0.5, 'No volatility data available',
                             transform=ax4.transAxes, ha='center', va='center')
            else:
                ax4.text(0.5, 0.5, 'No volatility metrics available',
                         transform=ax4.transAxes, ha='center', va='center')

            plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error creating risk analysis dashboard: {str(e)}")

    def plot_trading_signals_dashboard(self, signals: Dict[str, Any],
                                       current_price: float,
                                       predictions: np.ndarray,
                                       title: str = "Trading Signals Dashboard") -> None:
        """
        Plot trading signals dashboard
        """
        try:
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

            # 1. Price Chart with Signals
            ax1 = fig.add_subplot(gs[0, :])

            if len(predictions) > 0:
                days_ahead = range(1, len(predictions) + 1)

                # Current price line
                ax1.axhline(current_price, color='black', linestyle='-', linewidth=2,
                            label=f'Current Price: ${current_price:.2f}')

                # Prediction line
                ax1.plot(days_ahead, predictions, 'bo-', linewidth=2, markersize=4,
                         label='Predicted Prices', alpha=0.8)

                # Add stop-loss and take-profit if available
                levels = signals.get('stop_take_levels', {})
                if levels:
                    stop_loss = levels.get('stop_loss', 0)
                    take_profit = levels.get('take_profit', 0)

                    if stop_loss > 0:
                        ax1.axhline(stop_loss, color='red', linestyle='--', alpha=0.7,
                                    label=f'Stop Loss: ${stop_loss:.2f}')
                    if take_profit > 0:
                        ax1.axhline(take_profit, color='green', linestyle='--', alpha=0.7,
                                    label=f'Take Profit: ${take_profit:.2f}')
            else:
                ax1.text(0.5, 0.5, 'No prediction data available',
                         transform=ax1.transAxes, ha='center', va='center')

            ax1.set_title('Price Predictions with Trading Levels', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Days Ahead')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Trading Signals Summary
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.axis('off')

            signal_text = []
            signal_text.append("📈 TRADING SIGNALS")
            signal_text.append("-" * 25)

            # Primary signal
            primary = signals.get('primary_signal', {})
            if primary:
                signal_text.append(f"Primary: {primary.get('signal', 'N/A')}")
                signal_text.append(f"Target: ${primary.get('target_price', 0):.2f}")
                signal_text.append(f"Change: {primary.get('change_percent', 0):+.1f}%")

            # Position sizing
            position = signals.get('position_sizing', {})
            if position:
                signal_text.append("")
                signal_text.append("💰 POSITION SIZING")
                signal_text.append(f"Size: {position.get('position_percentage', 0):.1f}%")
                signal_text.append(f"Risk: {position.get('risk_per_trade', 0):.1f}%")

            ax2.text(0.05, 0.95, '\n'.join(signal_text), transform=ax2.transAxes,
                     verticalalignment='top', fontfamily='monospace', fontsize=11,
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

            # 3. Risk Metrics
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.axis('off')

            risk_text = []
            risk_text.append("⚠️ RISK MANAGEMENT")
            risk_text.append("-" * 25)

            if levels:
                risk_amount = levels.get('risk_amount', 0)
                reward_amount = levels.get('reward_amount', 0)
                rr_ratio = levels.get('risk_reward_ratio', 0)

                risk_text.append(f"Risk: ${risk_amount:.2f}")
                risk_text.append(f"Reward: ${reward_amount:.2f}")
                risk_text.append(f"R/R Ratio: {rr_ratio:.2f}")

            # Market regime
            regime = signals.get('market_regime', {})
            if regime:
                risk_text.append("")
                risk_text.append("📊 MARKET REGIME")
                risk_text.append(f"{regime.get('description', 'Normal conditions')}")

            ax3.text(0.05, 0.95, '\n'.join(risk_text), transform=ax3.transAxes,
                     verticalalignment='top', fontfamily='monospace', fontsize=11,
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

            plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error creating trading signals dashboard: {str(e)}")


def create_summary_report(prediction_results: Dict[str, Any],
                          risk_report: Dict[str, Any],
                          trading_signals: Dict[str, Any]) -> None:
    """
    Create a simplified summary report
    """
    try:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        fig.suptitle('🛢️ OIL PRICE ANALYSIS SUMMARY', fontsize=18, fontweight='bold', y=0.98)

        # Extract key data safely
        current_price = prediction_results.get('current_price', 0)
        predictions = prediction_results.get('predictions', [])

        # 1. Executive Summary
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')

        summary_lines = []
        summary_lines.append("📊 EXECUTIVE SUMMARY")
        summary_lines.append("=" * 50)
        summary_lines.append(f"Current Oil Price: ${current_price:.2f}")

        if isinstance(predictions, (list, np.ndarray)) and len(predictions) > 0:
            next_day = predictions[0]
            change_1d = ((next_day - current_price) / current_price) * 100 if current_price > 0 else 0
            summary_lines.append(f"Tomorrow's Prediction: ${next_day:.2f} ({change_1d:+.2f}%)")

            if len(predictions) >= 7:
                week_pred = predictions[6]
                change_1w = ((week_pred - current_price) / current_price) * 100 if current_price > 0 else 0
                summary_lines.append(f"1-Week Outlook: ${week_pred:.2f} ({change_1w:+.2f}%)")

        overall_risk = risk_report.get('overall_risk', 'UNKNOWN')
        summary_lines.append(f"Risk Level: {overall_risk}")

        primary_signal = trading_signals.get('primary_signal', {})
        if primary_signal:
            signal_text = primary_signal.get('signal', 'N/A')
            summary_lines.append(f"Trading Signal: {signal_text}")

        summary_text = '\n'.join(summary_lines)
        ax1.text(0.05, 0.85, summary_text, transform=ax1.transAxes,
                 fontfamily='monospace', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.9))

        # Add timestamp
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        ax1.text(0.95, 0.05, f'Generated: {timestamp}', transform=ax1.transAxes,
                 ha='right', va='bottom', fontsize=10, style='italic')

        # 2. Simple Price Chart
        ax2 = fig.add_subplot(gs[1, 0])

        if isinstance(predictions, (list, np.ndarray)) and len(predictions) > 0:
            days = range(1, min(len(predictions) + 1, 31))  # Show max 30 days
            pred_subset = predictions[:len(days)]

            ax2.plot(days, pred_subset, 'ro-', linewidth=2, markersize=4, alpha=0.8)
            ax2.axhline(current_price, color='black', linestyle='--', alpha=0.7,
                        label=f'Current: ${current_price:.2f}')

            ax2.set_title('Price Predictions', fontweight='bold', fontsize=12)
            ax2.set_xlabel('Days Ahead')
            ax2.set_ylabel('Price ($)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No prediction data available',
                     transform=ax2.transAxes, ha='center', va='center')

        # 3. Risk Summary
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')

        risk_text = []
        risk_text.append("⚠️ RISK SUMMARY")
        risk_text.append("-" * 20)

        risk_score = risk_report.get('risk_score', 0)
        risk_text.append(f"Risk Score: {risk_score:.0f}/100")

        vol_metrics = risk_report.get('volatility_metrics', {})
        if vol_metrics:
            annual_vol = vol_metrics.get('annual_volatility', 0) * 100
            risk_text.append(f"Volatility: {annual_vol:.1f}%")

        # Position sizing
        position = trading_signals.get('position_sizing', {})
        if position:
            pos_pct = position.get('position_percentage', 0)
            risk_text.append(f"Position Size: {pos_pct:.1f}%")

        # Overall recommendation
        overall_rec = trading_signals.get('overall_recommendation', {})
        if overall_rec:
            rec_desc = overall_rec.get('description', 'Standard approach')
            risk_text.append(f"Recommendation: {rec_desc}")

        ax3.text(0.05, 0.95, '\n'.join(risk_text), transform=ax3.transAxes,
                 verticalalignment='top', fontfamily='monospace', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error creating summary report: {str(e)}")
        # Create minimal fallback plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Summary Report Error: {str(e)}',
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        plt.title('Oil Price Analysis Summary', fontsize=16, fontweight='bold')
        plt.show()


# Utility functions for quick plotting
def quick_price_plot(dates: pd.DatetimeIndex, prices: np.ndarray,
                     predictions: np.ndarray, n_history: int = 100) -> None:
    """Quick price plot with predictions"""
    try:
        visualizer = PredictionVisualizer()
        visualizer.plot_price_history_with_predictions(
            dates, prices, predictions, n_history=n_history,
            title="Quick Oil Price Prediction"
        )
    except Exception as e:
        print(f"Error in quick_price_plot: {str(e)}")


def quick_risk_dashboard(risk_report: Dict[str, Any], prices: np.ndarray,
                         predictions: np.ndarray) -> None:
    """Quick risk analysis dashboard"""
    try:
        visualizer = PredictionVisualizer()
        visualizer.plot_risk_analysis_dashboard(risk_report, prices, predictions,
                                                title="Quick Risk Analysis")
    except Exception as e:
        print(f"Error in quick_risk_dashboard: {str(e)}")


def quick_signals_dashboard(signals: Dict[str, Any], current_price: float,
                            predictions: np.ndarray) -> None:
    """Quick trading signals dashboard"""
    try:
        visualizer = PredictionVisualizer()
        visualizer.plot_trading_signals_dashboard(signals, current_price, predictions,
                                                  title="Quick Trading Signals")
    except Exception as e:
        print(f"Error in quick_signals_dashboard: {str(e)}")


def plot_simple_comparison(results_dict: Dict[str, float],
                           title: str = "Model Comparison") -> None:
    """
    Simple comparison plot for different models or wavelets

    Args:
        results_dict: Dictionary with names as keys and values as numbers
        title: Plot title
    """
    try:
        if not results_dict:
            print("No data to compare")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        names = list(results_dict.keys())
        values = list(results_dict.values())

        # Create bar plot
        bars = ax.bar(names, values, alpha=0.8, color=plt.cm.viridis(np.linspace(0, 1, len(names))))

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        # Highlight best value (lowest for losses, highest for scores)
        if values:
            if all(v >= 0 for v in values if not np.isnan(v)):  # Positive values - higher is better
                best_idx = np.nanargmax(values)
            else:  # Mixed or negative values - lower is better
                best_idx = np.nanargmin([abs(v) for v in values])

            if best_idx < len(bars):
                bars[best_idx].set_color('gold')
                bars[best_idx].set_edgecolor('black')
                bars[best_idx].set_linewidth(2)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error in plot_simple_comparison: {str(e)}")


def validate_data_inputs(data: Any, data_name: str = "data") -> bool:
    """
    Validate data inputs for plotting functions

    Args:
        data: Data to validate
        data_name: Name of the data for error messages

    Returns:
        bool: True if data is valid, False otherwise
    """
    try:
        if data is None:
            print(f"Warning: {data_name} is None")
            return False

        if isinstance(data, (list, np.ndarray)):
            if len(data) == 0:
                print(f"Warning: {data_name} is empty")
                return False

            # Check for valid numerical data
            try:
                numeric_data = np.array(data, dtype=float)
                if np.all(np.isnan(numeric_data)):
                    print(f"Warning: {data_name} contains only NaN values")
                    return False
            except:
                print(f"Warning: {data_name} contains non-numeric values")
                return False

        elif isinstance(data, dict):
            if len(data) == 0:
                print(f"Warning: {data_name} dictionary is empty")
                return False

        return True

    except Exception as e:
        print(f"Error validating {data_name}: {str(e)}")
        return False


def safe_plot_wrapper(plot_function):
    """
    Decorator to safely wrap plotting functions with error handling
    """

    def wrapper(*args, **kwargs):
        try:
            return plot_function(*args, **kwargs)
        except Exception as e:
            print(f"Error in {plot_function.__name__}: {str(e)}")
            # Create a simple error plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Plotting Error:\n{str(e)[:100]}...',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
            plt.title(f'Error in {plot_function.__name__}', fontsize=14, fontweight='bold')
            plt.show()

    return wrapper


# Apply safe wrapper to main plotting functions
PredictionVisualizer.plot_price_history_with_predictions = safe_plot_wrapper(
    PredictionVisualizer.plot_price_history_with_predictions
)
PredictionVisualizer.plot_wavelet_decomposition = safe_plot_wrapper(
    PredictionVisualizer.plot_wavelet_decomposition
)
PredictionVisualizer.plot_risk_analysis_dashboard = safe_plot_wrapper(
    PredictionVisualizer.plot_risk_analysis_dashboard
)
PredictionVisualizer.plot_trading_signals_dashboard = safe_plot_wrapper(
    PredictionVisualizer.plot_trading_signals_dashboard
)

# Set default plotting parameters for better compatibility
plt.rcParams.update({
    'figure.max_open_warning': 0,  # Disable too many figures warning
    'axes.formatter.useoffset': False,  # Don't use offset notation
    'axes.formatter.limits': (-3, 3),  # Use scientific notation limits
})

print("✅ Visualization tools module loaded successfully with error handling")