#!/usr/bin/env python3
"""
Risk Analysis Module for Oil Price Prediction
Handles risk assessment, volatility analysis, and trading signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class RiskAnalyzer:
    """
    Comprehensive risk analysis for oil price predictions
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize risk analyzer
        
        Args:
            confidence_level: Confidence level for risk metrics (default: 0.95)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def calculate_volatility_metrics(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive volatility metrics
        
        Args:
            prices: Historical price series
        
        Returns:
            dict: Volatility metrics
        """
        returns = np.diff(prices) / prices[:-1]
        
        # Basic volatility metrics
        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)  # 252 trading days per year
        
        # Rolling volatilities
        returns_series = pd.Series(returns)
        vol_30d = returns_series.rolling(30).std().iloc[-1] * np.sqrt(252)
        vol_60d = returns_series.rolling(60).std().iloc[-1] * np.sqrt(252)
        vol_90d = returns_series.rolling(90).std().iloc[-1] * np.sqrt(252)
        
        # GARCH-like volatility (simplified exponential weighting)
        ewm_vol = returns_series.ewm(span=30).std().iloc[-1] * np.sqrt(252)
        
        # Volatility clustering metric (standard deviation of rolling volatility)
        rolling_vol = returns_series.rolling(30).std() * np.sqrt(252)
        vol_clustering = np.std(rolling_vol.dropna())
        
        # Volatility percentiles
        vol_percentile_75 = np.percentile(rolling_vol.dropna(), 75)
        vol_percentile_25 = np.percentile(rolling_vol.dropna(), 25)
        current_vol_percentile = stats.percentileofscore(rolling_vol.dropna(), vol_30d)
        
        return {
            'daily_volatility': daily_vol,
            'annual_volatility': annual_vol,
            'volatility_30d': vol_30d,
            'volatility_60d': vol_60d,
            'volatility_90d': vol_90d,
            'ewm_volatility': ewm_vol,
            'volatility_clustering': vol_clustering,
            'volatility_percentile_75': vol_percentile_75,
            'volatility_percentile_25': vol_percentile_25,
            'current_volatility_percentile': current_vol_percentile
        }
    
    def calculate_var_cvar(self, prices: np.ndarray, holding_period: int = 1) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR)
        
        Args:
            prices: Historical price series
            holding_period: Holding period in days
        
        Returns:
            dict: VaR and CVaR metrics
        """
        returns = np.diff(prices) / prices[:-1]
        current_price = prices[-1]
        
        # Scale returns for holding period
        scaled_returns = returns * np.sqrt(holding_period)
        
        # Historical VaR (percentile method)
        var_historical = np.percentile(scaled_returns, self.alpha * 100)
        var_dollar_historical = current_price * abs(var_historical)
        
        # Parametric VaR (assuming normal distribution)
        mean_return = np.mean(scaled_returns)
        std_return = np.std(scaled_returns)
        z_score = stats.norm.ppf(self.alpha)
        var_parametric = mean_return + z_score * std_return
        var_dollar_parametric = current_price * abs(var_parametric)
        
        # Conditional VaR (Expected Shortfall)
        below_var = scaled_returns[scaled_returns <= var_historical]
        cvar = np.mean(below_var) if len(below_var) > 0 else var_historical
        cvar_dollar = current_price * abs(cvar)
        
        # Modified VaR (Cornish-Fisher expansion for non-normal distributions)
        skewness = stats.skew(scaled_returns)
        kurtosis = stats.kurtosis(scaled_returns, fisher=True)  # Excess kurtosis
        
        # Cornish-Fisher correction
        cf_correction = (z_score + 
                        (z_score**2 - 1) * skewness / 6 +
                        (z_score**3 - 3*z_score) * kurtosis / 24 -
                        (2*z_score**3 - 5*z_score) * (skewness**2) / 36)
        
        var_modified = mean_return + cf_correction * std_return
        var_dollar_modified = current_price * abs(var_modified)
        
        return {
            'var_historical_pct': var_historical,
            'var_historical_dollar': var_dollar_historical,
            'var_parametric_pct': var_parametric,
            'var_parametric_dollar': var_dollar_parametric,
            'var_modified_pct': var_modified,
            'var_modified_dollar': var_dollar_modified,
            'cvar_pct': cvar,
            'cvar_dollar': cvar_dollar,
            'holding_period': holding_period
        }
    
    def analyze_prediction_risk(self, predictions: np.ndarray, current_price: float) -> Dict[str, Any]:
        """
        Analyze risk characteristics of predictions
        
        Args:
            predictions: Array of predicted prices
            current_price: Current market price
        
        Returns:
            dict: Prediction risk analysis
        """
        # Calculate prediction statistics
        pred_returns = (predictions - current_price) / current_price
        pred_changes = predictions - current_price
        
        # Basic statistics
        mean_prediction = np.mean(predictions)
        median_prediction = np.median(predictions)
        std_prediction = np.std(predictions)
        
        # Return statistics
        mean_return = np.mean(pred_returns)
        volatility_prediction = np.std(pred_returns)
        
        # Risk metrics
        max_gain = np.max(pred_changes)
        max_loss = np.min(pred_changes)
        max_gain_pct = np.max(pred_returns) * 100
        max_loss_pct = np.min(pred_returns) * 100
        
        # Probability of positive returns
        prob_positive = np.mean(pred_returns > 0)
        prob_negative = np.mean(pred_returns < 0)
        
        # Prediction intervals
        lower_bound = np.percentile(predictions, (self.alpha/2) * 100)
        upper_bound = np.percentile(predictions, (1 - self.alpha/2) * 100)
        prediction_range = upper_bound - lower_bound
        
        # Skewness and kurtosis of predictions
        pred_skewness = stats.skew(pred_returns)
        pred_kurtosis = stats.kurtosis(pred_returns, fisher=True)
        
        # Risk classification
        if volatility_prediction > 0.15:  # 15%
            risk_level = "HIGH"
            risk_color = "🔴"
        elif volatility_prediction > 0.08:  # 8%
            risk_level = "MEDIUM"
            risk_color = "🟡"
        else:
            risk_level = "LOW"
            risk_color = "🟢"
        
        return {
            'mean_prediction': mean_prediction,
            'median_prediction': median_prediction,
            'std_prediction': std_prediction,
            'volatility_prediction': volatility_prediction,
            'mean_return': mean_return,
            'max_gain': max_gain,
            'max_loss': max_loss,
            'max_gain_pct': max_gain_pct,
            'max_loss_pct': max_loss_pct,
            'prob_positive': prob_positive,
            'prob_negative': prob_negative,
            'prediction_range': prediction_range,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'skewness': pred_skewness,
            'kurtosis': pred_kurtosis,
            'risk_level': risk_level,
            'risk_color': risk_color
        }
    
    def calculate_drawdown_metrics(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Calculate drawdown metrics
        
        Args:
            prices: Historical price series
        
        Returns:
            dict: Drawdown metrics
        """
        # Calculate running maximum
        running_max = np.maximum.accumulate(prices)
        
        # Calculate drawdowns
        drawdowns = (prices - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = np.min(drawdowns)
        
        # Average drawdown
        negative_drawdowns = drawdowns[drawdowns < 0]
        avg_drawdown = np.mean(negative_drawdowns) if len(negative_drawdowns) > 0 else 0
        
        # Current drawdown
        current_drawdown = drawdowns[-1]
        
        # Drawdown duration (simplified - consecutive negative periods)
        drawdown_periods = []
        current_period = 0
        
        for dd in drawdowns:
            if dd < 0:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'current_drawdown': current_drawdown,
            'avg_drawdown_duration': avg_drawdown_duration,
            'max_drawdown_duration': max_drawdown_duration,
            'drawdown_periods_count': len(drawdown_periods)
        }
    
    def generate_confidence_intervals(self, predictions: np.ndarray, 
                                    confidence_levels: List[float] = [0.68, 0.95, 0.99]) -> Dict[str, Dict[str, float]]:
        """
        Generate confidence intervals for predictions
        
        Args:
            predictions: Array of predictions
            confidence_levels: List of confidence levels
        
        Returns:
            dict: Confidence intervals for each level
        """
        intervals = {}
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(predictions, lower_percentile)
            upper_bound = np.percentile(predictions, upper_percentile)
            
            intervals[f'{conf_level:.0%}'] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'width': upper_bound - lower_bound
            }
        
        return intervals
    
    def assess_model_stability(self, component_predictions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Assess the stability and agreement between component models
        
        Args:
            component_predictions: Predictions from different components
        
        Returns:
            dict: Model stability metrics
        """
        if len(component_predictions) < 2:
            return {'error': 'Need at least 2 components for stability analysis'}
        
        # Convert to array for easier calculation
        pred_matrix = np.array(list(component_predictions.values()))
        
        # Model agreement metrics
        pred_correlations = np.corrcoef(pred_matrix)
        avg_correlation = np.mean(pred_correlations[np.triu_indices_from(pred_correlations, k=1)])
        min_correlation = np.min(pred_correlations[np.triu_indices_from(pred_correlations, k=1)])
        
        # Prediction dispersion
        pred_means = np.mean(pred_matrix, axis=1)  # Mean prediction for each component
        pred_stds = np.std(pred_matrix, axis=1)    # Std prediction for each component
        
        # Cross-component statistics
        component_agreement = 1 - (np.std(pred_means) / np.mean(np.abs(pred_means)))
        
        # Stability classification
        if avg_correlation > 0.8 and component_agreement > 0.9:
            stability = "HIGH"
            stability_color = "🟢"
        elif avg_correlation > 0.6 and component_agreement > 0.8:
            stability = "MEDIUM"
            stability_color = "🟡"
        else:
            stability = "LOW"
            stability_color = "🔴"
        
        return {
            'avg_correlation': avg_correlation,
            'min_correlation': min_correlation,
            'component_agreement': component_agreement,
            'prediction_dispersion': np.std(pred_means),
            'stability_level': stability,
            'stability_color': stability_color,
            'component_means': dict(zip(component_predictions.keys(), pred_means)),
            'component_stds': dict(zip(component_predictions.keys(), pred_stds))
        }
    
    def generate_risk_report(self, prices: np.ndarray, predictions: np.ndarray,
                           component_predictions: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive risk report
        
        Args:
            prices: Historical price series
            predictions: Future price predictions
            component_predictions: Optional component predictions
        
        Returns:
            dict: Comprehensive risk report
        """
        current_price = prices[-1]
        
        # Calculate all risk metrics
        vol_metrics = self.calculate_volatility_metrics(prices)
        var_metrics = self.calculate_var_cvar(prices, holding_period=1)
        var_metrics_5d = self.calculate_var_cvar(prices, holding_period=5)
        pred_risk = self.analyze_prediction_risk(predictions, current_price)
        drawdown_metrics = self.calculate_drawdown_metrics(prices)
        confidence_intervals = self.generate_confidence_intervals(predictions)
        
        # Model stability (if component predictions available)
        stability_metrics = {}
        if component_predictions:
            stability_metrics = self.assess_model_stability(component_predictions)
        
        # Overall risk assessment
        risk_factors = []
        
        # Check volatility
        if vol_metrics['current_volatility_percentile'] > 75:
            risk_factors.append("High current volatility")
        
        # Check VaR
        if var_metrics['var_historical_dollar'] > current_price * 0.05:  # 5% of current price
            risk_factors.append("High Value at Risk")
        
        # Check prediction uncertainty
        if pred_risk['volatility_prediction'] > 0.1:  # 10%
            risk_factors.append("High prediction uncertainty")
        
        # Check drawdown
        if drawdown_metrics['current_drawdown'] < -0.1:  # 10% drawdown
            risk_factors.append("Currently in significant drawdown")
        
        # Overall risk score (0-100)
        risk_score = (
            min(vol_metrics['current_volatility_percentile'], 100) * 0.3 +
            min(abs(drawdown_metrics['current_drawdown']) * 1000, 100) * 0.2 +
            min(pred_risk['volatility_prediction'] * 500, 100) * 0.3 +
            min(var_metrics['var_historical_dollar'] / current_price * 2000, 100) * 0.2
        )
        
        if risk_score > 70:
            overall_risk = "HIGH"
            risk_color = "🔴"
        elif risk_score > 40:
            overall_risk = "MEDIUM"
            risk_color = "🟡"
        else:
            overall_risk = "LOW"
            risk_color = "🟢"
        
        return {
            'overall_risk': overall_risk,
            'risk_color': risk_color,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'volatility_metrics': vol_metrics,
            'var_metrics': {
                '1_day': var_metrics,
                '5_day': var_metrics_5d
            },
            'prediction_risk': pred_risk,
            'drawdown_metrics': drawdown_metrics,
            'confidence_intervals': confidence_intervals,
            'model_stability': stability_metrics,
            'current_price': current_price,
            'analysis_date': pd.Timestamp.now().isoformat()
        }

class TradingSignalGenerator:
    """
    Generate trading signals based on predictions and risk analysis
    """
    
    def __init__(self, risk_tolerance: str = 'medium'):
        """
        Initialize trading signal generator
        
        Args:
            risk_tolerance: Risk tolerance level ('low', 'medium', 'high')
        """
        self.risk_tolerance = risk_tolerance
        self.risk_multipliers = {
            'low': 0.5,
            'medium': 1.0,
            'high': 1.5
        }
    
    def generate_directional_signals(self, current_price: float, predictions: np.ndarray,
                                   time_horizons: List[int] = [1, 7, 30]) -> Dict[str, Dict[str, Any]]:
        """
        Generate directional trading signals
        
        Args:
            current_price: Current market price
            predictions: Array of future price predictions
            time_horizons: Time horizons to analyze (in days)
        
        Returns:
            dict: Directional signals for each time horizon
        """
        signals = {}
        
        for horizon in time_horizons:
            if horizon <= len(predictions):
                future_price = predictions[horizon - 1]  # 0-indexed
                change_pct = ((future_price - current_price) / current_price) * 100
                
                # Signal strength based on magnitude of change
                abs_change = abs(change_pct)
                
                if abs_change > 5:
                    strength = "STRONG"
                elif abs_change > 2:
                    strength = "MODERATE"
                elif abs_change > 0.5:
                    strength = "WEAK"
                else:
                    strength = "NEUTRAL"
                
                # Direction
                if change_pct > 2:
                    direction = "BUY"
                    emoji = "📈"
                elif change_pct < -2:
                    direction = "SELL"
                    emoji = "📉"
                else:
                    direction = "HOLD"
                    emoji = "➡️"
                
                # Confidence based on consistency
                if horizon <= len(predictions) - 5:
                    # Check if trend is consistent over next 5 days
                    next_5_days = predictions[horizon-1:horizon+4]
                    trend_consistency = np.mean(np.diff(next_5_days) > 0) if len(next_5_days) > 1 else 0.5
                    
                    if trend_consistency > 0.8 or trend_consistency < 0.2:
                        confidence = "HIGH"
                    elif trend_consistency > 0.6 or trend_consistency < 0.4:
                        confidence = "MEDIUM"
                    else:
                        confidence = "LOW"
                else:
                    confidence = "MEDIUM"
                
                signals[f'{horizon}_day'] = {
                    'direction': direction,
                    'strength': strength,
                    'confidence': confidence,
                    'change_percent': change_pct,
                    'target_price': future_price,
                    'signal': f"{emoji} {direction}",
                    'description': f"{strength} {direction} signal with {confidence} confidence"
                }
        
        return signals
    
    def calculate_position_sizing(self, risk_metrics: Dict[str, Any], 
                                 portfolio_value: float = 100000) -> Dict[str, Any]:
        """
        Calculate recommended position sizes based on risk
        
        Args:
            risk_metrics: Risk analysis results
            portfolio_value: Total portfolio value
        
        Returns:
            dict: Position sizing recommendations
        """
        base_risk_per_trade = {
            'low': 0.01,      # 1% risk per trade
            'medium': 0.02,   # 2% risk per trade
            'high': 0.03      # 3% risk per trade
        }
        
        risk_per_trade = base_risk_per_trade[self.risk_tolerance]
        
        # Adjust based on current market volatility
        current_vol = risk_metrics.get('volatility_metrics', {}).get('volatility_30d', 0.2)
        vol_adjustment = max(0.5, min(1.5, 0.2 / current_vol))  # Inverse relationship
        
        # Adjust based on prediction risk
        pred_risk = risk_metrics.get('prediction_risk', {}).get('volatility_prediction', 0.1)
        pred_adjustment = max(0.5, min(1.5, 0.05 / pred_risk))
        
        # Final position size
        adjusted_risk = risk_per_trade * vol_adjustment * pred_adjustment
        position_value = portfolio_value * adjusted_risk
        
        # VaR-based position sizing
        var_dollar = risk_metrics.get('var_metrics', {}).get('1_day', {}).get('var_historical_dollar', 0)
        current_price = risk_metrics.get('current_price', 100)
        
        if var_dollar > 0:
            var_position_size = (portfolio_value * adjusted_risk) / var_dollar
        else:
            var_position_size = position_value / current_price
        
        # Kelly criterion approximation
        pred_data = risk_metrics.get('prediction_risk', {})
        win_prob = pred_data.get('prob_positive', 0.5)
        avg_win = pred_data.get('max_gain_pct', 5) / 100
        avg_loss = abs(pred_data.get('max_loss_pct', -5)) / 100
        
        if avg_loss > 0:
            kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            kelly_position_value = portfolio_value * kelly_fraction
        else:
            kelly_position_value = position_value
        
        # Conservative approach - use minimum of all methods
        recommended_value = min(position_value, kelly_position_value, portfolio_value * 0.1)
        recommended_shares = recommended_value / current_price
        
        return {
            'recommended_position_value': recommended_value,
            'recommended_shares': recommended_shares,
            'position_percentage': (recommended_value / portfolio_value) * 100,
            'risk_per_trade': adjusted_risk * 100,
            'kelly_fraction': kelly_fraction * 100 if 'kelly_fraction' in locals() else 0,
            'var_based_size': var_position_size,
            'volatility_adjustment': vol_adjustment,
            'prediction_adjustment': pred_adjustment
        }
    
    def generate_stop_loss_take_profit(self, current_price: float, direction: str,
                                     risk_metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Generate stop-loss and take-profit levels
        
        Args:
            current_price: Current market price
            direction: Trading direction ('BUY' or 'SELL')
            risk_metrics: Risk analysis results
        
        Returns:
            dict: Stop-loss and take-profit levels
        """
        # Get volatility for dynamic levels
        vol_metrics = risk_metrics.get('volatility_metrics', {})
        daily_vol = vol_metrics.get('daily_volatility', 0.02)
        
        # Base levels adjusted by risk tolerance
        base_stop_pct = 0.02 * self.risk_multipliers[self.risk_tolerance]  # 2% base
        base_target_pct = 0.04 * self.risk_multipliers[self.risk_tolerance]  # 4% base
        
        # Adjust by volatility
        vol_adjusted_stop = base_stop_pct + (daily_vol * 2)
        vol_adjusted_target = base_target_pct + (daily_vol * 3)
        
        if direction == 'BUY':
            stop_loss = current_price * (1 - vol_adjusted_stop)
            take_profit = current_price * (1 + vol_adjusted_target)
            
            # Trailing stop (percentage)
            trailing_stop_pct = vol_adjusted_stop * 0.8
            
        elif direction == 'SELL':
            stop_loss = current_price * (1 + vol_adjusted_stop)
            take_profit = current_price * (1 - vol_adjusted_target)
            
            # Trailing stop (percentage)
            trailing_stop_pct = vol_adjusted_stop * 0.8
        else:
            return {}
        
        # Risk-reward ratio
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_stop_percent': trailing_stop_pct * 100,
            'risk_amount': risk,
            'reward_amount': reward,
            'risk_reward_ratio': risk_reward_ratio
        }
    
    def generate_comprehensive_signals(self, current_price: float, predictions: np.ndarray,
                                     risk_metrics: Dict[str, Any],
                                     portfolio_value: float = 100000) -> Dict[str, Any]:
        """
        Generate comprehensive trading signals
        
        Args:
            current_price: Current market price
            predictions: Future price predictions
            risk_metrics: Risk analysis results
            portfolio_value: Portfolio value for position sizing
        
        Returns:
            dict: Comprehensive trading signals
        """
        # Generate directional signals
        directional_signals = self.generate_directional_signals(current_price, predictions)
        
        # Get primary signal (1-day)
        primary_signal = directional_signals.get('1_day', {})
        primary_direction = primary_signal.get('direction', 'HOLD')
        
        # Position sizing
        position_sizing = self.calculate_position_sizing(risk_metrics, portfolio_value)
        
        # Stop-loss and take-profit
        stop_take_levels = {}
        if primary_direction in ['BUY', 'SELL']:
            stop_take_levels = self.generate_stop_loss_take_profit(
                current_price, primary_direction, risk_metrics
            )
        
        # Market regime analysis
        vol_percentile = risk_metrics.get('volatility_metrics', {}).get('current_volatility_percentile', 50)
        
        if vol_percentile > 80:
            market_regime = "High Volatility - Exercise Caution"
            regime_emoji = "⚠️"
        elif vol_percentile < 20:
            market_regime = "Low Volatility - Stable Conditions"
            regime_emoji = "😌"
        else:
            market_regime = "Normal Volatility - Standard Trading"
            regime_emoji = "📊"
        
        # Overall recommendation
        overall_risk = risk_metrics.get('overall_risk', 'MEDIUM')
        prediction_confidence = primary_signal.get('confidence', 'MEDIUM')
        
        if overall_risk == 'HIGH' or prediction_confidence == 'LOW':
            recommendation = "REDUCE POSITION SIZE - High uncertainty"
            rec_emoji = "⚠️"
        elif overall_risk == 'LOW' and prediction_confidence == 'HIGH':
            recommendation = "FAVORABLE CONDITIONS - Consider full position"
            rec_emoji = "✅"
        else:
            recommendation = "STANDARD POSITION - Normal conditions"
            rec_emoji = "📈"
        
        return {
            'primary_signal': primary_signal,
            'directional_signals': directional_signals,
            'position_sizing': position_sizing,
            'stop_take_levels': stop_take_levels,
            'market_regime': {
                'description': market_regime,
                'emoji': regime_emoji,
                'volatility_percentile': vol_percentile
            },
            'overall_recommendation': {
                'description': recommendation,
                'emoji': rec_emoji,
                'risk_level': overall_risk,
                'confidence': prediction_confidence
            },
            'risk_factors': risk_metrics.get('risk_factors', []),
            'timestamp': pd.Timestamp.now().isoformat()
        }

def print_risk_report(risk_report: Dict[str, Any]):
    """
    Print formatted risk report
    
    Args:
        risk_report: Risk analysis report
    """
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE RISK ANALYSIS REPORT")
    print("="*60)
    
    # Overall assessment
    overall_risk = risk_report.get('overall_risk', 'UNKNOWN')
    risk_color = risk_report.get('risk_color', '❓')
    risk_score = risk_report.get('risk_score', 0)
    
    print(f"\n{risk_color} OVERALL RISK LEVEL: {overall_risk}")
    print(f"📊 Risk Score: {risk_score:.1f}/100")
    
    # Risk factors
    risk_factors = risk_report.get('risk_factors', [])
    if risk_factors:
        print(f"\n⚠️ KEY RISK FACTORS:")
        for factor in risk_factors:
            print(f"   • {factor}")
    
    # Volatility metrics
    vol_metrics = risk_report.get('volatility_metrics', {})
    if vol_metrics:
        print(f"\n📈 VOLATILITY ANALYSIS:")
        print(f"   Annual Volatility: {vol_metrics.get('annual_volatility', 0):.1%}")
        print(f"   30-Day Volatility: {vol_metrics.get('volatility_30d', 0):.1%}")
        print(f"   Volatility Percentile: {vol_metrics.get('current_volatility_percentile', 0):.0f}%")
    
    # VaR metrics
    var_metrics = risk_report.get('var_metrics', {}).get('1_day', {})
    if var_metrics:
        print(f"\n💰 VALUE AT RISK (1-day, 95% confidence):")
        print(f"   Historical VaR: ${var_metrics.get('var_historical_dollar', 0):.2f}")
        print(f"   Parametric VaR: ${var_metrics.get('var_parametric_dollar', 0):.2f}")
        print(f"   Expected Shortfall: ${var_metrics.get('cvar_dollar', 0):.2f}")
    
    # Prediction risk
    pred_risk = risk_report.get('prediction_risk', {})
    if pred_risk:
        print(f"\n🔮 PREDICTION RISK ANALYSIS:")
        print(f"   Prediction Volatility: {pred_risk.get('volatility_prediction', 0):.1%}")
        print(f"   Max Potential Gain: {pred_risk.get('max_gain_pct', 0):+.1f}%")
        print(f"   Max Potential Loss: {pred_risk.get('max_loss_pct', 0):+.1f}%")
        print(f"   Probability of Gain: {pred_risk.get('prob_positive', 0):.1%}")
    
    # Model stability
    stability = risk_report.get('model_stability', {})
    if stability and 'stability_level' in stability:
        stability_color = stability.get('stability_color', '❓')
        print(f"\n🎯 MODEL STABILITY: {stability_color} {stability.get('stability_level', 'UNKNOWN')}")
        print(f"   Average Correlation: {stability.get('avg_correlation', 0):.3f}")
        print(f"   Component Agreement: {stability.get('component_agreement', 0):.1%}")

def print_trading_signals(signals: Dict[str, Any]):
    """
    Print formatted trading signals
    
    Args:
        signals: Trading signals report
    """
    print("\n" + "="*60)
    print("📈 COMPREHENSIVE TRADING SIGNALS")
    print("="*60)
    
    # Primary signal
    primary = signals.get('primary_signal', {})
    if primary:
        print(f"\n🎯 PRIMARY SIGNAL (Next Day):")
        print(f"   {primary.get('signal', 'N/A')} - {primary.get('description', 'N/A')}")
        print(f"   Target Price: ${primary.get('target_price', 0):.2f}")
        print(f"   Expected Change: {primary.get('change_percent', 0):+.2f}%")
    
    # All directional signals
    directional = signals.get('directional_signals', {})
    if directional:
        print(f"\n📅 TIME HORIZON SIGNALS:")
        for period, signal in directional.items():
            print(f"   {period:>7}: {signal.get('signal', 'N/A'):>12} | "
                  f"Target: ${signal.get('target_price', 0):6.2f} | "
                  f"Change: {signal.get('change_percent', 0):+5.1f}%")
    
    # Position sizing
    position = signals.get('position_sizing', {})
    if position:
        print(f"\n💰 POSITION SIZING RECOMMENDATION:")
        print(f"   Recommended Position: ${position.get('recommended_position_value', 0):,.0f}")
        print(f"   Portfolio Percentage: {position.get('position_percentage', 0):.1f}%")
        print(f"   Risk Per Trade: {position.get('risk_per_trade', 0):.1f}%")
    
    # Stop-loss and take-profit
    levels = signals.get('stop_take_levels', {})
    if levels:
        print(f"\n🎯 STOP-LOSS & TAKE-PROFIT:")
        print(f"   Stop Loss: ${levels.get('stop_loss', 0):.2f}")
        print(f"   Take Profit: ${levels.get('take_profit', 0):.2f}")
        print(f"   Risk/Reward Ratio: {levels.get('risk_reward_ratio', 0):.2f}")
    
    # Market regime
    regime = signals.get('market_regime', {})
    if regime:
        print(f"\n📊 MARKET REGIME:")
        print(f"   {regime.get('emoji', '')} {regime.get('description', 'N/A')}")
    
    # Overall recommendation
    recommendation = signals.get('overall_recommendation', {})
    if recommendation:
        print(f"\n✅ OVERALL RECOMMENDATION:")
        print(f"   {recommendation.get('emoji', '')} {recommendation.get('description', 'N/A')}")
        print(f"   Risk Level: {recommendation.get('risk_level', 'N/A')}")
        print(f"   Confidence: {recommendation.get('confidence', 'N/A')}")
