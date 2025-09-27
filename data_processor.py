#!/usr/bin/env python3
"""
Data Processing Module for Oil Price Prediction
Handles data fetching, preprocessing, and sequence generation
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Comprehensive data processing class for oil price prediction
    """
    
    def __init__(self, sequence_length=60, prediction_horizon=1):
        """
        Initialize the data processor
        
        Args:
            sequence_length: Length of input sequences for LSTM
            prediction_horizon: Number of days to predict ahead
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.raw_data = None
        self.oil_prices = None
        self.dates = None
    
    def fetch_oil_data(self, symbol="CL=F", start_date="2015-01-01", end_date=None):
        """
        Fetch oil price data from Yahoo Finance
        
        Args:
            symbol: Yahoo Finance symbol (CL=F for WTI, BZ=F for Brent)
            start_date: Start date for data fetching
            end_date: End date for data fetching (default: today)
        
        Returns:
            DataFrame: Oil price data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"Fetching {symbol} data from {start_date} to {end_date}...")
        
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                raise Exception("No data returned from Yahoo Finance")
            
            self.raw_data = data
            self.oil_prices = data['Close'].values
            self.dates = data.index
            
            print(f"Successfully fetched {len(self.oil_prices)} data points")
            print(f"Price range: ${self.oil_prices.min():.2f} - ${self.oil_prices.max():.2f}")
            
            return data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            print("Generating synthetic data for demonstration...")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self, n_points=2500):
        """
        Generate synthetic oil price data for testing/demonstration
        
        Args:
            n_points: Number of data points to generate
        
        Returns:
            DataFrame: Synthetic oil price data
        """
        np.random.seed(42)
        dates = pd.date_range(start='2015-01-01', periods=n_points, freq='D')
        
        # Create realistic oil price pattern
        trend = np.linspace(45, 75, n_points)  # Long-term trend
        seasonal = 8 * np.sin(2 * np.pi * np.arange(n_points) / 365.25)  # Seasonal
        volatility = np.random.normal(0, 3, n_points)  # Daily volatility
        shocks = np.random.normal(0, 12, n_points) * (np.random.random(n_points) < 0.03)
        
        oil_prices = trend + seasonal + volatility + shocks
        oil_prices = np.maximum(oil_prices, 25)  # Price floor
        
        # Create OHLCV data
        synthetic_data = pd.DataFrame({
            'Open': oil_prices * (1 + np.random.normal(0, 0.005, n_points)),
            'High': oil_prices * (1 + np.abs(np.random.normal(0, 0.015, n_points))),
            'Low': oil_prices * (1 - np.abs(np.random.normal(0, 0.015, n_points))),
            'Close': oil_prices,
            'Volume': np.random.randint(500000, 2000000, n_points)
        }, index=dates)
        
        self.raw_data = synthetic_data
        self.oil_prices = oil_prices
        self.dates = dates
        
        print(f"Generated {n_points} synthetic data points")
        
        return synthetic_data
    
    def create_sequences(self, data, target_col=0):
        """
        Create sequences for LSTM training
        
        Args:
            data: Input data array or DataFrame
            target_col: Target column index for multivariate data
        
        Returns:
            tuple: (X, y) - Feature sequences and target values
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(scaled_data) - self.prediction_horizon + 1):
            X.append(scaled_data[i - self.sequence_length:i])
            if scaled_data.shape[1] == 1:
                y.append(scaled_data[i:i + self.prediction_horizon, 0])
            else:
                y.append(scaled_data[i:i + self.prediction_horizon, target_col])
        
        return np.array(X), np.array(y)
    
    def split_data(self, X, y, train_ratio=0.8, validation_ratio=0.1):
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Feature sequences
            y: Target values
            train_ratio: Proportion for training
            validation_ratio: Proportion for validation
        
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        n_samples = len(X)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * validation_ratio)
        
        # Sequential split (important for time series)
        X_train = X[:train_size]
        X_val = X[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        
        y_train = y[:train_size]
        y_val = y[train_size:train_size + val_size]
        y_test = y[train_size + val_size:]
        
        print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def inverse_transform(self, scaled_data):
        """
        Inverse transform scaled data back to original scale
        
        Args:
            scaled_data: Scaled data to transform back
        
        Returns:
            np.array: Data in original scale
        """
        if scaled_data.ndim == 1:
            scaled_data = scaled_data.reshape(-1, 1)
        
        return self.scaler.inverse_transform(scaled_data)
    
    def get_data_statistics(self):
        """
        Get comprehensive statistics about the oil price data
        
        Returns:
            dict: Statistical information
        """
        if self.oil_prices is None:
            return None
        
        returns = np.diff(self.oil_prices) / self.oil_prices[:-1] * 100
        
        stats = {
            'data_points': len(self.oil_prices),
            'date_range': {
                'start': self.dates[0].strftime('%Y-%m-%d'),
                'end': self.dates[-1].strftime('%Y-%m-%d')
            },
            'price_stats': {
                'current': self.oil_prices[-1],
                'mean': np.mean(self.oil_prices),
                'std': np.std(self.oil_prices),
                'min': np.min(self.oil_prices),
                'max': np.max(self.oil_prices),
                'median': np.median(self.oil_prices)
            },
            'returns_stats': {
                'mean_daily_return': np.mean(returns),
                'daily_volatility': np.std(returns),
                'annual_volatility': np.std(returns) * np.sqrt(252),
                'skewness': self._calculate_skewness(returns),
                'kurtosis': self._calculate_kurtosis(returns)
            },
            'risk_metrics': {
                'var_95_1day': np.percentile(returns, 5),
                'var_95_5day': np.percentile(returns, 5) * np.sqrt(5),
                'max_drawdown': self._calculate_max_drawdown(self.oil_prices)
            }
        }
        
        return stats
    
    def _calculate_skewness(self, data):
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / peak
        return np.min(drawdown)
    
    def detect_anomalies(self, threshold=3):
        """
        Detect price anomalies using statistical methods
        
        Args:
            threshold: Z-score threshold for anomaly detection
        
        Returns:
            dict: Anomaly information
        """
        if self.oil_prices is None:
            return None
        
        returns = np.diff(self.oil_prices) / self.oil_prices[:-1]
        z_scores = np.abs((returns - np.mean(returns)) / np.std(returns))
        
        anomaly_indices = np.where(z_scores > threshold)[0]
        
        anomalies = {
            'total_anomalies': len(anomaly_indices),
            'anomaly_dates': [self.dates[i + 1].strftime('%Y-%m-%d') for i in anomaly_indices],
            'anomaly_returns': [returns[i] * 100 for i in anomaly_indices],
            'anomaly_prices': [self.oil_prices[i + 1] for i in anomaly_indices]
        }
        
        return anomalies
    
    def prepare_prediction_input(self, n_steps_back=None):
        """
        Prepare the most recent data for making predictions
        
        Args:
            n_steps_back: Number of steps to look back (default: sequence_length)
        
        Returns:
            np.array: Prepared input for prediction
        """
        if n_steps_back is None:
            n_steps_back = self.sequence_length
        
        if self.oil_prices is None:
            raise ValueError("No data available. Please fetch data first.")
        
        # Get the most recent data
        recent_data = self.oil_prices[-n_steps_back:].reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(self.oil_prices.reshape(-1, 1))
        recent_scaled = scaled_data[-n_steps_back:]
        
        # Reshape for prediction (1, sequence_length, 1)
        prediction_input = recent_scaled.reshape(1, n_steps_back, 1)
        
        return prediction_input

# Utility functions for data analysis
def calculate_technical_indicators(prices, window=20):
    """
    Calculate common technical indicators
    
    Args:
        prices: Price series
        window: Window size for calculations
    
    Returns:
        dict: Technical indicators
    """
    df = pd.DataFrame({'price': prices})
    
    indicators = {
        'sma': df['price'].rolling(window=window).mean().iloc[-1],
        'ema': df['price'].ewm(span=window).mean().iloc[-1],
        'rsi': calculate_rsi(prices, window),
        'bollinger_bands': calculate_bollinger_bands(prices, window),
        'macd': calculate_macd(prices)
    }
    
    return indicators

def calculate_rsi(prices, window=14):
    """Calculate RSI indicator"""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = pd.Series(prices).rolling(window=window).mean()
    std = pd.Series(prices).rolling(window=window).std()
    
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    return {
        'upper': upper_band.iloc[-1],
        'middle': sma.iloc[-1],
        'lower': lower_band.iloc[-1],
        'position': (prices[-1] - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
    }

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    exp1 = pd.Series(prices).ewm(span=fast).mean()
    exp2 = pd.Series(prices).ewm(span=slow).mean()
    
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line.iloc[-1],
        'signal': signal_line.iloc[-1],
        'histogram': histogram.iloc[-1]
    }