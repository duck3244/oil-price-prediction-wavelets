#!/usr/bin/env python3
"""
Main Application for Oil Price Prediction using Wavelets and TensorFlow 2.0
Complete oil price prediction system with wavelet decomposition and deep learning
"""

import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import custom modules
from data_processor import DataProcessor
from wavelet_analyzer import WaveletAnalyzer
from model_builder import ModelBuilder
from predictor_engine import PredictionEngine
from risk_analyzer import RiskAnalyzer, TradingSignalGenerator, print_risk_report, print_trading_signals
from visualization_tools import PredictionVisualizer, create_summary_report

class OilPricePredictionApp:
    """
    Complete oil price prediction application
    """
    
    def __init__(self):
        """Initialize the application"""
        self.engine = None
        self.results = None
        self.risk_analyzer = RiskAnalyzer()
        self.signal_generator = TradingSignalGenerator()
        self.visualizer = PredictionVisualizer()
        
        print("🛢️ Oil Price Prediction System Initialized")
        print("=" * 60)
    
    def run_basic_prediction(self, symbol: str = "CL=F", days_ahead: int = 30,
                           plot_results: bool = True) -> Dict[str, Any]:
        """
        Run basic prediction with default settings
        
        Args:
            symbol: Oil contract symbol
            days_ahead: Number of days to predict
            plot_results: Whether to show plots
        
        Returns:
            dict: Prediction results
        """
        print(f"🚀 Running basic prediction for {symbol}...")
        
        # Initialize engine with default settings
        self.engine = PredictionEngine(
            wavelet='db4',
            decomposition_level=4,
            sequence_length=60,
            prediction_horizon=1
        )
        
        # Run complete pipeline
        self.results = self.engine.run_full_pipeline(
            symbol=symbol,
            n_predictions=days_ahead,
            plot_decomposition=plot_results
        )
        
        if self.results:
            self._print_basic_summary()
            
            if plot_results:
                self._create_basic_plots()
        
        return self.results
    
    def run_advanced_analysis(self, symbol: str = "CL=F", 
                            config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run advanced analysis with comprehensive features
        
        Args:
            symbol: Oil contract symbol
            config: Custom configuration dictionary
        
        Returns:
            dict: Complete analysis results
        """
        print(f"🔬 Running advanced analysis for {symbol}...")
        
        # Default advanced configuration
        default_config = {
            'wavelet': 'db4',
            'decomposition_level': 5,
            'sequence_length': 60,
            'prediction_horizon': 1,
            'prediction_days': 30,
            'model_config': {
                'trend': 'ensemble',
                'detail_1': 'bidirectional',
                'detail_2': 'cnn_lstm',
                'detail_3': 'attention',
                'detail_4': 'simple',
                'detail_5': 'simple'
            },
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'train_ratio': 0.8,
                'validation_ratio': 0.1
            },
            'risk_analysis': {
                'confidence_level': 0.95,
                'risk_tolerance': 'medium'
            },
            'visualization': {
                'plot_decomposition': True,
                'plot_training': True,
                'plot_risk': True,
                'plot_signals': True
            }
        }
        
        # Merge with user config
        if config:
            default_config.update(config)
        
        config = default_config
        
        # Initialize components
        self.engine = PredictionEngine(
            wavelet=config['wavelet'],
            decomposition_level=config['decomposition_level'],
            sequence_length=config['sequence_length'],
            prediction_horizon=config['prediction_horizon']
        )
        
        self.signal_generator = TradingSignalGenerator(
            risk_tolerance=config['risk_analysis']['risk_tolerance']
        )
        
        # Run prediction pipeline
        print("\n📊 Step 1: Running prediction pipeline...")
        self.results = self.engine.run_full_pipeline(
            symbol=symbol,
            n_predictions=config['prediction_days'],
            plot_decomposition=config['visualization']['plot_decomposition'],
            **config['training']
        )
        
        if not self.results:
            print("❌ Prediction pipeline failed")
            return None
        
        # Extract prediction data
        prediction_data = self.results['predictions']
        historical_prices = self.engine.data_processor.oil_prices
        predictions = prediction_data['predictions']
        component_predictions = prediction_data['component_predictions']
        current_price = prediction_data['current_price']
        
        # Risk analysis
        print("\n⚠️ Step 2: Performing risk analysis...")
        risk_report = self.risk_analyzer.generate_risk_report(
            historical_prices, predictions, component_predictions
        )
        
        # Trading signals
        print("\n📈 Step 3: Generating trading signals...")
        trading_signals = self.signal_generator.generate_comprehensive_signals(
            current_price, predictions, risk_report
        )
        
        # Combine results
        complete_results = {
            'prediction_results': self.results,
            'risk_analysis': risk_report,
            'trading_signals': trading_signals,
            'configuration': config,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Print reports
        self._print_advanced_summary(complete_results)
        
        # Create visualizations
        if config['visualization']['plot_training']:
            self._plot_training_results()
        
        if config['visualization']['plot_risk']:
            self._plot_risk_analysis(risk_report, historical_prices, predictions)
        
        if config['visualization']['plot_signals']:
            self._plot_trading_signals(trading_signals, current_price, predictions)
        
        # Create comprehensive summary
        create_summary_report_plot(prediction_data, risk_report, trading_signals)
        
        return complete_results
    
    def run_comparison_analysis(self, symbols: list = None, wavelets: list = None) -> Dict[str, Any]:
        """
        Run comparison analysis across different symbols or wavelets
        
        Args:
            symbols: List of symbols to compare
            wavelets: List of wavelets to compare
        
        Returns:
            dict: Comparison results
        """
        if symbols is None:
            symbols = ["CL=F"]  # Default to WTI only for demo
        
        if wavelets is None:
            wavelets = ['db4', 'db8', 'haar', 'bior2.2']
        
        print(f"🔄 Running comparison analysis...")
        print(f"Symbols: {symbols}")
        print(f"Wavelets: {wavelets}")
        
        comparison_results = {}
        
        # Compare wavelets for each symbol
        for symbol in symbols:
            print(f"\n📊 Analyzing {symbol}...")
            
            symbol_results = {}
            
            for wavelet in wavelets:
                print(f"  🌊 Testing {wavelet} wavelet...")
                
                try:
                    # Create engine with specific wavelet
                    engine = PredictionEngine(
                        wavelet=wavelet,
                        decomposition_level=4,
                        sequence_length=40,  # Shorter for faster comparison
                        prediction_horizon=1
                    )
                    
                    # Run pipeline
                    results = engine.run_full_pipeline(
                        symbol=symbol,
                        n_predictions=7,  # Shorter prediction for comparison
                        plot_decomposition=False,
                        epochs=50,  # Fewer epochs for faster training
                        batch_size=32
                    )
                    
                    if results:
                        pred_data = results['predictions']
                        current_price = pred_data['current_price']
                        next_day_pred = pred_data['predictions'][0]
                        change_pct = ((next_day_pred - current_price) / current_price) * 100
                        
                        # Get model performance
                        performance = engine.get_model_performance()
                        avg_val_loss = sum(perf['final_val_loss'] for perf in performance.values()) / len(performance)
                        
                        symbol_results[wavelet] = {
                            'prediction': next_day_pred,
                            'change_percent': change_pct,
                            'avg_validation_loss': avg_val_loss,
                            'n_models_trained': len(engine.component_models),
                            'components': list(engine.components.keys()) if engine.components else []
                        }
                        
                        print(f"    ✅ {wavelet}: ${next_day_pred:.2f} ({change_pct:+.2f}%)")
                    
                except Exception as e:
                    print(f"    ❌ {wavelet}: Error - {str(e)[:50]}...")
                    symbol_results[wavelet] = {'error': str(e)}
            
            comparison_results[symbol] = symbol_results
        
        # Print comparison summary
        self._print_comparison_summary(comparison_results)
        
        return comparison_results
    
    def save_results(self, filepath: str):
        """
        Save results and models
        
        Args:
            filepath: Base filepath for saving
        """
        if not self.results:
            print("⚠️ No results to save")
            return
        
        try:
            # Save engine models
            if self.engine:
                self.engine.save_models(f"{filepath}_models")
            
            # Save results as CSV
            if 'predictions' in self.results:
                pred_data = self.results['predictions']
                predictions_df = pd.DataFrame({
                    'day_ahead': range(1, len(pred_data['predictions']) + 1),
                    'predicted_price': pred_data['predictions'],
                    'change_from_current': [p - pred_data['current_price'] for p in pred_data['predictions']],
                    'change_percent': [((p - pred_data['current_price']) / pred_data['current_price']) * 100 
                                     for p in pred_data['predictions']]
                })
                
                predictions_df.to_csv(f"{filepath}_predictions.csv", index=False)
            
            print(f"✅ Results saved to {filepath}")
            
        except Exception as e:
            print(f"❌ Error saving results: {str(e)}")
    
    def load_results(self, filepath: str):
        """
        Load previously saved results and models
        
        Args:
            filepath: Base filepath for loading
        """
        try:
            # Initialize engine
            if not self.engine:
                self.engine = PredictionEngine()
            
            # Load models
            self.engine.load_models(f"{filepath}_models")
            
            print(f"✅ Models loaded from {filepath}")
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
    
    def _print_basic_summary(self):
        """Print basic prediction summary"""
        if not self.results or 'predictions' not in self.results:
            return
        
        pred_data = self.results['predictions']
        current_price = pred_data['current_price']
        predictions = pred_data['predictions']
        
        print(f"\n📊 PREDICTION SUMMARY")
        print("=" * 40)
        print(f"Current Price: ${current_price:.2f}")
        
        if len(predictions) >= 1:
            next_day = predictions[0]
            change = ((next_day - current_price) / current_price) * 100
            print(f"Next Day: ${next_day:.2f} ({change:+.2f}%)")
        
        if len(predictions) >= 7:
            week = predictions[6]
            change = ((week - current_price) / current_price) * 100
            print(f"1 Week: ${week:.2f} ({change:+.2f}%)")
        
        if len(predictions) >= 30:
            month = predictions[29]
            change = ((month - current_price) / current_price) * 100
            print(f"1 Month: ${month:.2f} ({change:+.2f}%)")
    
    def _print_advanced_summary(self, complete_results: Dict[str, Any]):
        """Print advanced analysis summary"""
        print(f"\n🎯 ADVANCED ANALYSIS COMPLETE")
        print("=" * 60)
        
        # Print risk report
        risk_report = complete_results['risk_analysis']
        print_risk_report(risk_report)
        
        # Print trading signals
        trading_signals = complete_results['trading_signals']
        print_trading_signals(trading_signals)
    
    def _print_comparison_summary(self, comparison_results: Dict[str, Any]):
        """Print comparison analysis summary"""
        print(f"\n📋 COMPARISON ANALYSIS SUMMARY")
        print("=" * 50)
        
        for symbol, symbol_results in comparison_results.items():
            print(f"\n🛢️ {symbol} RESULTS:")
            print("-" * 30)
            
            current_price = None
            best_wavelet = None
            best_loss = float('inf')
            
            for wavelet, result in symbol_results.items():
                if 'error' in result:
                    print(f"   {wavelet:>8}: ❌ Error")
                    continue
                
                pred_price = result.get('prediction', 0)
                change_pct = result.get('change_percent', 0)
                val_loss = result.get('avg_validation_loss', float('inf'))
                
                if current_price is None:
                    current_price = pred_price - (pred_price * change_pct / 100)
                
                status = "✅"
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_wavelet = wavelet
                    status = "🏆"
                
                print(f"   {wavelet:>8}: {status} ${pred_price:.2f} ({change_pct:+5.2f}%) | Loss: {val_loss:.4f}")
            
            if best_wavelet:
                print(f"\n   🏆 Best performer: {best_wavelet}")
    
    def _create_basic_plots(self):
        """Create basic visualization plots"""
        if not self.results:
            return
        
        # Extract data
        pred_data = self.results['predictions']
        predictions = pred_data['predictions']
        dates = self.engine.data_processor.dates
        prices = self.engine.data_processor.oil_prices
        
        # Price prediction plot
        self.visualizer.plot_price_history_with_predictions(
            dates, prices, predictions,
            title="Oil Price Prediction - Basic Analysis"
        )
    
    def _plot_training_results(self):
        """Plot training history"""
        if self.engine and self.engine.component_histories:
            self.visualizer.plot_training_history(
                self.engine.component_histories,
                title="Model Training History"
            )
    
    def _plot_risk_analysis(self, risk_report: Dict[str, Any], 
                          prices: np.ndarray, predictions: np.ndarray):
        """Plot risk analysis dashboard"""
        self.visualizer.plot_risk_analysis(
            risk_report, prices, predictions,
            title="Comprehensive Risk Analysis"
        )
    
    def _plot_trading_signals(self, signals: Dict[str, Any],
                            current_price: float, predictions: np.ndarray):
        """Plot trading signals dashboard"""
        self.visualizer.plot_trading_signals_dashboard(
            signals, current_price, predictions,
            title="Trading Signals & Recommendations"
        )

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Oil Price Prediction using Wavelets and TensorFlow 2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode basic                    # Basic prediction
  python main.py --mode advanced                 # Advanced analysis
  python main.py --mode comparison              # Compare wavelets
  python main.py --mode basic --symbol BZ=F     # Brent oil prediction
  python main.py --save results/my_analysis     # Save results
        """
    )
    
    parser.add_argument('--mode', choices=['basic', 'advanced', 'comparison'],
                       default='basic', help='Analysis mode')
    parser.add_argument('--symbol', default='CL=F',
                       help='Oil contract symbol (CL=F for WTI, BZ=F for Brent)')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days to predict ahead')
    parser.add_argument('--wavelet', default='db4',
                       help='Wavelet type for decomposition')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plotting')
    parser.add_argument('--save', type=str,
                       help='Save results to specified path')
    parser.add_argument('--load', type=str,
                       help='Load models from specified path')
    parser.add_argument('--config', type=str,
                       help='Configuration file path (JSON)')
    
    args = parser.parse_args()
    
    # Initialize application
    app = OilPricePredictionApp()
    
    # Load models if specified
    if args.load:
        app.load_results(args.load)
    
    # Load configuration if specified
    config = None
    if args.config:
        try:
            import json
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading config: {e}")
    
    # Run analysis based on mode
    try:
        if args.mode == 'basic':
            print(f"🔄 Running basic analysis...")
            results = app.run_basic_prediction(
                symbol=args.symbol,
                days_ahead=args.days,
                plot_results=not args.no_plots
            )
            
        elif args.mode == 'advanced':
            print(f"🔄 Running advanced analysis...")
            
            # Update config with command line args
            if config is None:
                config = {}
            
            config.update({
                'prediction_days': args.days,
                'wavelet': args.wavelet,
                'visualization': {
                    'plot_decomposition': not args.no_plots,
                    'plot_training': not args.no_plots,
                    'plot_risk': not args.no_plots,
                    'plot_signals': not args.no_plots
                }
            })
            
            results = app.run_advanced_analysis(
                symbol=args.symbol,
                config=config
            )
            
        elif args.mode == 'comparison':
            print(f"🔄 Running comparison analysis...")
            results = app.run_comparison_analysis(
                symbols=[args.symbol],
                wavelets=['db4', 'db8', 'haar', 'bior2.2', 'coif3']
            )
        
        # Save results if specified
        if args.save and results:
            app.save_results(args.save)
        
        print(f"\n✅ Analysis completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()