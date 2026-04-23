#!/usr/bin/env python3
"""
Prediction Engine Module for Oil Price Prediction
Handles model training coordination and prediction generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import MinMaxScaler
import pickle

from data_processor import DataProcessor
from wavelet_analyzer import WaveletAnalyzer
from model_builder import ModelBuilder

class PredictionEngine:
    """
    Main prediction engine that coordinates all components
    """
    
    def __init__(self, wavelet: str = 'db4', decomposition_level: int = 5,
                 sequence_length: int = 60, prediction_horizon: int = 1):
        """
        Initialize the prediction engine
        
        Args:
            wavelet: Wavelet type for decomposition
            decomposition_level: Number of decomposition levels
            sequence_length: Length of input sequences
            prediction_horizon: Number of days to predict ahead
        """
        self.wavelet = wavelet
        self.decomposition_level = decomposition_level
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Initialize components
        self.data_processor = DataProcessor(sequence_length, prediction_horizon)
        self.wavelet_analyzer = WaveletAnalyzer(wavelet, decomposition_level)
        self.model_builder = ModelBuilder(sequence_length, prediction_horizon)
        
        # Storage for components and models
        self.components = None
        self.component_models = {}
        self.component_scalers = {}
        self.component_histories = {}
        self.is_trained = False
        
    def load_and_prepare_data(self, symbol: str = "CL=F", start_date: str = "2015-01-01",
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load and prepare oil price data
        
        Args:
            symbol: Oil contract symbol
            start_date: Start date for data
            end_date: End date for data
        
        Returns:
            DataFrame: Raw oil price data
        """
        print("🔄 Loading and preparing data...")
        
        # Fetch data
        raw_data = self.data_processor.fetch_oil_data(symbol, start_date, end_date)
        
        # Get statistics
        stats = self.data_processor.get_data_statistics()
        if stats:
            print(f"📊 Data loaded successfully:")
            print(f"   Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
            print(f"   Total points: {stats['data_points']}")
            print(f"   Current price: ${stats['price_stats']['current']:.2f}")
            print(f"   Annual volatility: {stats['returns_stats']['annual_volatility']:.1%}")
        
        return raw_data
    
    def decompose_signal(self, plot: bool = True) -> Dict[str, np.ndarray]:
        """
        Perform wavelet decomposition on oil prices
        
        Args:
            plot: Whether to plot decomposition results
        
        Returns:
            dict: Wavelet components
        """
        print(f"\n🌊 Performing wavelet decomposition...")
        
        if self.data_processor.oil_prices is None:
            raise ValueError("No data available. Please load data first.")
        
        # Decompose signal
        self.components = self.wavelet_analyzer.decompose(
            self.data_processor.oil_prices, plot=plot
        )
        
        # Analyze components
        analysis = self.wavelet_analyzer.analyze_components(self.data_processor.oil_prices)
        
        return self.components
    
    def train_component_models(self, model_config: Optional[Dict[str, str]] = None,
                             train_ratio: float = 0.8, validation_ratio: float = 0.1,
                             epochs: int = 100, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train models for each wavelet component
        
        Args:
            model_config: Configuration mapping component names to model types
            train_ratio: Ratio for training split
            validation_ratio: Ratio for validation split
            epochs: Number of training epochs
            batch_size: Training batch size
        
        Returns:
            dict: Training results
        """
        if self.components is None:
            raise ValueError("No components available. Please perform decomposition first.")
        
        print(f"\n🚀 Training component models...")
        
        # Default model configuration
        if model_config is None:
            model_config = {
                'trend': 'ensemble',
                'detail_1': 'bidirectional',
                'detail_2': 'cnn_lstm',
                'detail_3': 'attention',
                'detail_4': 'simple',
                'detail_5': 'simple'
            }
        
        training_results = {}
        
        for component_name, component_data in self.components.items():
            print(f"\n🔧 Processing {component_name} component...")
            
            # Skip if insufficient data
            if len(component_data) < self.sequence_length * 2:
                print(f"   ⚠️ Insufficient data for {component_name}, skipping...")
                continue
            
            # Get model type for this component
            model_type = model_config.get(component_name, 'simple')
            
            try:
                # Fit the scaler on the training portion only, then transform
                # the full series. This avoids leaking val/test statistics into
                # the scaling and keeps each component's scaler independent.
                train_end = int(len(component_data) * train_ratio)
                comp_scaler = MinMaxScaler(feature_range=(0, 1))
                comp_scaler.fit(component_data[:train_end].reshape(-1, 1))

                X, y, comp_scaler = self.data_processor.create_sequences(
                    component_data.reshape(-1, 1), scaler=comp_scaler
                )

                if len(X) == 0:
                    print(f"   ⚠️ No sequences generated for {component_name}")
                    continue

                # Split data
                X_train, X_val, X_test, y_train, y_val, y_test = self.data_processor.split_data(
                    X, y, train_ratio, validation_ratio
                )

                # Create model
                input_shape = (X.shape[1], X.shape[2])
                model = self.model_builder.create_model_by_type(
                    model_type, input_shape, name=f"{component_name}_{model_type}"
                )

                # Train model
                history = self.model_builder.train_model(
                    model, X_train, y_train, X_val, y_val,
                    epochs=epochs, batch_size=batch_size, verbose=0
                )

                # Store results
                self.component_models[component_name] = model
                self.component_scalers[component_name] = comp_scaler
                self.component_histories[component_name] = history
                
                # Evaluate on test set
                if len(X_test) > 0:
                    test_loss = model.evaluate(X_test, y_test, verbose=0)[0]
                    training_results[component_name] = {
                        'model_type': model_type,
                        'train_samples': len(X_train),
                        'val_samples': len(X_val),
                        'test_samples': len(X_test),
                        'final_train_loss': history.history['loss'][-1],
                        'final_val_loss': history.history['val_loss'][-1],
                        'test_loss': test_loss,
                        'parameters': model.count_params()
                    }
                else:
                    training_results[component_name] = {
                        'model_type': model_type,
                        'train_samples': len(X_train),
                        'val_samples': len(X_val),
                        'test_samples': 0,
                        'final_train_loss': history.history['loss'][-1],
                        'final_val_loss': history.history['val_loss'][-1],
                        'test_loss': None,
                        'parameters': model.count_params()
                    }
                
                print(f"   ✅ {component_name} model trained successfully")
                print(f"      Model type: {model_type}")
                print(f"      Parameters: {model.count_params():,}")
                
            except Exception as e:
                print(f"   ❌ Error training {component_name}: {str(e)}")
                continue
        
        self.is_trained = len(self.component_models) > 0
        
        if self.is_trained:
            print(f"\n🎉 Training completed! {len(self.component_models)} models trained.")
        else:
            print(f"\n⚠️ No models were trained successfully.")
        
        return training_results
    
    def predict_next_values(self, n_steps: int = 1) -> Dict[str, np.ndarray]:
        """
        Predict future values for each component
        
        Args:
            n_steps: Number of steps to predict ahead
        
        Returns:
            dict: Predictions for each component
        """
        if not self.is_trained:
            raise ValueError("Models not trained. Please train models first.")
        
        print(f"🔮 Generating predictions for next {n_steps} steps...")
        
        component_predictions = {}
        
        for component_name, model in self.component_models.items():
            try:
                # Get component data
                component_data = self.components[component_name]
                scaler = self.component_scalers[component_name]
                
                # Prepare input sequence
                scaled_data = scaler.transform(component_data.reshape(-1, 1))
                last_sequence = scaled_data[-self.sequence_length:]
                
                # Generate multi-step predictions
                predictions = []
                current_sequence = last_sequence.copy()
                
                for _ in range(n_steps):
                    # Reshape for prediction
                    X_pred = current_sequence.reshape(1, self.sequence_length, 1)
                    
                    # Make prediction
                    pred_scaled = model.predict(X_pred, verbose=0)
                    
                    # Update sequence for next prediction
                    current_sequence = np.append(current_sequence[1:], pred_scaled[0])
                    
                    # Store prediction (inverse transform)
                    pred_original = scaler.inverse_transform(pred_scaled)[0, 0]
                    predictions.append(pred_original)
                
                component_predictions[component_name] = np.array(predictions)
                print(f"   ✅ {component_name}: {len(predictions)} predictions generated")
                
            except Exception as e:
                print(f"   ❌ Error predicting {component_name}: {str(e)}")
                continue
        
        return component_predictions
    
    def reconstruct_predictions(self, component_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Reconstruct final predictions from component predictions
        
        Args:
            component_predictions: Predictions for each component
        
        Returns:
            np.ndarray: Final reconstructed predictions
        """
        if not component_predictions:
            raise ValueError("No component predictions provided")
        
        # Simple reconstruction: sum all components
        final_predictions = np.sum(list(component_predictions.values()), axis=0)
        
        return final_predictions
    
    def predict(self, n_steps: int = 1) -> Dict[str, Any]:
        """
        Complete prediction pipeline
        
        Args:
            n_steps: Number of steps to predict ahead
        
        Returns:
            dict: Complete prediction results
        """
        # Generate component predictions
        component_predictions = self.predict_next_values(n_steps)
        
        # Reconstruct final predictions
        final_predictions = self.reconstruct_predictions(component_predictions)
        
        # Current price for reference
        current_price = self.data_processor.oil_prices[-1]
        
        # Calculate changes
        changes = {}
        for i, pred in enumerate(final_predictions):
            days = i + 1
            change_pct = ((pred - current_price) / current_price) * 100
            changes[f'day_{days}'] = {
                'price': pred,
                'change_percent': change_pct,
                'change_absolute': pred - current_price
            }
        
        results = {
            'current_price': current_price,
            'predictions': final_predictions,
            'component_predictions': component_predictions,
            'changes': changes,
            'prediction_horizon': n_steps,
            'model_info': {
                'wavelet': self.wavelet,
                'decomposition_level': self.decomposition_level,
                'sequence_length': self.sequence_length,
                'n_components': len(self.components) if self.components else 0,
                'n_models': len(self.component_models)
            }
        }
        
        return results
    
    def run_full_pipeline(self, symbol: str = "CL=F", start_date: str = "2015-01-01",
                         end_date: Optional[str] = None, n_predictions: int = 30,
                         plot_decomposition: bool = True, **training_kwargs) -> Dict[str, Any]:
        """
        Run the complete prediction pipeline
        
        Args:
            symbol: Oil contract symbol
            start_date: Start date for data
            end_date: End date for data
            n_predictions: Number of days to predict
            plot_decomposition: Whether to plot wavelet decomposition
            **training_kwargs: Additional training arguments
        
        Returns:
            dict: Complete results
        """
        print("🚀 Starting complete oil price prediction pipeline...")
        print("=" * 60)
        
        try:
            # 1. Load data
            raw_data = self.load_and_prepare_data(symbol, start_date, end_date)
            
            # 2. Decompose signal
            components = self.decompose_signal(plot=plot_decomposition)
            
            # 3. Train models
            training_results = self.train_component_models(**training_kwargs)
            
            # 4. Generate predictions
            if self.is_trained:
                prediction_results = self.predict(n_predictions)
                
                # Print summary
                self._print_prediction_summary(prediction_results)
                
                return {
                    'raw_data': raw_data,
                    'components': components,
                    'training_results': training_results,
                    'predictions': prediction_results,
                    'engine_config': {
                        'wavelet': self.wavelet,
                        'decomposition_level': self.decomposition_level,
                        'sequence_length': self.sequence_length,
                        'prediction_horizon': self.prediction_horizon
                    }
                }
            else:
                print("❌ Pipeline failed - no models were trained successfully")
                return None
                
        except Exception as e:
            print(f"❌ Pipeline failed with error: {str(e)}")
            return None
    
    def _print_prediction_summary(self, results: Dict[str, Any]):
        """Print a summary of prediction results"""
        print(f"\n📊 PREDICTION SUMMARY")
        print("=" * 40)
        
        current_price = results['current_price']
        predictions = results['predictions']
        
        print(f"Current Oil Price: ${current_price:.2f}")
        print(f"Predictions generated: {len(predictions)}")
        
        # Key predictions
        if len(predictions) >= 1:
            next_day = predictions[0]
            change_1d = ((next_day - current_price) / current_price) * 100
            print(f"Next Day: ${next_day:.2f} ({change_1d:+.2f}%)")
        
        if len(predictions) >= 7:
            week_ahead = predictions[6]
            change_1w = ((week_ahead - current_price) / current_price) * 100
            print(f"1 Week: ${week_ahead:.2f} ({change_1w:+.2f}%)")
        
        if len(predictions) >= 30:
            month_ahead = predictions[29]
            change_1m = ((month_ahead - current_price) / current_price) * 100
            print(f"1 Month: ${month_ahead:.2f} ({change_1m:+.2f}%)")
        
        # Model info
        model_info = results['model_info']
        print(f"\nModel Configuration:")
        print(f"  Wavelet: {model_info['wavelet']}")
        print(f"  Components: {model_info['n_components']}")
        print(f"  Models trained: {model_info['n_models']}")
    
    def save_models(self, filepath: str):
        """
        Save trained models and configuration
        
        Args:
            filepath: Path to save the models
        """
        if not self.is_trained:
            print("⚠️ No trained models to save")
            return
        
        save_data = {
            'config': {
                'wavelet': self.wavelet,
                'decomposition_level': self.decomposition_level,
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon
            },
            'components': self.components,
            'component_scalers': self.component_scalers,
            'model_names': list(self.component_models.keys())
        }
        
        # Save configuration and data
        with open(f"{filepath}_config.pkl", 'wb') as f:
            pickle.dump(save_data, f)
        
        # Save individual models
        for name, model in self.component_models.items():
            model.save(f"{filepath}_{name}_model.h5")
        
        print(f"✅ Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """
        Load trained models and configuration
        
        Args:
            filepath: Path to load the models from
        """
        try:
            # Load configuration and data
            with open(f"{filepath}_config.pkl", 'rb') as f:
                save_data = pickle.load(f)
            
            # Restore configuration
            config = save_data['config']
            self.wavelet = config['wavelet']
            self.decomposition_level = config['decomposition_level']
            self.sequence_length = config['sequence_length']
            self.prediction_horizon = config['prediction_horizon']
            
            # Restore components and scalers
            self.components = save_data['components']
            self.component_scalers = save_data['component_scalers']
            
            # Load individual models
            from tensorflow import keras
            self.component_models = {}
            
            for model_name in save_data['model_names']:
                model_path = f"{filepath}_{model_name}_model.h5"
                self.component_models[model_name] = keras.models.load_model(model_path)
            
            # Update components
            self.data_processor = DataProcessor(self.sequence_length, self.prediction_horizon)
            self.wavelet_analyzer = WaveletAnalyzer(self.wavelet, self.decomposition_level)
            self.model_builder = ModelBuilder(self.sequence_length, self.prediction_horizon)
            
            self.is_trained = True
            
            print(f"✅ Models loaded from {filepath}")
            print(f"   Loaded {len(self.component_models)} models")
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            self.is_trained = False
    
    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics for all trained models
        
        Returns:
            dict: Performance metrics for each model
        """
        if not self.is_trained:
            return {}
        
        performance = {}
        
        for name, history in self.component_histories.items():
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            min_val_loss = min(history.history['val_loss'])
            
            performance[name] = {
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                'best_val_loss': min_val_loss,
                'improvement': ((final_train_loss - final_val_loss) / final_train_loss) * 100,
                'epochs_trained': len(history.history['loss'])
            }
        
        return performance