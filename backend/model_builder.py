#!/usr/bin/env python3
"""
Model Builder Module for Oil Price Prediction
Handles LSTM model creation and training with TensorFlow 2.0
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from typing import Dict, List, Tuple, Optional


class ModelBuilder:
    """
    Comprehensive model building class for oil price prediction
    """

    def __init__(self, sequence_length: int = 60, prediction_horizon: int = 1,
                 seed: Optional[int] = 42):
        """
        Initialize the model builder

        Args:
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps to predict ahead
            seed: Optional random seed for reproducibility. Pass None to skip
                seeding (e.g. when multiple builders are created in a process
                and global seed reset is undesirable).
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.models = {}
        self.histories = {}

        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU available: {len(gpus)} device(s)")
            # Enable memory growth to avoid allocation issues
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            print("No GPU available, using CPU")
    
    def create_simple_lstm(self, input_shape: Tuple[int, int], units: int = 64, 
                          dropout: float = 0.2, name: str = "simple_lstm") -> Model:
        """
        Create a simple LSTM model
        
        Args:
            input_shape: Shape of input sequences (timesteps, features)
            units: Number of LSTM units
            dropout: Dropout rate
            name: Model name
        
        Returns:
            tf.keras.Model: Compiled LSTM model
        """
        model = Sequential([
            layers.LSTM(units, return_sequences=True, input_shape=input_shape,
                       dropout=dropout, recurrent_dropout=dropout, name=f'{name}_lstm1'),
            layers.LSTM(units//2, dropout=dropout, name=f'{name}_lstm2'),
            layers.Dense(32, activation='relu', name=f'{name}_dense1'),
            layers.Dropout(dropout),
            layers.Dense(16, activation='relu', name=f'{name}_dense2'),
            layers.Dense(self.prediction_horizon, name=f'{name}_output')
        ], name=name)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',  # Robust to outliers
            metrics=['mae', 'mse']
        )
        
        return model
    
    def create_bidirectional_lstm(self, input_shape: Tuple[int, int], units: int = 64,
                                 dropout: float = 0.2, name: str = "bidirectional_lstm") -> Model:
        """
        Create a bidirectional LSTM model
        
        Args:
            input_shape: Shape of input sequences
            units: Number of LSTM units
            dropout: Dropout rate
            name: Model name
        
        Returns:
            tf.keras.Model: Compiled bidirectional LSTM model
        """
        model = Sequential([
            layers.Bidirectional(
                layers.LSTM(units, return_sequences=True, dropout=dropout),
                input_shape=input_shape, name=f'{name}_bilstm1'
            ),
            layers.Bidirectional(
                layers.LSTM(units//2, dropout=dropout),
                name=f'{name}_bilstm2'
            ),
            layers.Dense(32, activation='relu', name=f'{name}_dense1'),
            layers.Dropout(dropout),
            layers.Dense(16, activation='relu', name=f'{name}_dense2'),
            layers.Dense(self.prediction_horizon, name=f'{name}_output')
        ], name=name)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def create_cnn_lstm(self, input_shape: Tuple[int, int], filters: int = 32,
                       lstm_units: int = 64, dropout: float = 0.2, 
                       name: str = "cnn_lstm") -> Model:
        """
        Create a CNN-LSTM hybrid model
        
        Args:
            input_shape: Shape of input sequences
            filters: Number of CNN filters
            lstm_units: Number of LSTM units
            dropout: Dropout rate
            name: Model name
        
        Returns:
            tf.keras.Model: Compiled CNN-LSTM model
        """
        model = Sequential([
            layers.Conv1D(filters=filters, kernel_size=3, activation='relu',
                         input_shape=input_shape, name=f'{name}_conv1'),
            layers.BatchNormalization(name=f'{name}_bn1'),
            layers.Conv1D(filters=filters//2, kernel_size=3, activation='relu',
                         name=f'{name}_conv2'),
            layers.MaxPooling1D(pool_size=2, name=f'{name}_maxpool'),
            layers.LSTM(lstm_units, dropout=dropout, name=f'{name}_lstm'),
            layers.Dense(32, activation='relu', name=f'{name}_dense1'),
            layers.Dropout(dropout),
            layers.Dense(16, activation='relu', name=f'{name}_dense2'),
            layers.Dense(self.prediction_horizon, name=f'{name}_output')
        ], name=name)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def create_attention_lstm(self, input_shape: Tuple[int, int], units: int = 64,
                            dropout: float = 0.2, name: str = "attention_lstm") -> Model:
        """
        Create an LSTM model with attention mechanism
        
        Args:
            input_shape: Shape of input sequences
            units: Number of LSTM units
            dropout: Dropout rate
            name: Model name
        
        Returns:
            tf.keras.Model: Compiled attention LSTM model
        """
        inputs = layers.Input(shape=input_shape, name=f'{name}_input')
        
        # LSTM layers
        lstm_out = layers.LSTM(units, return_sequences=True, dropout=dropout,
                              name=f'{name}_lstm1')(inputs)
        lstm_out = layers.LSTM(units//2, return_sequences=True, dropout=dropout,
                              name=f'{name}_lstm2')(lstm_out)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh', name=f'{name}_attention_dense')(lstm_out)
        attention = layers.Flatten(name=f'{name}_attention_flatten')(attention)
        attention = layers.Activation('softmax', name=f'{name}_attention_softmax')(attention)
        attention = layers.RepeatVector(units//2, name=f'{name}_attention_repeat')(attention)
        attention = layers.Permute([2, 1], name=f'{name}_attention_permute')(attention)
        
        # Apply attention
        sent_representation = layers.Multiply(name=f'{name}_attention_multiply')([lstm_out, attention])
        sent_representation = layers.Lambda(lambda xin: tf.reduce_sum(xin, axis=1),
                                          name=f'{name}_attention_sum')(sent_representation)
        
        # Output layers
        dense_out = layers.Dense(32, activation='relu', name=f'{name}_dense1')(sent_representation)
        dense_out = layers.Dropout(dropout)(dense_out)
        dense_out = layers.Dense(16, activation='relu', name=f'{name}_dense2')(dense_out)
        outputs = layers.Dense(self.prediction_horizon, name=f'{name}_output')(dense_out)
        
        model = Model(inputs=inputs, outputs=outputs, name=name)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def create_ensemble_model(self, input_shape: Tuple[int, int], name: str = "ensemble") -> Model:
        """
        Create an ensemble model combining different architectures
        
        Args:
            input_shape: Shape of input sequences
            name: Model name
        
        Returns:
            tf.keras.Model: Compiled ensemble model
        """
        inputs = layers.Input(shape=input_shape, name=f'{name}_input')
        
        # Branch 1: Standard LSTM
        lstm1 = layers.LSTM(64, return_sequences=True, dropout=0.2,
                           name=f'{name}_lstm1_1')(inputs)
        lstm1 = layers.LSTM(32, dropout=0.2, name=f'{name}_lstm1_2')(lstm1)
        dense1 = layers.Dense(16, activation='relu', name=f'{name}_dense1')(lstm1)
        output1 = layers.Dense(self.prediction_horizon, name=f'{name}_output1')(dense1)
        
        # Branch 2: Bidirectional LSTM
        lstm2 = layers.Bidirectional(
            layers.LSTM(32, return_sequences=True, dropout=0.2),
            name=f'{name}_bilstm2_1'
        )(inputs)
        lstm2 = layers.LSTM(16, dropout=0.2, name=f'{name}_lstm2_2')(lstm2)
        dense2 = layers.Dense(8, activation='relu', name=f'{name}_dense2')(lstm2)
        output2 = layers.Dense(self.prediction_horizon, name=f'{name}_output2')(dense2)
        
        # Branch 3: CNN-LSTM
        conv = layers.Conv1D(filters=32, kernel_size=3, activation='relu',
                           name=f'{name}_conv3')(inputs)
        conv = layers.MaxPooling1D(pool_size=2, name=f'{name}_maxpool3')(conv)
        lstm3 = layers.LSTM(16, dropout=0.2, name=f'{name}_lstm3')(conv)
        dense3 = layers.Dense(8, activation='relu', name=f'{name}_dense3')(lstm3)
        output3 = layers.Dense(self.prediction_horizon, name=f'{name}_output3')(dense3)
        
        # Ensemble output (weighted average)
        ensemble_output = layers.Average(name=f'{name}_ensemble')([output1, output2, output3])
        
        model = Model(inputs=inputs, outputs=ensemble_output, name=name)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def create_advanced_ensemble(self, input_shape: Tuple[int, int], 
                               name: str = "advanced_ensemble") -> Model:
        """
        Create an advanced ensemble with learned weights
        
        Args:
            input_shape: Shape of input sequences
            name: Model name
        
        Returns:
            tf.keras.Model: Compiled advanced ensemble model
        """
        inputs = layers.Input(shape=input_shape, name=f'{name}_input')
        
        # Multiple expert models
        experts = []
        
        # Expert 1: Deep LSTM
        lstm1 = layers.LSTM(128, return_sequences=True, dropout=0.2)(inputs)
        lstm1 = layers.LSTM(64, return_sequences=True, dropout=0.2)(lstm1)
        lstm1 = layers.LSTM(32, dropout=0.2)(lstm1)
        expert1 = layers.Dense(self.prediction_horizon)(lstm1)
        experts.append(expert1)
        
        # Expert 2: Wide LSTM with residual
        lstm2 = layers.LSTM(64, return_sequences=True, dropout=0.2)(inputs)
        lstm2_out = layers.LSTM(64, dropout=0.2)(lstm2)
        # Residual connection (simplified)
        if input_shape[0] >= 64:  # Only if sequence is long enough
            residual = layers.Dense(64, activation='relu')(layers.Flatten()(inputs))
            lstm2_out = layers.Add()([lstm2_out, residual])
        expert2 = layers.Dense(self.prediction_horizon)(lstm2_out)
        experts.append(expert2)
        
        # Expert 3: Attention-based. Score each timestep with Dense(1), then
        # normalise across the time axis so the weights actually sum to 1.
        attention_lstm = layers.LSTM(64, return_sequences=True, dropout=0.2)(inputs)
        attention_scores = layers.Dense(1)(attention_lstm)
        attention_weights = layers.Softmax(axis=1)(attention_scores)
        attention_applied = layers.Multiply()([attention_lstm, attention_weights])
        attention_sum = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(attention_applied)
        expert3 = layers.Dense(self.prediction_horizon)(attention_sum)
        experts.append(expert3)
        
        # Gating network to learn ensemble weights
        gate_input = layers.LSTM(32, dropout=0.2)(inputs)
        gate_weights = layers.Dense(len(experts), activation='softmax', 
                                  name=f'{name}_gate')(gate_input)
        
        # Weighted combination
        expert_stack = layers.Lambda(lambda x: tf.stack(x, axis=1), 
                                   name=f'{name}_expert_stack')(experts)
        gate_weights_expanded = layers.RepeatVector(self.prediction_horizon)(gate_weights)
        gate_weights_expanded = layers.Permute([2, 1])(gate_weights_expanded)
        
        ensemble_output = layers.Lambda(
            lambda x: tf.reduce_sum(x[0] * x[1], axis=1),
            name=f'{name}_weighted_sum'
        )([expert_stack, gate_weights_expanded])
        
        model = Model(inputs=inputs, outputs=ensemble_output, name=name)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train_model(self, model: Model, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray, epochs: int = 100,
                   batch_size: int = 32, verbose: int = 1, 
                   save_best: bool = True) -> keras.callbacks.History:
        """
        Train a model with comprehensive callbacks
        
        Args:
            model: Model to train
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Maximum number of epochs
            batch_size: Training batch size
            verbose: Verbosity level
            save_best: Whether to save the best model
        
        Returns:
            History: Training history
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.0001,
                verbose=verbose
            )
        ]
        
        if save_best:
            callbacks.append(
                ModelCheckpoint(
                    filepath=f'best_{model.name}.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=verbose
                )
            )
        
        print(f"\n🔧 Training {model.name}...")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Model parameters: {model.count_params():,}")
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Store model and history
        self.models[model.name] = model
        self.histories[model.name] = history
        
        # Print training summary
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        print(f"   ✅ Training completed!")
        print(f"   Final train loss: {final_train_loss:.6f}")
        print(f"   Final validation loss: {final_val_loss:.6f}")
        print(f"   Best epoch: {len(history.history['loss']) - 20}")  # Approximate
        
        return history
    
    def create_model_by_type(self, model_type: str, input_shape: Tuple[int, int],
                           **kwargs) -> Model:
        """
        Create a model by type name
        
        Args:
            model_type: Type of model to create
            input_shape: Shape of input sequences
            **kwargs: Additional arguments
        
        Returns:
            tf.keras.Model: Created model
        """
        model_creators = {
            'simple': self.create_simple_lstm,
            'bidirectional': self.create_bidirectional_lstm,
            'cnn_lstm': self.create_cnn_lstm,
            'attention': self.create_attention_lstm,
            'ensemble': self.create_ensemble_model,
            'advanced_ensemble': self.create_advanced_ensemble
        }
        
        if model_type not in model_creators:
            available_types = list(model_creators.keys())
            raise ValueError(f"Unknown model type '{model_type}'. "
                           f"Available types: {available_types}")
        
        return model_creators[model_type](input_shape, **kwargs)
    
    def get_model_summary(self, model_name: Optional[str] = None) -> None:
        """
        Print summary of models
        
        Args:
            model_name: Specific model name (prints all if None)
        """
        if model_name:
            if model_name in self.models:
                print(f"\n📋 {model_name.upper()} MODEL SUMMARY:")
                print("-" * 50)
                self.models[model_name].summary()
            else:
                print(f"Model '{model_name}' not found")
        else:
            print(f"\n📋 ALL MODELS SUMMARY:")
            print("-" * 50)
            for name, model in self.models.items():
                print(f"\n{name.upper()}:")
                print(f"  Parameters: {model.count_params():,}")
                print(f"  Layers: {len(model.layers)}")
                print(f"  Input shape: {model.input_shape}")
                print(f"  Output shape: {model.output_shape}")

# Utility functions for model configuration
def get_optimal_batch_size(n_samples: int, sequence_length: int) -> int:
    """
    Get optimal batch size based on data size and memory constraints
    
    Args:
        n_samples: Number of training samples
        sequence_length: Length of sequences
    
    Returns:
        int: Optimal batch size
    """
    # Simple heuristic based on data size
    if n_samples < 1000:
        return min(16, n_samples // 4)
    elif n_samples < 5000:
        return 32
    elif n_samples < 10000:
        return 64
    else:
        return 128

def calculate_model_complexity(model: Model) -> Dict[str, int]:
    """
    Calculate model complexity metrics
    
    Args:
        model: Keras model
    
    Returns:
        dict: Complexity metrics
    """
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    # Layer type analysis
    layer_types = {}
    for layer in model.layers:
        layer_type = type(layer).__name__
        layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params,
        'total_layers': len(model.layers),
        'layer_types': layer_types
    }