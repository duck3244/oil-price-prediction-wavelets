#!/usr/bin/env python3
"""
Wavelet Analysis Module for Oil Price Prediction
Handles wavelet decomposition and reconstruction
"""

import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


class WaveletAnalyzer:
    """
    Comprehensive wavelet analysis class for time series decomposition.
    """
    
    def __init__(self, wavelet='db4', decomposition_level=5):
        """
        Initialize the wavelet analyzer
        
        Args:
            wavelet: Wavelet type (e.g., 'db4', 'haar', 'bior2.2')
            decomposition_level: Number of decomposition levels
        """
        self.wavelet = wavelet
        self.decomposition_level = decomposition_level
        self.coefficients = None
        self.components = None
        self.original_length = None
        
        # Validate wavelet
        try:
            pywt.Wavelet(wavelet)
        except ValueError:
            print(f"Warning: Wavelet '{wavelet}' not found. Using 'db4' instead.")
            self.wavelet = 'db4'
    
    def decompose(self, signal: np.ndarray, plot: bool = False) -> Dict[str, np.ndarray]:
        """
        Perform multi-level discrete wavelet decomposition
        
        Args:
            signal: Input time series signal
            plot: Whether to plot the decomposition results
        
        Returns:
            dict: Dictionary containing decomposed components
        """
        print(f"Performing {self.decomposition_level}-level decomposition using {self.wavelet} wavelet...")
        
        # Store original length
        self.original_length = len(signal)
        
        # Perform decomposition
        self.coefficients = pywt.wavedec(signal, self.wavelet, level=self.decomposition_level)
        
        # Reconstruct individual components
        self.components = self._reconstruct_components(signal)
        
        if plot:
            self.plot_decomposition(signal)
        
        return self.components
    
    def _reconstruct_components(self, original_signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Reconstruct individual wavelet components
        
        Args:
            original_signal: Original time series signal
        
        Returns:
            dict: Dictionary of reconstructed components
        """
        components = {}
        
        # Approximation component (trend)
        approx_coeffs = [self.coefficients[0]] + [np.zeros_like(c) for c in self.coefficients[1:]]
        components['trend'] = pywt.waverec(approx_coeffs, self.wavelet)[:len(original_signal)]
        
        # Detail components (high-frequency patterns)
        for i, detail in enumerate(self.coefficients[1:]):
            detail_coeffs = [np.zeros_like(self.coefficients[0])]
            for j, c in enumerate(self.coefficients[1:]):
                if j == i:
                    detail_coeffs.append(c)
                else:
                    detail_coeffs.append(np.zeros_like(c))
            
            components[f'detail_{i+1}'] = pywt.waverec(detail_coeffs, self.wavelet)[:len(original_signal)]
        
        return components
    
    def plot_decomposition(self, original_signal: np.ndarray, dates: Optional[pd.DatetimeIndex] = None):
        """
        Plot the wavelet decomposition results
        
        Args:
            original_signal: Original time series signal
            dates: Optional datetime index for x-axis
        """
        n_components = len(self.components)
        fig, axes = plt.subplots(n_components + 1, 1, figsize=(15, 3 * (n_components + 1)))
        
        if dates is None:
            dates = range(len(original_signal))
        
        # Original signal
        axes[0].plot(dates[:len(original_signal)], original_signal, 'k-', linewidth=1.5)
        axes[0].set_title('Original Signal', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        # Components
        colors = plt.cm.tab10(np.linspace(0, 1, n_components))
        for i, (name, component) in enumerate(self.components.items()):
            axes[i + 1].plot(dates[:len(component)], component, color=colors[i], linewidth=1.5)
            
            # Add statistics to title
            mean_val = np.mean(component)
            std_val = np.std(component)
            title = f'{name.replace("_", " ").title()}'
            
            if name == 'trend':
                title += f' (μ={mean_val:.2f}, σ={std_val:.2f})'
            else:
                title += f' (σ={std_val:.2f})'
            
            axes[i + 1].set_title(title, fontsize=12)
            axes[i + 1].set_ylabel('Amplitude')
            axes[i + 1].grid(True, alpha=0.3)
            
            # Add zero line for detail components
            if name != 'trend':
                axes[i + 1].axhline(0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_components(self, original_signal: np.ndarray) -> Dict[str, Dict]:
        """
        Analyze the characteristics of each wavelet component
        
        Args:
            original_signal: Original time series signal
        
        Returns:
            dict: Analysis results for each component
        """
        if self.components is None:
            raise ValueError("No decomposition found. Please run decompose() first.")
        
        total_variance = np.var(original_signal)
        analysis = {}
        
        print("Component Analysis:")
        print("-" * 50)
        
        for name, component in self.components.items():
            # Calculate statistics
            variance = np.var(component)
            variance_contrib = variance / total_variance * 100
            mean_val = np.mean(component)
            std_val = np.std(component)
            
            # Calculate energy (sum of squares)
            energy = np.sum(component**2)
            energy_contrib = energy / np.sum(original_signal**2) * 100
            
            # Calculate frequency characteristics
            fft_component = np.fft.fft(component)
            dominant_freq = np.argmax(np.abs(fft_component[1:len(fft_component)//2])) + 1
            
            component_analysis = {
                'variance': variance,
                'variance_contribution': variance_contrib,
                'energy_contribution': energy_contrib,
                'mean': mean_val,
                'std': std_val,
                'min': np.min(component),
                'max': np.max(component),
                'dominant_frequency_bin': dominant_freq,
                'component_type': 'trend' if name == 'trend' else 'detail'
            }
            
            analysis[name] = component_analysis
            
            print(f"{name:>12}: Var={variance:.2f} ({variance_contrib:5.1f}%), "
                  f"Energy={energy_contrib:5.1f}%, σ={std_val:.2f}")
        
        return analysis
    
    def reconstruct_signal(self, components: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """
        Reconstruct signal from components
        
        Args:
            components: Dictionary of components (uses self.components if None)
        
        Returns:
            np.ndarray: Reconstructed signal
        """
        if components is None:
            components = self.components
        
        if components is None:
            raise ValueError("No components available for reconstruction")
        
        # Sum all components
        reconstructed = np.sum(list(components.values()), axis=0)
        
        return reconstructed
    
    def validate_reconstruction(self, original_signal: np.ndarray, tolerance: float = 1e-10) -> Dict[str, float]:
        """
        Validate the quality of signal reconstruction
        
        Args:
            original_signal: Original signal to compare against
            tolerance: Acceptable error tolerance
        
        Returns:
            dict: Reconstruction quality metrics
        """
        reconstructed = self.reconstruct_signal()
        
        # Trim to same length
        min_len = min(len(original_signal), len(reconstructed))
        orig_trimmed = original_signal[:min_len]
        recon_trimmed = reconstructed[:min_len]
        
        # Calculate metrics
        mse = np.mean((orig_trimmed - recon_trimmed)**2)
        mae = np.mean(np.abs(orig_trimmed - recon_trimmed))
        max_error = np.max(np.abs(orig_trimmed - recon_trimmed))
        
        # Correlation
        correlation = np.corrcoef(orig_trimmed, recon_trimmed)[0, 1]
        
        # Quality assessment
        is_perfect = mse < tolerance
        
        validation_results = {
            'mse': mse,
            'mae': mae,
            'max_error': max_error,
            'correlation': correlation,
            'is_perfect_reconstruction': is_perfect,
            'quality': 'Excellent' if mse < 1e-10 else 'Good' if mse < 1e-6 else 'Fair'
        }
        
        print(f"\nReconstruction Validation:")
        print(f"MSE: {mse:.2e}")
        print(f"MAE: {mae:.2e}")
        print(f"Max Error: {max_error:.2e}")
        print(f"Correlation: {correlation:.6f}")
        print(f"Quality: {validation_results['quality']}")
        
        return validation_results
    
    def denoise_signal(self, signal: np.ndarray, threshold_type: str = 'soft', 
                      threshold_mode: str = 'symmetric') -> np.ndarray:
        """
        Denoise signal using wavelet thresholding
        
        Args:
            signal: Input signal to denoise
            threshold_type: 'soft' or 'hard' thresholding
            threshold_mode: Thresholding mode
        
        Returns:
            np.ndarray: Denoised signal
        """
        # Calculate noise level using MAD (Median Absolute Deviation)
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.decomposition_level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        
        # Calculate threshold
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        
        # Apply thresholding to detail coefficients
        coeffs_thresh = coeffs.copy()
        for i in range(1, len(coeffs)):
            coeffs_thresh[i] = pywt.threshold(coeffs[i], threshold, mode=threshold_type)
        
        # Reconstruct denoised signal
        denoised = pywt.waverec(coeffs_thresh, self.wavelet)
        
        return denoised[:len(signal)]
    
    def compare_wavelets(self, signal: np.ndarray, wavelet_list: List[str]) -> Dict[str, Dict]:
        """
        Compare different wavelets for the given signal
        
        Args:
            signal: Input signal
            wavelet_list: List of wavelets to compare
        
        Returns:
            dict: Comparison results
        """
        comparison_results = {}
        
        print("Wavelet Comparison Analysis:")
        print("-" * 50)
        
        for wavelet in wavelet_list:
            try:
                # Create temporary analyzer
                temp_analyzer = WaveletAnalyzer(wavelet, self.decomposition_level)
                components = temp_analyzer.decompose(signal, plot=False)
                
                # Analyze reconstruction quality
                reconstructed = temp_analyzer.reconstruct_signal(components)
                min_len = min(len(signal), len(reconstructed))
                
                mse = np.mean((signal[:min_len] - reconstructed[:min_len])**2)
                variance_explained = 1 - np.var(signal[:min_len] - reconstructed[:min_len]) / np.var(signal[:min_len])
                
                # Calculate component distribution
                component_analysis = temp_analyzer.analyze_components(signal)
                trend_contribution = component_analysis.get('trend', {}).get('variance_contribution', 0)
                
                comparison_results[wavelet] = {
                    'reconstruction_mse': mse,
                    'variance_explained': variance_explained,
                    'trend_contribution': trend_contribution,
                    'n_components': len(components),
                    'components': component_analysis
                }
                
                print(f"{wavelet:>8}: MSE={mse:.2e}, Var_Exp={variance_explained:.3f}, "
                      f"Trend={trend_contribution:.1f}%")
                
            except Exception as e:
                print(f"{wavelet:>8}: Error - {str(e)[:30]}...")
                comparison_results[wavelet] = {'error': str(e)}
        
        return comparison_results
    
    def extract_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract wavelet-based features from the signal
        
        Args:
            signal: Input signal
        
        Returns:
            dict: Extracted features
        """
        if self.components is None:
            self.decompose(signal, plot=False)
        
        features = {}
        
        # Energy features
        total_energy = np.sum(signal**2)
        for name, component in self.components.items():
            energy = np.sum(component**2)
            features[f'{name}_energy_ratio'] = energy / total_energy
        
        # Statistical features
        for name, component in self.components.items():
            features[f'{name}_mean'] = np.mean(component)
            features[f'{name}_std'] = np.std(component)
            features[f'{name}_skewness'] = self._calculate_skewness(component)
            features[f'{name}_kurtosis'] = self._calculate_kurtosis(component)
        
        # Frequency domain features
        trend_component = self.components.get('trend', signal)
        trend_diff = np.diff(trend_component)
        features['trend_slope'] = np.mean(trend_diff)
        features['trend_volatility'] = np.std(trend_diff)
        
        return features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        if len(data) == 0:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        if len(data) == 0:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

# Utility functions for wavelet analysis
def get_available_wavelets() -> Dict[str, List[str]]:
    """
    Get all available wavelets organized by family
    
    Returns:
        dict: Wavelets organized by family
    """
    wavelet_families = {
        'Daubechies': [f'db{i}' for i in range(1, 21)],
        'Biorthogonal': [f'bior{i}.{j}' for i in range(1, 7) for j in range(1, 9) if f'bior{i}.{j}' in pywt.wavelist()],
        'Coiflets': [f'coif{i}' for i in range(1, 18)],
        'Haar': ['haar'],
        'Discrete Meyer': ['dmey'],
        'Symlets': [f'sym{i}' for i in range(2, 21)]
    }
    
    # Filter only available wavelets
    available_families = {}
    for family, wavelets in wavelet_families.items():
        available = [w for w in wavelets if w in pywt.wavelist()]
        if available:
            available_families[family] = available
    
    return available_families

def recommend_wavelet(signal: np.ndarray, signal_type: str = 'financial') -> str:
    """
    Recommend optimal wavelet based on signal characteristics
    
    Args:
        signal: Input signal
        signal_type: Type of signal ('financial', 'smooth', 'spiky')
    
    Returns:
        str: Recommended wavelet name
    """
    recommendations = {
        'financial': 'db4',      # Good for financial time series
        'smooth': 'db8',         # Better for smooth signals
        'spiky': 'haar',         # Good for signals with discontinuities
        'general': 'db4'         # General purpose
    }
    
    # Additional analysis could be added here to automatically determine
    # the best wavelet based on signal characteristics
    
    recommended = recommendations.get(signal_type, 'db4')
    
    print(f"Recommended wavelet for {signal_type} signal: {recommended}")
    
    return recommended

def plot_wavelet_functions(wavelet_list: List[str], level: int = 8):
    """
    Plot wavelet and scaling functions for comparison
    
    Args:
        wavelet_list: List of wavelets to plot
        level: Decomposition level for function generation
    """
    n_wavelets = len(wavelet_list)
    fig, axes = plt.subplots(2, n_wavelets, figsize=(4 * n_wavelets, 8))
    
    if n_wavelets == 1:
        axes = axes.reshape(2, 1)
    
    for i, wavelet_name in enumerate(wavelet_list):
        try:
            wavelet = pywt.Wavelet(wavelet_name)
            phi, psi, x = wavelet.wavefun(level=level)
            
            # Scaling function
            axes[0, i].plot(x, phi, 'b-', linewidth=2)
            axes[0, i].set_title(f'{wavelet_name.upper()}\nScaling Function φ', fontweight='bold')
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].set_ylabel('Amplitude')
            
            # Wavelet function
            axes[1, i].plot(x, psi, 'r-', linewidth=2)
            axes[1, i].set_title(f'Wavelet Function ψ', fontweight='bold')
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].set_ylabel('Amplitude')
            axes[1, i].set_xlabel('Time')
            
        except Exception as e:
            axes[0, i].text(0.5, 0.5, f'Error loading\n{wavelet_name}', 
                           ha='center', va='center', transform=axes[0, i].transAxes)
            axes[1, i].text(0.5, 0.5, f'{str(e)[:20]}...', 
                           ha='center', va='center', transform=axes[1, i].transAxes)
    
    plt.tight_layout()
    plt.show()