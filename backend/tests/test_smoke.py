"""Smoke tests for the oil price prediction pipeline.

These tests exercise the happy path on synthetic data so regressions in the
core plumbing (decomposition, per-component scalers, training, prediction)
surface quickly without hitting the network or running a full training job.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor import DataProcessor
from predictor_engine import PredictionEngine
from wavelet_analyzer import WaveletAnalyzer


def test_create_sequences_returns_independent_scaler():
    """Each call must produce its own fitted scaler — regression for the
    scaler-sharing bug where all components reused a single MinMaxScaler."""
    dp = DataProcessor(sequence_length=10)
    a = np.linspace(0, 100, 200).reshape(-1, 1)
    b = np.linspace(-5, 5, 200).reshape(-1, 1)

    _, _, scaler_a = dp.create_sequences(a)
    _, _, scaler_b = dp.create_sequences(b)

    assert scaler_a is not scaler_b
    assert scaler_a.data_min_[0] == pytest.approx(0.0)
    assert scaler_a.data_max_[0] == pytest.approx(100.0)
    assert scaler_b.data_min_[0] == pytest.approx(-5.0)
    assert scaler_b.data_max_[0] == pytest.approx(5.0)


def test_wavelet_decomposition_reconstructs_signal():
    signal = np.sin(np.linspace(0, 20 * np.pi, 1024)) + 0.1 * np.random.RandomState(0).randn(1024)
    wa = WaveletAnalyzer(wavelet='db4', decomposition_level=4)
    wa.decompose(signal, plot=False)
    reconstructed = wa.reconstruct_signal()
    np.testing.assert_allclose(reconstructed[: len(signal)], signal, atol=1e-10)


def test_pipeline_smoke_on_synthetic_data():
    """End-to-end: synthetic data → decompose → train (tiny) → predict.

    Uses short sequences, shallow decomposition and 2 epochs to keep runtime
    bounded; the goal is to catch wiring regressions, not to measure accuracy.
    """
    engine = PredictionEngine(
        sequence_length=20, decomposition_level=2, prediction_horizon=1
    )
    engine.data_processor._generate_synthetic_data(n_points=400)
    engine.decompose_signal(plot=False)

    results = engine.train_component_models(
        model_config={'trend': 'simple', 'detail_1': 'simple', 'detail_2': 'simple'},
        epochs=2,
        batch_size=16,
    )
    assert results, 'expected at least one component to train'
    assert set(engine.component_scalers.keys()) == set(engine.components.keys())

    # Each component's scaler must be distinct — regression guard for #1.
    scaler_ids = {id(s) for s in engine.component_scalers.values()}
    assert len(scaler_ids) == len(engine.component_scalers)

    preds = engine.predict(n_steps=3)
    assert preds['predictions'].shape == (3,)
    assert np.all(np.isfinite(preds['predictions']))
    assert preds['current_price'] > 0
