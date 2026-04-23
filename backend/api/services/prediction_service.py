"""Thin service layer that adapts PredictionEngine output to API schemas."""

from datetime import datetime, timezone
from typing import Dict, List

import numpy as np

from predictor_engine import PredictionEngine
from wavelet_analyzer import get_available_wavelets

from api.schemas import ComponentPrediction, PredictRequest, PredictResponse


# Limit how much history we return to the client so payloads stay bounded
# regardless of how far back the data processor fetched.
_HISTORY_WINDOW_DAYS = 252


def list_wavelets() -> Dict[str, List[str]]:
    return get_available_wavelets()


def run_prediction(req: PredictRequest) -> PredictResponse:
    engine = PredictionEngine(
        wavelet=req.wavelet,
        decomposition_level=req.decomposition_level,
        sequence_length=req.sequence_length,
        prediction_horizon=1,
    )

    results = engine.run_full_pipeline(
        symbol=req.symbol,
        n_predictions=req.days,
        plot_decomposition=False,
        epochs=req.epochs,
        batch_size=32,
    )
    if not results:
        raise ValueError("prediction pipeline failed to produce any results")

    pred = results["predictions"]
    dates = engine.data_processor.dates
    prices = engine.data_processor.oil_prices

    component_preds = [
        ComponentPrediction(name=name, values=np.asarray(vals, dtype=float).tolist())
        for name, vals in pred["component_predictions"].items()
    ]

    return PredictResponse(
        symbol=req.symbol,
        current_price=float(pred["current_price"]),
        historical_dates=[d.strftime("%Y-%m-%d") for d in dates[-_HISTORY_WINDOW_DAYS:]],
        historical_prices=[float(p) for p in prices[-_HISTORY_WINDOW_DAYS:]],
        predictions=[float(x) for x in pred["predictions"]],
        component_predictions=component_preds,
        wavelet=req.wavelet,
        decomposition_level=req.decomposition_level,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
