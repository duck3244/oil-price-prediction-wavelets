"""Pydantic I/O schemas for the prediction API."""

from typing import Dict, List

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    symbol: str = Field(default="CL=F", description="Yahoo Finance contract symbol")
    days: int = Field(default=30, ge=1, le=90, description="Forecast horizon in days")
    wavelet: str = Field(default="db4", description="PyWavelets wavelet name")
    decomposition_level: int = Field(default=4, ge=1, le=8)
    sequence_length: int = Field(default=60, ge=20, le=200)
    epochs: int = Field(default=30, ge=1, le=300)


class ComponentPrediction(BaseModel):
    name: str
    values: List[float]


class PredictResponse(BaseModel):
    symbol: str
    current_price: float
    historical_dates: List[str]
    historical_prices: List[float]
    predictions: List[float]
    component_predictions: List[ComponentPrediction]
    wavelet: str
    decomposition_level: int
    generated_at: str


class WaveletListResponse(BaseModel):
    wavelets: Dict[str, List[str]]


class HealthResponse(BaseModel):
    status: str
    tensorflow_version: str
    gpu_available: bool
