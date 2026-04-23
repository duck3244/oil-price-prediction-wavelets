"""Prediction endpoints."""

from fastapi import APIRouter, HTTPException

from api.schemas import PredictRequest, PredictResponse, WaveletListResponse
from api.services.prediction_service import list_wavelets, run_prediction

router = APIRouter()


@router.get("/wavelets", response_model=WaveletListResponse)
def get_wavelets() -> WaveletListResponse:
    return WaveletListResponse(wavelets=list_wavelets())


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        return run_prediction(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
