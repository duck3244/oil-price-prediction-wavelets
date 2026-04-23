"""Liveness/readiness endpoint."""

import tensorflow as tf
from fastapi import APIRouter

from api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        tensorflow_version=tf.__version__,
        gpu_available=bool(tf.config.list_physical_devices("GPU")),
    )
