"""FastAPI application factory.

Run locally from `backend/`:
    uvicorn api.app:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import health, predictions


def create_app() -> FastAPI:
    app = FastAPI(title="Oil Price Prediction API", version="0.1.0")

    # Vite dev server proxies /api, but keep explicit CORS so the frontend
    # can also be served from a different origin during manual testing.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(predictions.router, prefix="/api", tags=["predictions"])
    return app


app = create_app()
