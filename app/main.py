"""
FastAPI application entrypoint for ShieldML.
"""

from __future__ import annotations

import contextlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.security import RequestSizeLimitMiddleware
from app.services.model_loader import model_loader

setup_logging(settings.log_level)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model artifacts
    model_loader.load(
        model_path=settings.model_artifact_path,
        pipeline_path=settings.pipeline_artifact_path,
        metadata_path=settings.metadata_path,
    )
    yield
    # Shutdown logic if needed


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Real-time financial fraud detection scoring API with SHAP explainability.",
    lifespan=lifespan,
)

# CORS configuration
origins = [o.strip() for o in settings.cors_origins.split(",")] if settings.cors_origins != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Size limit middleware
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_size_bytes=settings.max_request_size_bytes,
)

# Include routes
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
