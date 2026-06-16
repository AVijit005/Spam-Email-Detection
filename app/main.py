from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.v1.router import v1_router
from app.config import settings
from app.core.domain import load_domain_catalog, load_user_whitelist
from app.ml.registry import load_model
from app.api.v1 import health as health_mod
from app.api.v1 import predict as predict_mod
from app.api.v1 import feedback as feedback_mod
from app.api.v1 import retrain as retrain_mod


def load_resources() -> None:
    metadata: dict = {}
    if settings.metadata_path.exists():
        with open(settings.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    model = None
    vectorizer = None
    artifact = load_model(settings.model_path, settings.vectorizer_path, settings.metadata_path)
    if artifact is not None:
        model = artifact.model
        vectorizer = artifact.vectorizer
        metadata = artifact.metadata

    whitelist = load_user_whitelist(settings.whitelist_path)
    trusted = load_domain_catalog(settings.trusted_domains_path)

    # --- Ensemble construction ---
    # Build EnsemblePredictor if transformer artifacts are available.
    # Falls back gracefully to XGBoost-only if loading fails.
    ensemble_info = metadata.get("ensemble_info")
    if settings.enable_transformer and ensemble_info:
        try:
            from app.ml.registry import load_transformer

            fusion_weight = ensemble_info.get("fusion_weight", 0.50)
            transformer_model_name = metadata.get("transformer_info", {}).get(
                "model_id", settings.transformer_model_name
            )

            result = load_transformer(
                model_path=settings.transformer_model_path,
                tokenizer_path=settings.transformer_tokenizer_path,
                model_name=transformer_model_name,
                device=settings.transformer_device,
                cache_dir=str(settings.transformer_cache_dir) if settings.transformer_cache_dir else None,
            )

            if result is not None:
                transformer_model, transformer_tokenizer = result
                from app.ml.ensemble import EnsemblePredictor

                ensemble = EnsemblePredictor(
                    classical_model=model,
                    classical_vectorizer_bundle=vectorizer,
                    transformer_model=transformer_model,
                    transformer_tokenizer=transformer_tokenizer,
                    fusion_weight=fusion_weight,
                    transformer_device=settings.transformer_device,
                )
                model = ensemble
                logger.info(
                    "Ensemble Predictor active: XGBoost + %s (w=%.2f) | F1=%.4f",
                    transformer_model_name, fusion_weight,
                    metadata.get("selected_metrics", {}).get("ensemble_f1", 0),
                )
            else:
                logger.warning(
                    "Transformer model not found — running XGBoost-only. "
                    "Expected ensemble F1: %.4f.",
                    metadata.get("selected_metrics", {}).get("ensemble_f1", 0),
                )
        except Exception:
            logger.exception(
                "Transformer loading failed — falling back to XGBoost-only."
            )

    if model is None or vectorizer is None:
        logger.warning("Model or vectorizer not found at startup. Predict will return 500.")

    health_mod.model = model
    health_mod.vectorizer = vectorizer
    health_mod.user_whitelist_domains = whitelist
    health_mod.trusted_domain_catalog = trusted
    health_mod.model_metadata = metadata

    predict_mod.model = model
    predict_mod.vectorizer = vectorizer
    predict_mod.user_whitelist_domains = whitelist
    predict_mod.trusted_domain_catalog = trusted
    predict_mod.model_metadata = metadata

    feedback_mod.model_metadata = metadata
    retrain_mod.model_metadata = metadata


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_resources()
    yield


def create_app() -> FastAPI:
    limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
    app = FastAPI(title="Spam Detector API", version="3.0.0", lifespan=lifespan)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_origin_regex=settings.allow_origin_regex,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(v1_router, prefix="/v1")
    return app


app = create_app()


if __name__ == "__main__":
    import signal
    import uvicorn

    def _shutdown_handler(signum, frame):
        logger.info("Received shutdown signal %s, stopping...", signum)

    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    uvicorn.run(app, host=settings.api_host, port=settings.api_port, log_level=settings.log_level)
