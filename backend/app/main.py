from __future__ import annotations

import json
import pickle
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import v1_router
from app.config import settings
from app.core.domain import load_domain_catalog, load_user_whitelist
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
    if settings.model_path.exists() and settings.vectorizer_path.exists():
        with open(settings.model_path, "rb") as f:
            model = pickle.load(f)
        with open(settings.vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)

    whitelist = load_user_whitelist(settings.whitelist_path)
    trusted = load_domain_catalog(settings.trusted_domains_path)

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
    app = FastAPI(title="Spam Detector API", version="3.0.0", lifespan=lifespan)
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
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, log_level=settings.log_level)
