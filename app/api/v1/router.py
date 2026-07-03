from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.health import router as health_router
from app.api.v1.predict import router as predict_router
from app.api.v1.feedback import router as feedback_router
from app.api.v1.retrain import router as retrain_router

v1_router = APIRouter()
v1_router.include_router(health_router, tags=["health"])
v1_router.include_router(predict_router, tags=["prediction"])
v1_router.include_router(feedback_router, tags=["feedback"])
v1_router.include_router(retrain_router, tags=["retrain"])
