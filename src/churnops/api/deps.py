"""FastAPI dependency: retrieve the loaded model from app.state."""

from __future__ import annotations

from fastapi import HTTPException, Request, status


def get_model_state(request: Request):
    """Return the ModelState stored on app.state.

    Raises HTTP 503 if the model failed to load at startup so that every
    prediction endpoint responds with a clean error rather than an AttributeError.
    """
    state = getattr(request.app.state, "model_state", None)
    if state is None or not state.get("model_loaded"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Check /health for details.",
        )
    return state
