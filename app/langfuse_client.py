from functools import lru_cache
from typing import Any

from app.config import get_langfuse_settings


@lru_cache
def get_langfuse_client() -> Any | None:
    settings = get_langfuse_settings()
    if not (settings.langfuse_secret_key and settings.langfuse_public_key):
        return None
    try:
        from langfuse import Langfuse
        kwargs: dict[str, str] = {
            "secret_key": settings.langfuse_secret_key,
            "public_key": settings.langfuse_public_key,
        }
        if settings.langfuse_base_url:
            kwargs["host"] = settings.langfuse_base_url
        return Langfuse(**kwargs)
    except (ImportError, Exception):
        return None
