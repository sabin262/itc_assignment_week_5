import os
from typing import Any

import requests


API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
TIMEOUT_SECONDS = 5


def get_endpoint(path: str) -> tuple[bool, dict[str, Any]]:
    try:
        response = requests.get(f"{API_URL}{path}", timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as exc:
        return False, {"error": str(exc)}


def post_endpoint(path: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    try:
        response = requests.post(f"{API_URL}{path}", json=payload, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as exc:
        return False, {"error": str(exc)}
