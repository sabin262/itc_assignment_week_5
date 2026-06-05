import os
from typing import Any

import requests


API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
TIMEOUT_SECONDS = 5
FILE_TIMEOUT_SECONDS = 30


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


def post_files_endpoint(path: str, uploaded_files: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    files = {
        field_name: (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
        for field_name, uploaded_file in uploaded_files.items()
    }

    try:
        response = requests.post(f"{API_URL}{path}", files=files, timeout=FILE_TIMEOUT_SECONDS)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as exc:
        message = str(exc)

        if exc.response is not None:
            try:
                return False, exc.response.json()
            except ValueError:
                message = exc.response.text or message

        return False, {"error": message}
