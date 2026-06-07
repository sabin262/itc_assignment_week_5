from typing import Any

import httpx


def extract_error_message(response: httpx.Response) -> str:
    try:
        payload: Any = response.json()
    except ValueError:
        return response.text or "Unknown API error."

    if isinstance(payload, dict):
        detail = payload.get("detail", payload)
    else:
        detail = payload

    if isinstance(detail, str):
        return detail

    if isinstance(detail, list):
        messages = [_format_error_item(item) for item in detail]
        return "\n".join(message for message in messages if message)

    return str(detail)


def _format_error_item(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item)

    message = item.get("msg") or item.get("message") or str(item)
    location = item.get("loc")
    if not location:
        return str(message)

    location_text = ".".join(str(part) for part in location)
    return f"{location_text}: {message}"
