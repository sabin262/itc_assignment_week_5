import json
from typing import Any


class JSONParseError(ValueError):
    """Raised when an LLM response cannot be parsed as a JSON object."""


def parse_json_object(raw_content: str) -> dict[str, Any]:
    content = raw_content.strip()
    if content.startswith("```"):
        content = _strip_markdown_fence(content)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = _parse_embedded_json(content)

    if not isinstance(parsed, dict):
        raise JSONParseError("LLM response must be a JSON object.")
    return parsed


def _strip_markdown_fence(content: str) -> str:
    lines = content.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_embedded_json(content: str) -> Any:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise JSONParseError("LLM response did not contain a JSON object.")

    try:
        return json.loads(content[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JSONParseError("LLM response did not contain valid JSON.") from exc

