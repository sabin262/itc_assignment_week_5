from copy import deepcopy
from typing import Any

from openai import AzureOpenAI
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.json_utils import JSONParseError, parse_json_object
from app.prompts import (
    COMPARE_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    GUARDRAIL_SYSTEM_PROMPT,
    build_compare_prompt,
    build_extraction_prompt,
    build_guardrail_prompt,
)
from app.schemas import GuardrailResult, LeaseComparison, LeaseExtraction


class LLMResponseError(RuntimeError):
    """Raised when the model response cannot be parsed or validated."""


class AzureLeaseLLMClient:
    def __init__(self, settings: Settings):
        self._client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
        self._deployment = settings.azure_openai_deployment

    def extract(self, lease_text: str, trace:Any | None= None) -> LeaseExtraction:
        payload = self._chat_json(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=build_extraction_prompt(lease_text),
            response_model=LeaseExtraction,
            operation_name = "extract",
            trace = trace
        )
        return self._validate(payload, LeaseExtraction)

    def verify(self, lease_text: str, extraction: LeaseExtraction, trace:Any | None= None) -> GuardrailResult:
        payload = self._chat_json(
            system_prompt=GUARDRAIL_SYSTEM_PROMPT,
            user_prompt=build_guardrail_prompt(
                lease_text,
                extraction.model_dump(mode="json"),
            ),
            response_model=GuardrailResult,
            operation_name = "verify",
            trace = trace
        )
        return self._validate(payload, GuardrailResult)

    def compare(
        self,
        lease_a_extraction: LeaseExtraction,
        lease_b_extraction: LeaseExtraction,
        trace:Any | None= None
    ) -> LeaseComparison:
        payload = self._chat_json(
            system_prompt=COMPARE_SYSTEM_PROMPT,
            user_prompt=build_compare_prompt(
                lease_a_extraction.model_dump(mode="json"),
                lease_b_extraction.model_dump(mode="json"),
            ),
            response_model=LeaseComparison,
            operation_name = "compare",
            trace = trace
        )
        return self._validate(payload, LeaseComparison)

    def _chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        operation_name: str,
        trace: Any | None = None
    ) -> dict[str, Any]:
        messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

        generation = None
        if trace is not None:
            generation = trace.generation(
                name=operation_name,
                model=self._deployment,
                input=messages,
            )

        try:
            response = self._client.chat.completions.create(
                model=self._deployment,
                messages=messages,
                temperature=0.0,
                response_format=_build_json_schema_response_format(response_model),
            )
        except Exception:
            if generation is not None:
                generation.end(level="ERROR")
            raise

        content = response.choices[0].message.content

        if generation is not None:
            usage = response.usage
            generation.end(
                output=content or "",
                usage={
                    "input": usage.prompt_tokens if usage else 0,
                    "output": usage.completion_tokens if usage else 0,
                },
            )

        if not content:
            raise LLMResponseError("Azure OpenAI returned an empty response.")

        try:
            return parse_json_object(content)
        except JSONParseError as exc:
            raise LLMResponseError(str(exc)) from exc

    @staticmethod
    def _validate(payload: dict[str, Any], model_type: type[Any]) -> Any:
        try:
            return model_type.model_validate(payload)
        except ValidationError as exc:
            raise LLMResponseError("Azure OpenAI returned JSON with an unexpected shape.") from exc


def _build_json_schema_response_format(model_type: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model_type.__name__,
            "strict": True,
            "schema": _make_openai_strict_schema(model_type.model_json_schema()),
        },
    }


def _make_openai_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    strict_schema = deepcopy(schema)
    _require_all_object_properties(strict_schema)
    return strict_schema


def _require_all_object_properties(schema_node: Any) -> None:
    if isinstance(schema_node, dict):
        schema_node.pop("default", None)

        properties = schema_node.get("properties")
        if isinstance(properties, dict):
            schema_node["additionalProperties"] = False
            schema_node["required"] = list(properties.keys())

        for value in list(schema_node.values()):
            _require_all_object_properties(value)
    elif isinstance(schema_node, list):
        for item in schema_node:
            _require_all_object_properties(item)
