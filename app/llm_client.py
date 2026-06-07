from typing import Any

from openai import AzureOpenAI
from pydantic import ValidationError

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

    def extract(self, lease_text: str) -> LeaseExtraction:
        payload = self._chat_json(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=build_extraction_prompt(lease_text),
        )
        return self._validate(payload, LeaseExtraction)

    def verify(self, lease_text: str, extraction: LeaseExtraction) -> GuardrailResult:
        payload = self._chat_json(
            system_prompt=GUARDRAIL_SYSTEM_PROMPT,
            user_prompt=build_guardrail_prompt(
                lease_text,
                extraction.model_dump(mode="json"),
            ),
        )
        return self._validate(payload, GuardrailResult)

    def compare(
        self,
        lease_a_extraction: LeaseExtraction,
        lease_b_extraction: LeaseExtraction,
    ) -> LeaseComparison:
        payload = self._chat_json(
            system_prompt=COMPARE_SYSTEM_PROMPT,
            user_prompt=build_compare_prompt(
                lease_a_extraction.model_dump(mode="json"),
                lease_b_extraction.model_dump(mode="json"),
            ),
        )
        return self._validate(payload, LeaseComparison)

    def _chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # Lease extraction is a deterministic information-retrieval task, so
            # temperature stays at 0.0 to minimise creative wording and guessing.
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
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