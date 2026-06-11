import json
from types import SimpleNamespace
from typing import Any

from app.config import Settings
from app.llm_client import AzureLeaseLLMClient, _build_json_schema_response_format
from app.schemas import RAGChatAnswer


def test_llm_calls_use_matching_structured_response_schemas(monkeypatch):
    calls: list[dict[str, Any]] = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            schema_name = kwargs["response_format"]["json_schema"]["name"]
            payloads = {
                "LeaseExtraction": {
                    "tenant_name": "Alex Rivera",
                    "landlord_name": None,
                    "property_address": "12 Garden Street, London",
                    "lease_start_date": "1 January 2026",
                    "lease_end_date": "31 December 2026",
                    "monthly_rent_amount": "1,500 pounds",
                    "rent_payment_due_date": "first day of each month",
                    "security_deposit_amount": "1,500 pounds",
                    "notice_period_to_vacate": "two months",
                    "tenant_obligations": ["Keep the premises clean."],
                    "landlord_obligations": ["Give notice before entering."],
                    "unusual_clauses": None,
                    "plain_english_summary": "Alex rents the property for one year.",
                },
                "GuardrailResult": {
                    "overall_supported": True,
                    "checks": [
                        {
                            "field_name": "tenant_name",
                            "status": "supported",
                            "extracted_value": "Alex Rivera",
                            "evidence": "tenant Alex Rivera",
                            "explanation": None,
                        }
                    ],
                },
                "LeaseComparison": {
                    "summary": "Lease B has a higher rent.",
                    "differences": [
                        {
                            "field_name": "monthly_rent_amount",
                            "lease_a_value": "1,500 pounds",
                            "lease_b_value": "1,700 pounds",
                            "difference": "Lease B costs 200 pounds more per month.",
                            "practical_impact": "Lease A is cheaper month to month.",
                        }
                    ],
                },
            }
            content = json.dumps(payloads[schema_name])
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    class FakeAzureOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("app.llm_client.AzureOpenAI", FakeAzureOpenAI)

    settings = Settings(
        AZURE_OPENAI_API_KEY="test-key",
        AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com",
        AZURE_OPENAI_API_VERSION="2024-08-01-preview",
        AZURE_OPENAI_DEPLOYMENT="test-deployment",
    )
    client = AzureLeaseLLMClient(settings)

    extraction = client.extract("lease text")
    client.verify("lease text", extraction)
    client.compare(extraction, extraction)

    assert [call["response_format"]["json_schema"]["name"] for call in calls] == [
        "LeaseExtraction",
        "GuardrailResult",
        "LeaseComparison",
    ]

    for call in calls:
        response_format = call["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        _assert_openai_strict_schema(response_format["json_schema"]["schema"])


def test_rag_chat_answer_uses_strict_structured_response_schema():
    response_format = _build_json_schema_response_format(RAGChatAnswer)

    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    _assert_openai_strict_schema(response_format["json_schema"]["schema"])


def _assert_openai_strict_schema(schema_node: Any) -> None:
    if isinstance(schema_node, dict):
        assert "default" not in schema_node

        properties = schema_node.get("properties")
        if isinstance(properties, dict):
            assert schema_node["additionalProperties"] is False
            assert set(schema_node["required"]) == set(properties)

        for value in schema_node.values():
            _assert_openai_strict_schema(value)
    elif isinstance(schema_node, list):
        for item in schema_node:
            _assert_openai_strict_schema(item)
