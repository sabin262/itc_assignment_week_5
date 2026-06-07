import json
from typing import Any


EXTRACTION_SYSTEM_PROMPT = """You are a lease information extraction engine.
Use only the lease text supplied by the user. Do not infer, guess, or add facts that
are not stated in the lease. Return valid JSON only, with no Markdown or commentary.
If a field is missing or unclear, return null for that field."""


GUARDRAIL_SYSTEM_PROMPT = """You verify whether a lease extraction is grounded in
the original lease text. Use only the original lease text. Return valid JSON only,
with no Markdown or commentary."""


COMPARE_SYSTEM_PROMPT = """You compare two residential lease extractions for a
non-legal audience. Describe practical differences in plain English, avoid legal
advice, and return valid JSON only with no Markdown or commentary."""


def build_extraction_prompt(lease_text: str) -> str:
    return f"""You are a Professional Solicitor focusing on commercial and residential property law. Extract the requested lease fields from the lease text.

Rules:
- Use the lease text as the only source of truth.
- Do not fabricate names, dates, amounts, obligations, restrictions, or clauses.
- Return null when a field is not explicitly present or is too ambiguous.
- Keep obligations and unusual clauses concise and plain-English.
- The one-paragraph summary must describe only what is supported by the lease text.

Return JSON with exactly these keys:
{{
  "tenant_name": string or null,
  "landlord_name": string or null,
  "property_address": string or null,
  "lease_start_date": string or null,
  "lease_end_date": string or null,
  "monthly_rent_amount": string or null,
  "rent_payment_due_date": string or null,
  "security_deposit_amount": string or null,
  "notice_period_to_vacate": string or null,
  "tenant_obligations": array of strings or null,
  "landlord_obligations": array of strings or null,
  "unusual_clauses": array of strings or null,
  "plain_english_summary": string or null
}}

Lease text:
{lease_text}"""


def build_guardrail_prompt(lease_text: str, extraction: dict[str, Any]) -> str:
    extraction_json = json.dumps(extraction, ensure_ascii=True, indent=2)
    return f"""Check whether each extracted field is supported by the original lease text.

Rules:
- Mark a field "supported" only when the extracted value is clearly present in the lease text.
- Mark a field "unsupported" when the extracted value is not present, contradicts the lease,
  or is more specific than the lease text supports.
- Mark a field "missing_from_extraction" when the extracted value is null.
- For supported fields, include a short exact evidence snippet from the lease text.
- For unsupported or missing fields, set evidence to null and explain the issue briefly.

Return JSON with this shape:
{{
  "overall_supported": boolean,
  "checks": [
    {{
      "field_name": string,
      "status": "supported" | "unsupported" | "missing_from_extraction",
      "extracted_value": any or null,
      "evidence": string or null,
      "explanation": string or null
    }}
  ]
}}

Extracted JSON:
{extraction_json}

Original lease text:
{lease_text}"""


def build_compare_prompt(lease_a: dict[str, Any], lease_b: dict[str, Any]) -> str:
    lease_a_json = json.dumps(lease_a, ensure_ascii=True, indent=2)
    lease_b_json = json.dumps(lease_b, ensure_ascii=True, indent=2)
    return f"""You are a Professional Solicitor focusing on commercial and residential property law. Compare these two lease extractions.

Rules:
- Focus on differences that matter to a tenant or property manager.
- Include differences in rent, dates, deposit, notice period, obligations,
  restrictions, unusual clauses, and overall tenant impact when present.
- Do not invent a difference if both values are the same or both are null.
- Avoid legal advice.

Return JSON with this shape:
{{
  "summary": string or null,
  "differences": [
    {{
      "field_name": string,
      "lease_a_value": any or null,
      "lease_b_value": any or null,
      "difference": string,
      "practical_impact": string or null
    }}
  ]
}}

Lease A extraction:
{lease_a_json}

Lease B extraction:
{lease_b_json}"""