import re
from datetime import datetime
from enum import Enum
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field, field_validator


MIN_LEASE_WORDS = 100
MIN_LEASE_WORDS_MESSAGE = "Lease text must contain at least 100 words."


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def validate_lease_text(text: str) -> str:
    cleaned = text.strip()
    if count_words(cleaned) < MIN_LEASE_WORDS:
        raise ValueError(MIN_LEASE_WORDS_MESSAGE)
    return cleaned


class LeaseTextRequest(BaseModel):
    lease_text: str = Field(..., description="Raw lease text to summarise.")

    @field_validator("lease_text")
    @classmethod
    def validate_minimum_words(cls, value: str) -> str:
        return validate_lease_text(value)


class CompareRequest(BaseModel):
    lease_a: str = Field(..., description="Raw text for the first lease.")
    lease_b: str = Field(..., description="Raw text for the second lease.")

    @field_validator("lease_a", "lease_b")
    @classmethod
    def validate_minimum_words(cls, value: str) -> str:
        return validate_lease_text(value)


def validate_s3_key(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("S3 key is required.")
    return cleaned


class S3LeaseRequest(BaseModel):
    key: str = Field(..., description="S3 object key for the lease file.")

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        return validate_s3_key(value)


class CompareS3Request(BaseModel):
    lease_a_key: str = Field(..., description="S3 object key for the first lease.")
    lease_b_key: str = Field(..., description="S3 object key for the second lease.")

    @field_validator("lease_a_key", "lease_b_key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        return validate_s3_key(value)


class S3LeaseFile(BaseModel):
    key: str
    filename: str
    size: int
    last_modified: datetime | None = None


class LeaseExtraction(BaseModel):
    tenant_name: str | None = None
    landlord_name: str | None = None
    property_address: str | None = None
    lease_start_date: str | None = None
    lease_end_date: str | None = None
    monthly_rent_amount: str | None = None
    rent_payment_due_date: str | None = None
    security_deposit_amount: str | None = None
    notice_period_to_vacate: str | None = None
    tenant_obligations: list[str] | None = None
    landlord_obligations: list[str] | None = None
    unusual_clauses: list[str] | None = None
    plain_english_summary: str | None = None


LeaseFieldValue: TypeAlias = str | list[str] | None


class VerificationStatus(str, Enum):
    supported = "supported"
    unsupported = "unsupported"
    missing_from_extraction = "missing_from_extraction"


class VerificationItem(BaseModel):
    field_name: str
    status: VerificationStatus
    extracted_value: LeaseFieldValue = None
    evidence: str | None = None
    explanation: str | None = None


class GuardrailResult(BaseModel):
    overall_supported: bool
    checks: list[VerificationItem]


class SummariseResponse(BaseModel):
    extraction: LeaseExtraction
    verification: GuardrailResult
    warnings: list[str] = Field(default_factory=list)


class LeaseDifference(BaseModel):
    field_name: str
    lease_a_value: LeaseFieldValue = None
    lease_b_value: LeaseFieldValue = None
    difference: str
    practical_impact: str | None = None


class LeaseComparison(BaseModel):
    summary: str | None = None
    differences: list[LeaseDifference]


class CompareResponse(BaseModel):
    lease_a: SummariseResponse
    lease_b: SummariseResponse
    comparison: LeaseComparison


class ErrorResponse(BaseModel):
    detail: str | list[dict[str, Any]]


LLMResponseFormat = Literal["json_schema"]

