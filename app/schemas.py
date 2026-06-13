import re
from datetime import datetime
from enum import Enum
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field, field_validator, model_validator


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


def validate_chat_session_id(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Chat session id is required.")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", cleaned):
        raise ValueError(
            "Chat session id may only contain letters, numbers, underscores, and hyphens."
        )
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


def validate_question(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Question is required.")
    return cleaned


class FileCollectionInfo(BaseModel):
    s3_key: str
    filename: str
    collection_name: str
    chunk_count: int
    indexed_at: str | None = None


class RAGStatusResponse(BaseModel):
    collection_name: str
    indexed_lease_count: int
    chunk_count: int
    last_indexed_at: str | None = None
    indexed_summary_count: int = 0
    file_collections: list[FileCollectionInfo] = Field(default_factory=list)


class UploadAndIndexResponse(BaseModel):
    s3_key: str
    filename: str
    collection_name: str
    chunk_count: int
    word_count: int
    summarised: bool = False


class RAGIndexResponse(BaseModel):
    indexed_lease_count: int
    indexed_chunk_count: int
    skipped_files: list[str] = Field(default_factory=list)
    failed_files: list[str] = Field(default_factory=list)
    summarised_lease_count: int = 0
    summary_failed_files: list[str] = Field(default_factory=list)


class RAGIndexJobStatus(BaseModel):
    job_id: str | None = None
    status: Literal["idle", "running", "completed", "failed"]
    started_at: str | None = None
    finished_at: str | None = None
    result: RAGIndexResponse | None = None
    error: str | None = None
    progress_current: int = 0
    progress_total: int = 0
    progress_percent: float = 0.0
    message: str | None = None
    current_key: str | None = None


class RAGSearchRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def validate_search_question(cls, value: str) -> str:
        return validate_question(value)


class RAGSearchMatch(BaseModel):
    key: str
    filename: str
    snippet: str
    score: float | None = None
    chunk_index: int


class RAGSearchResponse(BaseModel):
    question: str
    matches: list[RAGSearchMatch]


class RAGChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        return validate_question(value)


class RAGChatRequest(BaseModel):
    question: str
    lease_keys: list[str] = Field(default_factory=list)
    history: list[RAGChatMessage] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: str | None = None

    @field_validator("question")
    @classmethod
    def validate_chat_question(cls, value: str) -> str:
        return validate_question(value)

    @field_validator("lease_keys")
    @classmethod
    def validate_lease_keys(cls, value: list[str]) -> list[str]:
        return [validate_s3_key(key) for key in value]

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_chat_session_id(value)

    @model_validator(mode="after")
    def limit_history(self) -> "RAGChatRequest":
        self.history = self.history[-10:]
        return self


class RAGCitation(BaseModel):
    key: str
    filename: str
    snippet: str
    chunk_index: int
    source_type: Literal["chunk", "summary"] = "chunk"


class RAGChatAnswer(BaseModel):
    answer: str
    citations: list[RAGCitation] = Field(default_factory=list)


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


class RAGEvalResult(BaseModel):
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None


class RAGEvalRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str] = Field(default_factory=list)

    @field_validator("question", "answer")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        return validate_question(value)


class RAGChatResponse(BaseModel):
    question: str
    answer: str
    citations: list[RAGCitation]
    verification: GuardrailResult | None = None
    warnings: list[str] = Field(default_factory=list)
<<<<<<< HEAD
    eval: RAGEvalResult | None = None
=======
    session_id: str | None = None
    saved_at: str | None = None


class RAGChatStoredMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    citations: list[RAGCitation] = Field(default_factory=list)
    verification: GuardrailResult | None = None
    warnings: list[str] = Field(default_factory=list)
    created_at: str | None = None


class RAGChatSessionSummary(BaseModel):
    session_id: str
    title: str
    lease_keys: list[str] = Field(default_factory=list)
    message_count: int = 0
    created_at: str
    updated_at: str


class RAGChatSessionListResponse(BaseModel):
    sessions: list[RAGChatSessionSummary]


class RAGChatSessionResponse(RAGChatSessionSummary):
    messages: list[RAGChatStoredMessage] = Field(default_factory=list)
>>>>>>> b217404d5777596d29b33f9e5cbae81b9b326d47


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

