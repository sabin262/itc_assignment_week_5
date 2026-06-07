from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import ValidationError, BaseModel, Field

from app.config import get_settings
from app.document_parser import DocumentParseError, extract_text_from_file
from app.schemas import (
    CompareRequest,
    CompareResponse,
    LeaseTextRequest,
    SummariseResponse,
    validate_lease_text,
)


app = FastAPI(
    title="Smart Lease Summariser",
    description="Grounded lease extraction and comparison API backed by Azure OpenAI.",
    version="1.0.0",
)


class SummaryRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to summarize.")


class SummaryResponse(BaseModel):
    status: str
    summary: str
    original_length: int


class CompareRequest(BaseModel):
    first_text: str = Field(..., min_length=1, description="First text value.")
    second_text: str = Field(..., min_length=1, description="Second text value.")


class CompareResponse(BaseModel):
    status: str
    are_equal: bool
    length_difference: int
    message: str


def endpoint_status(name: str) -> dict[str, str]:
    return {"endpoint": name, "status": "available"}


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "service": "fastapi",
        "status": "ok",
        "endpoints": {
            "summarize": "available",
            "compare": "available",
            "health": "available",
        },
    }


@app.get("/summarise")
def summarize_status() -> dict[str, str]:
    return endpoint_status("/summarize")


@app.post("/summarise", response_model=SummaryResponse)
def summarize(payload: SummaryRequest) -> SummaryResponse:
    cleaned = " ".join(payload.text.split())
    words = cleaned.split()
    summary = " ".join(words[:30])

    if len(words) > 30:
        summary = f"{summary}..."

    return SummaryResponse(
        status="ok",
        summary=summary,
        original_length=len(payload.text),
    )


@app.get("/compare")
def compare_status() -> dict[str, str]:
    return endpoint_status("/compare")


@app.post("/compare", response_model=CompareResponse)
def compare(payload: CompareRequest) -> CompareResponse:
    normalized_first = " ".join(payload.first_text.lower().split())
    normalized_second = " ".join(payload.second_text.lower().split())
    are_equal = normalized_first == normalized_second
    length_difference = abs(len(payload.first_text) - len(payload.second_text))

    return CompareResponse(
        status="ok",
        are_equal=are_equal,
        length_difference=length_difference,
        message="The submitted texts match." if are_equal else "The submitted texts differ.",
    )