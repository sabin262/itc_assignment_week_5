from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(
    title="Week 5 FastAPI Service",
    description="Backend service for Streamlit summarize, compare, and health pages.",
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


@app.get("/summarize")
def summarize_status() -> dict[str, str]:
    return endpoint_status("/summarize")


@app.post("/summarize", response_model=SummaryResponse)
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
