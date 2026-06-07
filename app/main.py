from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import ValidationError

from app.config import get_settings
from app.document_parser import DocumentParseError, extract_text_from_file
from app.llm_client import AzureLeaseLLMClient, LLMResponseError
from app.schemas import (
    CompareRequest,
    CompareResponse,
    LeaseTextRequest,
    SummariseResponse,
    validate_lease_text,
)
from app.service import LeaseSummariserService


app = FastAPI(
    title="Smart Lease Summariser",
    description="Grounded lease extraction and comparison API backed by Azure OpenAI.",
    version="1.0.0",
)


def create_lease_service() -> LeaseSummariserService:
    settings = get_settings()
    return LeaseSummariserService(AzureLeaseLLMClient(settings))


def get_lease_service_factory() -> Callable[[], LeaseSummariserService]:
    return create_lease_service


LeaseServiceFactoryDependency = Annotated[
    Callable[[], LeaseSummariserService],
    Depends(get_lease_service_factory),
]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/summarise-text", response_model=SummariseResponse)
def summarise_lease_text(
    request: LeaseTextRequest,
    service_factory: LeaseServiceFactoryDependency,
) -> SummariseResponse:
    return _summarise_text(request.lease_text, service_factory)


@app.post("/summarise", response_model=SummariseResponse)
async def summarise_lease_document(
    service_factory: LeaseServiceFactoryDependency,
    file: UploadFile = File(...),
) -> SummariseResponse:
    lease_text = await _extract_upload_text(file)
    return _summarise_text(lease_text, service_factory)


@app.post("/compare-text", response_model=CompareResponse)
def compare_lease_texts(
    request: CompareRequest,
    service_factory: LeaseServiceFactoryDependency,
) -> CompareResponse:
    return _compare_text(request.lease_a, request.lease_b, service_factory)


@app.post("/compare", response_model=CompareResponse)
async def compare_lease_documents(
    service_factory: LeaseServiceFactoryDependency,
    lease_a: UploadFile = File(...),
    lease_b: UploadFile = File(...),
) -> CompareResponse:
    lease_a_text = await _extract_upload_text(lease_a)
    lease_b_text = await _extract_upload_text(lease_b)
    return _compare_text(lease_a_text, lease_b_text, service_factory)


def _summarise_text(
    lease_text: str,
    service_factory: Callable[[], LeaseSummariserService],
) -> SummariseResponse:
    try:
        validated_text = validate_lease_text(lease_text)
        service = service_factory()
        return service.summarise(validated_text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(
            status_code=500,
            detail="Azure OpenAI environment configuration is missing or invalid.",
        ) from exc
    except LLMResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc



def _compare_text(
    lease_a_text: str,
    lease_b_text: str,
    service_factory: Callable[[], LeaseSummariserService],
) -> CompareResponse:
    try:
        validated_lease_a = validate_lease_text(lease_a_text)
        validated_lease_b = validate_lease_text(lease_b_text)
        service = service_factory()
        return service.compare(validated_lease_a, validated_lease_b)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(
            status_code=500,
            detail="Azure OpenAI environment configuration is missing or invalid.",
        ) from exc
    except LLMResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


async def _extract_upload_text(file: UploadFile) -> str:
    content = await file.read()
    try:
        return extract_text_from_file(file.filename or "", content)
    except DocumentParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
