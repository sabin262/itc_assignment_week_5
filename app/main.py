from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import ValidationError

from app.config import get_s3_settings, get_settings
from app.document_parser import DocumentParseError, extract_text_from_file
from app.llm_client import AzureLeaseLLMClient, LLMResponseError
from app.schemas import (
    CompareS3Request,
    CompareRequest,
    CompareResponse,
    LeaseTextRequest,
    S3LeaseFile,
    S3LeaseRequest,
    SummariseResponse,
    validate_lease_text,
)
from app.s3_storage import (
    S3ConfigurationError,
    S3InvalidKeyError,
    S3LeaseStorage,
    S3ObjectNotFoundError,
    S3StorageError,
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


def create_s3_storage() -> S3LeaseStorage:
    settings = get_s3_settings()
    return S3LeaseStorage(
        bucket=settings.s3_bucket_name,
        prefix=settings.s3_prefix,
    )


def get_lease_service_factory() -> Callable[[], LeaseSummariserService]:
    return create_lease_service


def get_s3_storage_factory() -> Callable[[], S3LeaseStorage]:
    return create_s3_storage


LeaseServiceFactoryDependency = Annotated[
    Callable[[], LeaseSummariserService],
    Depends(get_lease_service_factory),
]

S3StorageFactoryDependency = Annotated[
    Callable[[], S3LeaseStorage],
    Depends(get_s3_storage_factory),
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


@app.get("/s3/leases", response_model=list[S3LeaseFile])
def list_s3_leases(
    s3_storage_factory: S3StorageFactoryDependency,
) -> list[S3LeaseFile]:
    try:
        return s3_storage_factory().list_lease_files()
    except S3ConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except S3StorageError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/summarise-s3", response_model=SummariseResponse)
def summarise_s3_lease(
    request: S3LeaseRequest,
    service_factory: LeaseServiceFactoryDependency,
    s3_storage_factory: S3StorageFactoryDependency,
) -> SummariseResponse:
    lease_text = _extract_s3_text(request.key, s3_storage_factory)
    return _summarise_text(lease_text, service_factory)


@app.post("/compare-s3", response_model=CompareResponse)
def compare_s3_leases(
    request: CompareS3Request,
    service_factory: LeaseServiceFactoryDependency,
    s3_storage_factory: S3StorageFactoryDependency,
) -> CompareResponse:
    lease_a_text = _extract_s3_text(request.lease_a_key, s3_storage_factory)
    lease_b_text = _extract_s3_text(request.lease_b_key, s3_storage_factory)
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


def _extract_s3_text(
    key: str,
    s3_storage_factory: Callable[[], S3LeaseStorage],
) -> str:
    try:
        filename, content = s3_storage_factory().get_file(key)
        return extract_text_from_file(filename, content)
    except S3InvalidKeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except S3ObjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except S3ConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except S3StorageError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except DocumentParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
