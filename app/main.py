from collections.abc import Callable
from datetime import UTC, datetime
import threading
from uuid import uuid4
from typing import Annotated
from app.langfuse_client import get_langfuse_client

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import ValidationError

from app.config import get_rag_settings, get_s3_settings, get_settings
from app.document_parser import DocumentParseError, extract_text_from_file
from app.llm_client import AzureLeaseLLMClient, LLMResponseError
from app.schemas import (
    CompareS3Request,
    CompareRequest,
    CompareResponse,
    LeaseTextRequest,
    RAGChatRequest,
    RAGChatResponse,
    RAGIndexJobStatus,
    RAGIndexResponse,
    RAGSearchRequest,
    RAGSearchResponse,
    RAGStatusResponse,
    S3LeaseFile,
    S3LeaseRequest,
    SummariseResponse,
    validate_lease_text,
)
from app.rag_service import (
    RAGConfigurationError,
    RAGError,
    RAGInvalidKeyError,
    RAGLeaseNotIndexedError,
    RAGService,
    create_rag_service as build_rag_service,
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


_INDEX_JOB_LOCK = threading.Lock()
_INDEX_JOB_STATE = RAGIndexJobStatus(status="idle")


def create_lease_service() -> LeaseSummariserService:
    settings = get_settings()
    return LeaseSummariserService(
        AzureLeaseLLMClient(settings),
        langfuse=get_langfuse_client(),
    )


def create_s3_storage() -> S3LeaseStorage:
    settings = get_s3_settings()
    return S3LeaseStorage(
        bucket=settings.s3_bucket_name,
        prefix=settings.s3_prefix,
    )


def create_rag_service() -> RAGService:
    return build_rag_service(
        settings=get_settings(),
        rag_settings=get_rag_settings(),
        s3_settings=get_s3_settings(),
        langfuse=get_langfuse_client(),
    )


def get_lease_service_factory() -> Callable[[], LeaseSummariserService]:
    return create_lease_service


def get_s3_storage_factory() -> Callable[[], S3LeaseStorage]:
    return create_s3_storage


def get_rag_service_factory() -> Callable[[], RAGService]:
    return create_rag_service


LeaseServiceFactoryDependency = Annotated[
    Callable[[], LeaseSummariserService],
    Depends(get_lease_service_factory),
]

S3StorageFactoryDependency = Annotated[
    Callable[[], S3LeaseStorage],
    Depends(get_s3_storage_factory),
]

RAGServiceFactoryDependency = Annotated[
    Callable[[], RAGService],
    Depends(get_rag_service_factory),
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


@app.post("/summarise-indexed", response_model=SummariseResponse)
def summarise_indexed_lease(
    request: S3LeaseRequest,
    service_factory: LeaseServiceFactoryDependency,
    rag_service_factory: RAGServiceFactoryDependency,
) -> SummariseResponse:
    lease_text = _extract_indexed_text(request.key, rag_service_factory)
    return _summarise_text(lease_text, service_factory)


@app.post("/compare-indexed", response_model=CompareResponse)
def compare_indexed_leases(
    request: CompareS3Request,
    service_factory: LeaseServiceFactoryDependency,
    rag_service_factory: RAGServiceFactoryDependency,
) -> CompareResponse:
    lease_a_text = _extract_indexed_text(request.lease_a_key, rag_service_factory)
    lease_b_text = _extract_indexed_text(request.lease_b_key, rag_service_factory)
    return _compare_text(lease_a_text, lease_b_text, service_factory)


@app.get("/rag/status", response_model=RAGStatusResponse)
def rag_status(
    rag_service_factory: RAGServiceFactoryDependency,
) -> RAGStatusResponse:
    try:
        return rag_service_factory().status()
    except (ValidationError, RAGConfigurationError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RAGError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/rag/index", response_model=RAGIndexJobStatus)
def index_s3_leases_for_rag(
    background_tasks: BackgroundTasks,
    rag_service_factory: RAGServiceFactoryDependency,
    s3_storage_factory: S3StorageFactoryDependency,
    service_factory: LeaseServiceFactoryDependency,
) -> RAGIndexJobStatus:
    with _INDEX_JOB_LOCK:
        if _INDEX_JOB_STATE.status == "running":
            return _INDEX_JOB_STATE.model_copy(deep=True)

        job_id = str(uuid4())
        _set_index_job_state_unlocked(
            RAGIndexJobStatus(
                job_id=job_id,
                status="running",
                started_at=datetime.now(UTC).isoformat(),
                finished_at=None,
                result=None,
                error=None,
                progress_current=0,
                progress_total=0,
                progress_percent=0.0,
                message="Starting S3 lease indexing.",
                current_key=None,
            )
        )

    background_tasks.add_task(
        _run_index_job,
        job_id,
        rag_service_factory,
        s3_storage_factory,
        service_factory,
    )
    return _get_index_job_state()


@app.get("/rag/index/status", response_model=RAGIndexJobStatus)
def rag_index_status() -> RAGIndexJobStatus:
    return _get_index_job_state()


@app.post("/rag/search", response_model=RAGSearchResponse)
def search_rag_leases(
    request: RAGSearchRequest,
    rag_service_factory: RAGServiceFactoryDependency,
) -> RAGSearchResponse:
    try:
        return rag_service_factory().search(request.question, request.top_k)
    except (ValidationError, RAGConfigurationError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RAGError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/rag/chat", response_model=RAGChatResponse)
def chat_with_rag_leases(
    request: RAGChatRequest,
    rag_service_factory: RAGServiceFactoryDependency,
) -> RAGChatResponse:
    try:
        return rag_service_factory().chat(
            question=request.question,
            lease_keys=request.lease_keys,
            history=request.history,
            top_k=request.top_k,
        )
    except (ValidationError, RAGConfigurationError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except (RAGError, LLMResponseError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _run_index_job(
    job_id: str,
    rag_service_factory: Callable[[], RAGService],
    s3_storage_factory: Callable[[], S3LeaseStorage],
    service_factory: Callable[[], LeaseSummariserService],
) -> None:
    try:
        result = rag_service_factory().index_s3_leases(
            s3_storage_factory(),
            service_factory(),
            progress_callback=lambda current, total, message, current_key: (
                _update_index_job_progress(
                    job_id,
                    current,
                    total,
                    message,
                    current_key,
                )
            ),
        )
    except Exception as exc:
        with _INDEX_JOB_LOCK:
            if _INDEX_JOB_STATE.job_id != job_id:
                return
            _set_index_job_state_unlocked(
                _INDEX_JOB_STATE.model_copy(
                    update={
                        "status": "failed",
                        "finished_at": datetime.now(UTC).isoformat(),
                        "error": str(exc),
                        "message": "Indexing failed.",
                        "current_key": None,
                    },
                    deep=True,
                )
            )
        return

    with _INDEX_JOB_LOCK:
        if _INDEX_JOB_STATE.job_id != job_id:
            return
        _set_index_job_state_unlocked(
            _INDEX_JOB_STATE.model_copy(
                update={
                    "status": "completed",
                    "finished_at": datetime.now(UTC).isoformat(),
                    "result": result,
                    "error": None,
                    "progress_current": _INDEX_JOB_STATE.progress_total,
                    "progress_percent": 1.0,
                    "message": "Indexing completed.",
                    "current_key": None,
                },
                deep=True,
            )
        )


def _update_index_job_progress(
    job_id: str,
    current: int,
    total: int,
    message: str,
    current_key: str | None,
) -> None:
    progress_total = max(total, 0)
    progress_current = max(min(current, progress_total), 0)
    progress_percent = (
        progress_current / progress_total
        if progress_total
        else 0.0
    )
    with _INDEX_JOB_LOCK:
        if _INDEX_JOB_STATE.job_id != job_id or _INDEX_JOB_STATE.status != "running":
            return
        _set_index_job_state_unlocked(
            _INDEX_JOB_STATE.model_copy(
                update={
                    "progress_current": progress_current,
                    "progress_total": progress_total,
                    "progress_percent": progress_percent,
                    "message": message,
                    "current_key": current_key,
                },
                deep=True,
            )
        )


def _get_index_job_state() -> RAGIndexJobStatus:
    with _INDEX_JOB_LOCK:
        return _INDEX_JOB_STATE.model_copy(deep=True)


def _set_index_job_state_unlocked(state: RAGIndexJobStatus) -> None:
    global _INDEX_JOB_STATE
    _INDEX_JOB_STATE = state


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


def _extract_indexed_text(
    key: str,
    rag_service_factory: Callable[[], RAGService],
) -> str:
    try:
        return rag_service_factory().lease_text_from_index(key)
    except RAGInvalidKeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RAGLeaseNotIndexedError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValidationError, RAGConfigurationError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RAGError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


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
