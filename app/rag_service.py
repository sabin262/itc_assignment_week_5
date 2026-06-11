from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any

from openai import AzureOpenAI
from pydantic import ValidationError

from app.config import RAGSettings, S3Settings, Settings
from app.document_parser import (
    SUPPORTED_EXTENSIONS,
    DocumentParseError,
    extract_text_from_file,
)
from app.llm_client import LLMResponseError, _build_json_schema_response_format
from app.schemas import (
    RAGChatAnswer,
    RAGChatMessage,
    RAGChatResponse,
    RAGCitation,
    RAGIndexResponse,
    RAGSearchMatch,
    RAGSearchResponse,
    RAGStatusResponse,
    S3LeaseFile,
)


CHUNK_WORDS = 1024
CHUNK_OVERLAP_WORDS = 100
EMBEDDING_BATCH_SIZE = 32


class RAGError(RuntimeError):
    """Raised when retrieval-augmented lease search cannot complete."""


class RAGConfigurationError(RAGError):
    """Raised when RAG settings are incomplete or invalid."""


class RAGInvalidKeyError(RAGError):
    """Raised when an indexed lease key is not allowed for retrieval."""


class RAGLeaseNotIndexedError(RAGError):
    """Raised when an indexed lease key has no chunks in ChromaDB."""


@dataclass
class LeaseChunk:
    key: str
    filename: str
    text: str
    chunk_index: int
    s3_prefix: str
    source_extension: str
    size: int
    last_modified: str
    indexed_at: str
    score: float | None = None


class AzureEmbeddingClient:
    def __init__(self, settings: Settings):
        self._client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
        self._deployment = settings.azure_openai_embedding_deployment

    def embed_texts(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        if not self._deployment:
            raise RAGConfigurationError(
                "Azure OpenAI embedding deployment is not configured. "
                "Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT."
            )
        if not texts:
            return []

        generation = None
        if trace is not None:
            generation = trace.generation(
                name="embed-texts",
                model=self._deployment,
                input=texts,
            )

        try:
            response = self._client.embeddings.create(
                model=self._deployment,
                input=texts,
            )
        except Exception:
            if generation is not None:
                generation.end(level="ERROR")
            raise

        data = sorted(response.data, key=lambda item: item.index)

        if generation is not None:
            usage = response.usage
            generation.end(
                output=f"{len(data)} embeddings",
                usage={"input": usage.total_tokens if usage else 0},
            )

        return [item.embedding for item in data]



class AzureRAGChatClient:
    def __init__(self, settings: Settings):
        self._client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
        self._deployment = settings.azure_openai_deployment

    def answer(
    self,
    question: str,
    history: list[RAGChatMessage],
    chunks: list[LeaseChunk],
    trace: Any | None = None,
) -> RAGChatAnswer:
        if not chunks:
            return RAGChatAnswer(
                answer=(
                    "I could not find relevant lease text in the indexed S3 leases "
                    "for that question."
                ),
                citations=[],
            )

        messages = _build_chat_messages(question, history, chunks)

        generation = None
        if trace is not None:
            generation = trace.generation(
                name="rag-answer",
                model=self._deployment,
                input=messages,
            )

        try:
            response = self._client.chat.completions.create(
                model=self._deployment,
                messages=messages,
                temperature=0.0,
                response_format=_build_json_schema_response_format(RAGChatAnswer),
            )
        except Exception:
            if generation is not None:
                generation.end(level="ERROR")
            raise

        content = response.choices[0].message.content

        if generation is not None:
            usage = response.usage
            generation.end(
                output=content or "",
                usage={
                    "input": usage.prompt_tokens if usage else 0,
                    "output": usage.completion_tokens if usage else 0,
                },
            )

        if not content:
            raise LLMResponseError("Azure OpenAI returned an empty RAG answer.")

        try:
            payload = json.loads(content)
            return RAGChatAnswer.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMResponseError(
                "Azure OpenAI returned a RAG answer with an unexpected shape."
            ) from exc


class ChromaLeaseVectorStore:
    def __init__(self, settings: RAGSettings):
        try:
            import chromadb
        except ImportError as exc:
            raise RAGConfigurationError(
                "ChromaDB is not installed. Install the chromadb package."
            ) from exc

        self.collection_name = settings.chroma_collection_name
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset_prefix(self, prefix: str) -> None:
        try:
            self._collection.delete(where={"s3_prefix": prefix})
        except Exception:
            # Chroma raises when there is nothing matching the filter in some versions.
            return

    def upsert_chunks(
        self,
        chunks: list[LeaseChunk],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise RAGError("Chunk and embedding counts did not match.")

        self._collection.upsert(
            ids=[_chunk_id(chunk) for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[_chunk_metadata(chunk) for chunk in chunks],
        )

    def chunks_for_key(self, key: str) -> list[LeaseChunk]:
        try:
            results = self._collection.get(
                where={"s3_key": key},
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise RAGError("Could not load indexed lease chunks.") from exc

        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        chunks: list[LeaseChunk] = []
        for document, metadata in zip(documents, metadatas):
            if not isinstance(metadata, dict):
                continue
            chunks.append(_chunk_from_result(document, metadata, None))

        return sorted(chunks, key=lambda chunk: chunk.chunk_index)

    def status(self) -> RAGStatusResponse:
        data = self._collection.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
        keys = {
            metadata.get("s3_key")
            for metadata in metadatas
            if isinstance(metadata, dict) and metadata.get("s3_key")
        }
        indexed_values = [
            metadata.get("indexed_at", "")
            for metadata in metadatas
            if isinstance(metadata, dict) and metadata.get("indexed_at")
        ]

        return RAGStatusResponse(
            collection_name=self.collection_name,
            indexed_lease_count=len(keys),
            chunk_count=self._collection.count(),
            last_indexed_at=max(indexed_values) if indexed_values else None,
        )

    def query(
        self,
        embedding: list[float],
        top_k: int,
        lease_keys: list[str] | None = None,
    ) -> list[LeaseChunk]:
        where = _lease_key_filter(lease_keys or [])
        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            results = self._collection.query(**kwargs)
        except Exception as exc:
            if self._collection.count() == 0:
                return []
            raise RAGError("Could not query the ChromaDB lease index.") from exc

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        chunks: list[LeaseChunk] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            if not isinstance(metadata, dict):
                continue
            chunks.append(_chunk_from_result(document, metadata, distance))
        return chunks


class RAGService:
    def __init__(
        self,
        vector_store: ChromaLeaseVectorStore,
        embedding_client: AzureEmbeddingClient,
        chat_client: AzureRAGChatClient,
        s3_prefix: str,
        langfuse: Any | None = None,
    ):
        self._vector_store = vector_store
        self._embedding_client = embedding_client
        self._chat_client = chat_client
        self._s3_prefix = _normalize_prefix(s3_prefix)
        self._langfuse = langfuse

    def status(self) -> RAGStatusResponse:
        return self._vector_store.status()

    def index_s3_leases(self, s3_storage) -> RAGIndexResponse:
        trace = self._langfuse.trace(name="rag-index") if self._langfuse else None
        try:
            indexed_at = datetime.now(UTC).isoformat()
            s3_files = s3_storage.list_lease_files()
            chunks: list[LeaseChunk] = []
            skipped_files: list[str] = []
            failed_files: list[str] = []

            self._vector_store.reset_prefix(self._s3_prefix)

            for s3_file in s3_files:
                try:
                    filename, content = s3_storage.get_file(s3_file.key)
                    text = extract_text_from_file(filename, content)
                except DocumentParseError:
                    failed_files.append(s3_file.key)
                    continue
                except Exception:
                    failed_files.append(s3_file.key)
                    continue

                lease_chunks = _chunks_for_file(s3_file, text, self._s3_prefix, indexed_at)
                if not lease_chunks:
                    skipped_files.append(s3_file.key)
                    continue
                chunks.extend(lease_chunks)

            embeddings: list[list[float]] = []
            for batch in _batches([chunk.text for chunk in chunks], EMBEDDING_BATCH_SIZE):
                embeddings.extend(self._embedding_client.embed_texts(batch, trace=trace))

            self._vector_store.upsert_chunks(chunks, embeddings)

            indexed_keys = {chunk.key for chunk in chunks}
            return RAGIndexResponse(
                indexed_lease_count=len(indexed_keys),
                indexed_chunk_count=len(chunks),
                skipped_files=skipped_files,
                failed_files=failed_files,
            )
        finally:
            if self._langfuse:
                self._langfuse.flush()


    def search(self, question: str, top_k: int) -> RAGSearchResponse:
        trace = self._langfuse.trace(name="rag-search") if self._langfuse else None
        try:
            query_embedding = self._embedding_client.embed_texts([question], trace=trace)[0]
            chunks = self._vector_store.query(query_embedding, top_k)
            return RAGSearchResponse(
                question=question,
                matches=[_search_match(chunk) for chunk in chunks],
            )
        finally:
            if self._langfuse:
                self._langfuse.flush()


    def lease_text_from_index(self, key: str) -> str:
        validated_key = _validate_indexed_key(key, self._s3_prefix)
        chunks = self._vector_store.chunks_for_key(validated_key)
        if not chunks:
            raise RAGLeaseNotIndexedError(
                f"Indexed lease was not found: {validated_key}"
            )
        return _merge_indexed_chunks(chunks)

    def chat(
        self,
        question: str,
        lease_keys: list[str],
        history: list[RAGChatMessage],
        top_k: int,
    ) -> RAGChatResponse:
        trace = self._langfuse.trace(name="rag-chat") if self._langfuse else None
        try:
            query_embedding = self._embedding_client.embed_texts([question], trace=trace)[0]
            chunks = self._vector_store.query(
                query_embedding,
                top_k,
                lease_keys=lease_keys,
            )
            answer = self._chat_client.answer(question, history, chunks, trace=trace)
            return RAGChatResponse(
                question=question,
                answer=answer.answer,
                citations=[_citation(chunk) for chunk in chunks],
            )
        finally:
            if self._langfuse:
                self._langfuse.flush()



def create_rag_service(
    settings: Settings,
    rag_settings: RAGSettings,
    s3_settings: S3Settings,
    langfuse: Any | None = None,
) -> RAGService:
    return RAGService(
        vector_store=ChromaLeaseVectorStore(rag_settings),
        embedding_client=AzureEmbeddingClient(settings),
        chat_client=AzureRAGChatClient(settings),
        s3_prefix=s3_settings.s3_prefix,
        langfuse=langfuse,
    )



def _chunks_for_file(
    s3_file: S3LeaseFile,
    text: str,
    s3_prefix: str,
    indexed_at: str,
) -> list[LeaseChunk]:
    chunks: list[LeaseChunk] = []
    filename = s3_file.filename or PurePosixPath(s3_file.key).name
    last_modified = s3_file.last_modified.isoformat() if s3_file.last_modified else ""
    extension = PurePosixPath(filename).suffix.lower()
    for index, chunk_text in enumerate(split_text_into_chunks(text)):
        chunks.append(
            LeaseChunk(
                key=s3_file.key,
                filename=filename,
                text=chunk_text,
                chunk_index=index,
                s3_prefix=s3_prefix,
                source_extension=extension,
                size=s3_file.size,
                last_modified=last_modified,
                indexed_at=indexed_at,
            )
        )
    return chunks


def split_text_into_chunks(
    text: str,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap_words, start + 1)
    return chunks


def _build_chat_messages(
    question: str,
    history: list[RAGChatMessage],
    chunks: list[LeaseChunk],
) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[{index}] {chunk.filename} ({chunk.key}), chunk {chunk.chunk_index}:\n"
        f"{chunk.text}"
        for index, chunk in enumerate(chunks, start=1)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions about residential leases for a non-legal "
                "audience. Use only the retrieved snippets. If the snippets do not "
                "support an answer, say that the indexed lease text does not contain "
                "that information. Return the structured JSON answer requested by "
                "the API schema."
            ),
        }
    ]
    messages.extend(
        {"role": item.role, "content": item.content}
        for item in history[-10:]
    )
    messages.append(
        {
            "role": "user",
            "content": f"Question: {question}\n\nRetrieved snippets:\n{context}",
        }
    )
    return messages


def _search_match(chunk: LeaseChunk) -> RAGSearchMatch:
    return RAGSearchMatch(
        key=chunk.key,
        filename=chunk.filename,
        snippet=chunk.text,
        score=chunk.score,
        chunk_index=chunk.chunk_index,
    )


def _citation(chunk: LeaseChunk) -> RAGCitation:
    return RAGCitation(
        key=chunk.key,
        filename=chunk.filename,
        snippet=chunk.text,
        chunk_index=chunk.chunk_index,
    )


def _merge_indexed_chunks(chunks: list[LeaseChunk]) -> str:
    merged_words: list[str] = []
    for chunk in sorted(chunks, key=lambda item: item.chunk_index):
        chunk_words = chunk.text.split()
        if not merged_words:
            merged_words.extend(chunk_words)
            continue
        _append_without_overlap(merged_words, chunk_words)
    return " ".join(merged_words)


def _append_without_overlap(
    existing_words: list[str],
    chunk_words: list[str],
) -> None:
    max_overlap = min(
        CHUNK_OVERLAP_WORDS,
        len(existing_words),
        len(chunk_words),
    )
    for overlap in range(max_overlap, 0, -1):
        if existing_words[-overlap:] == chunk_words[:overlap]:
            existing_words.extend(chunk_words[overlap:])
            return
    existing_words.extend(chunk_words)


def _chunk_metadata(chunk: LeaseChunk) -> dict[str, Any]:
    return {
        "s3_key": chunk.key,
        "filename": chunk.filename,
        "chunk_index": chunk.chunk_index,
        "s3_prefix": chunk.s3_prefix,
        "source_extension": chunk.source_extension,
        "size": chunk.size,
        "last_modified": chunk.last_modified,
        "indexed_at": chunk.indexed_at,
    }


def _chunk_from_result(
    document: str,
    metadata: dict[str, Any],
    distance: float | None,
) -> LeaseChunk:
    score = None if distance is None else 1 / (1 + float(distance))
    return LeaseChunk(
        key=str(metadata.get("s3_key", "")),
        filename=str(metadata.get("filename", "")),
        text=document,
        chunk_index=int(metadata.get("chunk_index", 0)),
        s3_prefix=str(metadata.get("s3_prefix", "")),
        source_extension=str(metadata.get("source_extension", "")),
        size=int(metadata.get("size", 0)),
        last_modified=str(metadata.get("last_modified", "")),
        indexed_at=str(metadata.get("indexed_at", "")),
        score=score,
    )


def _chunk_id(chunk: LeaseChunk) -> str:
    digest = hashlib.sha256(f"{chunk.key}:{chunk.chunk_index}".encode("utf-8")).hexdigest()
    return f"lease-chunk-{digest}"


def _lease_key_filter(lease_keys: list[str]) -> dict[str, Any] | None:
    keys = [key for key in lease_keys if key]
    if not keys:
        return None
    if len(keys) == 1:
        return {"s3_key": keys[0]}
    return {"s3_key": {"$in": keys}}


def _validate_indexed_key(key: str, prefix: str) -> str:
    normalized_key = key.strip().lstrip("/")
    if not normalized_key:
        raise RAGInvalidKeyError("S3 key is required.")
    if "\\" in normalized_key:
        raise RAGInvalidKeyError("S3 key must use forward slashes.")
    if prefix and not normalized_key.startswith(f"{prefix}/"):
        raise RAGInvalidKeyError("S3 key must be inside the configured S3_PREFIX.")
    if PurePosixPath(normalized_key).suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise RAGInvalidKeyError(f"Unsupported file type. Use {supported}.")
    return normalized_key


def _batches(items: list[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _normalize_prefix(prefix: str | None) -> str:
    return (prefix or "").strip().strip("/")
