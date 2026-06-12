from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Callable

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
    FileCollectionInfo,
    RAGChatAnswer,
    RAGChatMessage,
    RAGChatResponse,
    RAGCitation,
    RAGIndexResponse,
    RAGSearchMatch,
    RAGSearchResponse,
    RAGStatusResponse,
    S3LeaseFile,
    SummariseResponse,
    validate_lease_text,
)


CHUNK_WORDS = 1024
CHUNK_OVERLAP_WORDS = 100
EMBEDDING_BATCH_SIZE = 32

IndexProgressCallback = Callable[[int, int, str, str | None], None]


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


@dataclass
class LeaseSummaryRecord:
    key: str
    filename: str
    s3_prefix: str
    size: int
    last_modified: str
    indexed_at: str
    summary: SummariseResponse
    monthly_rent_amount_numeric: float | None = None


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
        summaries: list[LeaseSummaryRecord],
        trace: Any | None = None,
    ) -> RAGChatAnswer:
        if not chunks and not summaries:
            return RAGChatAnswer(
                answer=(
                    "I could not find relevant lease text or indexed lease summaries "
                    "for that question."
                ),
                citations=[],
            )

        generation = None
        if trace is not None:
            generation = trace.generation(
                name="rag-chat-answer",
                model=self._deployment,
                input={"question": question, "chunks": len(chunks), "summaries": len(summaries)},
            )

        try:
            response = self._client.chat.completions.create(
                model=self._deployment,
                messages=_build_chat_messages(question, history, chunks, summaries),
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

        self._persist_dir = settings.chroma_persist_dir
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        # Per-file collection cache: s3_key -> collection object
        self._collections: dict[str, Any] = {}
        print(f"[RAG] ChromaLeaseVectorStore ready at {settings.chroma_persist_dir!r}")

    def _get_or_create_collection(self, key: str) -> Any:
        if key not in self._collections:
            name = _collection_name_for_key(key)
            print(f"[RAG] get_or_create_collection: key={key!r} -> collection={name!r}")
            self._collections[key] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine", "s3_key": key},
            )
        return self._collections[key]

    def _list_all_collection_names(self) -> list[str]:
        # ChromaDB v0.6+ list_collections() returns name-only objects; str() extracts the name.
        try:
            return [str(col) for col in self._client.list_collections()]
        except Exception as exc:
            print(f"[RAG] list_collections error: {exc}")
            return []

    def _collection_s3_key(self, col_obj: Any) -> str:
        """Extract s3_key from collection-level metadata, fall back to first doc metadata."""
        meta = getattr(col_obj, "metadata", {}) or {}
        s3_key = str(meta.get("s3_key", ""))
        if not s3_key:
            # Legacy single-collection: read first document's metadata
            try:
                peek = col_obj.peek(limit=1)
                for m in (peek.get("metadatas") or []):
                    if isinstance(m, dict) and m.get("s3_key"):
                        s3_key = str(m["s3_key"])
                        break
            except Exception:
                pass
        return s3_key

    def reset_prefix(self, prefix: str) -> None:
        all_names = self._list_all_collection_names()
        print(f"[RAG] reset_prefix: scanning {len(all_names)} collections for prefix={prefix!r}")
        for name in all_names:
            try:
                col_obj = self._client.get_collection(name)
                s3_key = self._collection_s3_key(col_obj)
                if prefix and (
                    s3_key.startswith(f"{prefix}/")
                    # Also catch legacy single collections that store s3_prefix in docs
                    or _col_has_prefix(col_obj, prefix)
                ):
                    print(f"[RAG] reset_prefix: deleting collection {name!r} (key={s3_key!r})")
                    self._client.delete_collection(name)
                    self._collections.pop(s3_key, None)
            except Exception as exc:
                print(f"[RAG] reset_prefix: error on {name!r}: {exc}")
                continue

    def upsert_chunks(
        self,
        chunks: list[LeaseChunk],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise RAGError("Chunk and embedding counts did not match.")

        # Group chunks and their embeddings by s3 key so each file gets its own collection.
        key_groups: dict[str, tuple[list[LeaseChunk], list[list[float]]]] = {}
        for chunk, embedding in zip(chunks, embeddings):
            if chunk.key not in key_groups:
                key_groups[chunk.key] = ([], [])
            key_groups[chunk.key][0].append(chunk)
            key_groups[chunk.key][1].append(embedding)

        for key, (file_chunks, file_embeddings) in key_groups.items():
            print(f"[RAG] upsert_chunks: {len(file_chunks)} chunks for key={key!r}")
            collection = self._get_or_create_collection(key)
            collection.upsert(
                ids=[_chunk_id(c) for c in file_chunks],
                documents=[c.text for c in file_chunks],
                embeddings=file_embeddings,
                metadatas=[_chunk_metadata(c) for c in file_chunks],
            )

    def chunks_for_key(self, key: str) -> list[LeaseChunk]:
        try:
            collection = self._get_or_create_collection(key)
            results = collection.get(include=["documents", "metadatas"])
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

    def file_collection_details(self) -> list[FileCollectionInfo]:
        """Return per-file collection stats for display in the UI."""
        all_names = self._list_all_collection_names()
        print(f"[RAG] file_collection_details: scanning {len(all_names)} collections")
        details: list[FileCollectionInfo] = []
        for name in all_names:
            try:
                col_obj = self._client.get_collection(name)
                s3_key = self._collection_s3_key(col_obj)
                count = col_obj.count()
                indexed_at: str | None = None
                if count > 0:
                    peek = col_obj.peek(limit=1)
                    for m in (peek.get("metadatas") or []):
                        if isinstance(m, dict) and m.get("indexed_at"):
                            indexed_at = str(m["indexed_at"])
                            break
                filename = PurePosixPath(s3_key).name if s3_key else name
                print(f"[RAG]   {name!r}: key={s3_key!r} filename={filename!r} chunks={count}")
                details.append(FileCollectionInfo(
                    s3_key=s3_key,
                    filename=filename,
                    collection_name=name,
                    chunk_count=count,
                    indexed_at=indexed_at,
                ))
            except Exception as exc:
                print(f"[RAG] file_collection_details: error on {name!r}: {exc}")
                continue
        return details

    def status(self) -> RAGStatusResponse:
        all_names = self._list_all_collection_names()
        print(f"[RAG] status: checking {len(all_names)} collections")
        total_chunks = 0
        indexed_at_values: list[str] = []
        lease_keys: set[str] = set()
        file_collections: list[FileCollectionInfo] = []

        for name in all_names:
            try:
                col_obj = self._client.get_collection(name)
                s3_key = self._collection_s3_key(col_obj)
                if s3_key:
                    lease_keys.add(s3_key)
                count = col_obj.count()
                total_chunks += count
                indexed_at: str | None = None
                print(f"[RAG] status:  {name!r} key={s3_key!r} chunks={count}")
                if count > 0:
                    peek = col_obj.peek(limit=1)
                    for m in (peek.get("metadatas") or []):
                        if isinstance(m, dict):
                            if m.get("indexed_at"):
                                indexed_at = str(m["indexed_at"])
                                indexed_at_values.append(indexed_at)
                filename = PurePosixPath(s3_key).name if s3_key else name
                file_collections.append(FileCollectionInfo(
                    s3_key=s3_key,
                    filename=filename,
                    collection_name=name,
                    chunk_count=count,
                    indexed_at=indexed_at,
                ))
            except Exception as exc:
                print(f"[RAG] status: error on {name!r}: {exc}")
                continue

        num_collections = len(all_names)
        print(f"[RAG] status result: {len(lease_keys)} leases, {total_chunks} chunks, {num_collections} collections")
        return RAGStatusResponse(
            collection_name=f"per-file ({num_collections} collections)",
            indexed_lease_count=len(lease_keys),
            chunk_count=total_chunks,
            last_indexed_at=max(indexed_at_values) if indexed_at_values else None,
            file_collections=file_collections,
        )

    def query(
        self,
        embedding: list[float],
        top_k: int,
        lease_keys: list[str] | None = None,
    ) -> list[LeaseChunk]:
        if lease_keys:
            target_pairs = [(key, self._get_or_create_collection(key)) for key in lease_keys]
        else:
            target_pairs = []
            for name in self._list_all_collection_names():
                try:
                    col_obj = self._client.get_collection(name)
                    s3_key = self._collection_s3_key(col_obj)
                    target_pairs.append((s3_key, col_obj))
                except Exception as exc:
                    print(f"[RAG] query: error loading {name!r}: {exc}")
                    continue

        all_chunks: list[LeaseChunk] = []
        for _, collection in target_pairs:
            try:
                count = collection.count()
                if count == 0:
                    continue
                results = collection.query(
                    query_embeddings=[embedding],
                    n_results=min(top_k, count),
                    include=["documents", "metadatas", "distances"],
                )
                documents = (results.get("documents") or [[]])[0]
                metadatas = (results.get("metadatas") or [[]])[0]
                distances = (results.get("distances") or [[]])[0]
                for document, metadata, distance in zip(documents, metadatas, distances):
                    if not isinstance(metadata, dict):
                        continue
                    all_chunks.append(_chunk_from_result(document, metadata, distance))
            except Exception as exc:
                print(f"[RAG] query: error querying collection: {exc}")
                continue

        # Merge across collections: best score first, then return top_k
        all_chunks.sort(key=lambda c: c.score or 0.0, reverse=True)
        return all_chunks[:top_k]


class LeaseSummaryStore:
    def __init__(self, persist_dir: str, filename: str = "lease_summaries.json"):
        self._path = Path(persist_dir) / filename

    def reset_prefix(self, prefix: str) -> None:
        records = [
            record
            for record in self._load_records()
            if str(record.get("s3_prefix", "")) != prefix
        ]
        self._write_records(records)

    def upsert_summaries(self, summaries: list[LeaseSummaryRecord]) -> None:
        if not summaries:
            return

        existing = {
            str(record.get("key", "")): record
            for record in self._load_records()
            if record.get("key")
        }
        for summary in summaries:
            existing[summary.key] = _summary_record_to_payload(summary)
        self._write_records(list(existing.values()))

    def list_summaries(
        self,
        prefix: str,
        lease_keys: list[str] | None = None,
    ) -> list[LeaseSummaryRecord]:
        key_filter = {key for key in lease_keys or [] if key}
        summaries: list[LeaseSummaryRecord] = []
        for record in self._load_records():
            if str(record.get("s3_prefix", "")) != prefix:
                continue
            if key_filter and str(record.get("key", "")) not in key_filter:
                continue
            summary = _summary_record_from_payload(record)
            if summary is not None:
                summaries.append(summary)
        return summaries

    def count(self, prefix: str) -> int:
        return len(self.list_summaries(prefix))

    def _load_records(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RAGError("Could not read indexed lease summaries.") from exc

        records = payload.get("summaries", [])
        if not isinstance(records, list):
            return []
        return [record for record in records if isinstance(record, dict)]

    def _write_records(self, records: list[dict[str, Any]]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self._path.with_suffix(f"{self._path.suffix}.tmp")
            temporary_path.write_text(
                json.dumps({"summaries": records}, indent=2),
                encoding="utf-8",
            )
            temporary_path.replace(self._path)
        except OSError as exc:
            raise RAGError("Could not write indexed lease summaries.") from exc


class InMemoryLeaseSummaryStore:
    def __init__(self) -> None:
        self.records: list[LeaseSummaryRecord] = []

    def reset_prefix(self, prefix: str) -> None:
        self.records = [record for record in self.records if record.s3_prefix != prefix]

    def upsert_summaries(self, summaries: list[LeaseSummaryRecord]) -> None:
        existing = {record.key: record for record in self.records}
        for summary in summaries:
            existing[summary.key] = summary
        self.records = list(existing.values())

    def list_summaries(
        self,
        prefix: str,
        lease_keys: list[str] | None = None,
    ) -> list[LeaseSummaryRecord]:
        key_filter = {key for key in lease_keys or [] if key}
        return [
            record
            for record in self.records
            if record.s3_prefix == prefix
            and (not key_filter or record.key in key_filter)
        ]

    def count(self, prefix: str) -> int:
        return len(self.list_summaries(prefix))


class RAGService:
    def __init__(
        self,
        vector_store: ChromaLeaseVectorStore,
        embedding_client: AzureEmbeddingClient,
        chat_client: AzureRAGChatClient,
        s3_prefix: str,
        summary_store: LeaseSummaryStore | InMemoryLeaseSummaryStore | None = None,
        langfuse: Any | None = None,
    ):
        self._vector_store = vector_store
        self._embedding_client = embedding_client
        self._chat_client = chat_client
        self._s3_prefix = _normalize_prefix(s3_prefix)
        self._summary_store = summary_store if summary_store is not None else InMemoryLeaseSummaryStore()
        self._langfuse = langfuse

    def status(self) -> RAGStatusResponse:
        status = self._vector_store.status()
        status.indexed_summary_count = self._summary_store.count(self._s3_prefix)
        return status

    def index_s3_leases(
        self,
        s3_storage: Any,
        _lease_service: Any | None = None,
        progress_callback: IndexProgressCallback | None = None,
    ) -> RAGIndexResponse:
        indexed_at = datetime.now(UTC).isoformat()
        s3_files = s3_storage.list_lease_files()
        total_files = len(s3_files)
        chunks: list[LeaseChunk] = []
        skipped_files: list[str] = []
        failed_files: list[str] = []

        self._vector_store.reset_prefix(self._s3_prefix)
        _report_index_progress(progress_callback, 0, total_files, "Loading lease files.")

        for file_index, s3_file in enumerate(s3_files):
            _report_index_progress(
                progress_callback,
                file_index,
                total_files,
                f"Reading {s3_file.key}",
                s3_file.key,
            )
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

        _report_index_progress(
            progress_callback, total_files, total_files, "Embedding and indexing chunks."
        )
        trace = self._langfuse.trace(name="rag-index") if self._langfuse else None
        try:
            embeddings: list[list[float]] = []
            for batch in _batches([chunk.text for chunk in chunks], EMBEDDING_BATCH_SIZE):
                embeddings.extend(self._embedding_client.embed_texts(batch, trace=trace))

            self._vector_store.upsert_chunks(chunks, embeddings)
        finally:
            if self._langfuse:
                self._langfuse.flush()

        indexed_keys = {chunk.key for chunk in chunks}
        return RAGIndexResponse(
            indexed_lease_count=len(indexed_keys),
            indexed_chunk_count=len(chunks),
            skipped_files=skipped_files,
            failed_files=failed_files,
        )

    def index_file(
        self,
        s3_key: str,
        filename: str,
        content: bytes,
        size: int,
    ) -> dict[str, Any]:
        """Chunk, embed and index a single file into its own per-file collection."""
        indexed_at = datetime.now(UTC).isoformat()
        print(f"[RAG] index_file: key={s3_key!r} filename={filename!r} size={size}")

        text = extract_text_from_file(filename, content)
        word_count = len(text.split())
        print(f"[RAG] index_file: extracted {word_count} words")

        s3_file = S3LeaseFile(key=s3_key, filename=filename, size=size)
        chunks = _chunks_for_file(s3_file, text, self._s3_prefix, indexed_at)
        print(f"[RAG] index_file: generated {len(chunks)} chunks for {s3_key!r}")

        trace = self._langfuse.trace(name="rag-index-file") if self._langfuse else None
        try:
            embeddings: list[list[float]] = []
            for batch in _batches([c.text for c in chunks], EMBEDDING_BATCH_SIZE):
                embeddings.extend(self._embedding_client.embed_texts(batch, trace=trace))
            self._vector_store.upsert_chunks(chunks, embeddings)
        finally:
            if self._langfuse:
                self._langfuse.flush()

        collection_name = _collection_name_for_key(s3_key)
        print(f"[RAG] index_file: done — collection={collection_name!r} chunks={len(chunks)}")
        return {
            "s3_key": s3_key,
            "filename": filename,
            "collection_name": collection_name,
            "chunk_count": len(chunks),
            "word_count": word_count,
        }

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
            summaries = self._summary_store.list_summaries(self._s3_prefix, lease_keys or None)
            aggregate = _answer_summary_aggregate_question(question, summaries)
            if aggregate is not None:
                return aggregate
            query_embedding = self._embedding_client.embed_texts([question], trace=trace)[0]
            chunks = self._vector_store.query(
                query_embedding,
                top_k,
                lease_keys=lease_keys,
            )
            answer = self._chat_client.answer(question, history, chunks, summaries, trace=trace)
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
        summary_store=LeaseSummaryStore(rag_settings.chroma_persist_dir),
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
    summaries: list[LeaseSummaryRecord],
) -> list[dict[str, str]]:
    summary_context = "\n\n".join(
        _summary_context(summary)
        for summary in summaries
    )
    chunk_context = "\n\n".join(
        f"[{index}] {chunk.filename} ({chunk.key}), chunk {chunk.chunk_index}:\n"
        f"{chunk.text}"
        for index, chunk in enumerate(chunks, start=1)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions about residential leases for a non-legal "
                "audience. Use the structured lease summaries for lease-level facts "
                "and comparisons. Use retrieved snippets for supporting detail and "
                "exact wording. If neither source supports an answer, say that the "
                "indexed lease information does not contain that information. Return "
                "the structured JSON answer requested by the API schema. If you cite "
                "a source, set source_type to 'summary' for summary facts and 'chunk' "
                "for retrieved snippets."
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
            "content": (
                f"Question: {question}\n\n"
                f"Structured lease summaries:\n{summary_context or 'None'}\n\n"
                f"Retrieved lease snippets:\n{chunk_context or 'None'}"
            ),
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
        source_type="chunk",
    )


def _summary_citation(summary: LeaseSummaryRecord) -> RAGCitation:
    return RAGCitation(
        key=summary.key,
        filename=summary.filename,
        snippet=_summary_snippet(summary),
        chunk_index=-1,
        source_type="summary",
    )


def _summary_record_for_file(
    s3_file: S3LeaseFile,
    summary: SummariseResponse,
    s3_prefix: str,
    indexed_at: str,
) -> LeaseSummaryRecord:
    filename = s3_file.filename or PurePosixPath(s3_file.key).name
    last_modified = s3_file.last_modified.isoformat() if s3_file.last_modified else ""
    return LeaseSummaryRecord(
        key=s3_file.key,
        filename=filename,
        s3_prefix=s3_prefix,
        size=s3_file.size,
        last_modified=last_modified,
        indexed_at=indexed_at,
        summary=summary,
        monthly_rent_amount_numeric=_normalise_money_amount(
            summary.extraction.monthly_rent_amount
        ),
    )


def _summary_record_to_payload(summary: LeaseSummaryRecord) -> dict[str, Any]:
    return {
        "key": summary.key,
        "filename": summary.filename,
        "s3_prefix": summary.s3_prefix,
        "size": summary.size,
        "last_modified": summary.last_modified,
        "indexed_at": summary.indexed_at,
        "monthly_rent_amount_numeric": summary.monthly_rent_amount_numeric,
        "summary": summary.summary.model_dump(mode="json"),
    }


def _summary_record_from_payload(
    payload: dict[str, Any],
) -> LeaseSummaryRecord | None:
    summary_payload = payload.get("summary")
    if not isinstance(summary_payload, dict):
        return None
    try:
        summary = SummariseResponse.model_validate(summary_payload)
    except ValidationError:
        return None

    return LeaseSummaryRecord(
        key=str(payload.get("key", "")),
        filename=str(payload.get("filename", "")),
        s3_prefix=str(payload.get("s3_prefix", "")),
        size=int(payload.get("size", 0)),
        last_modified=str(payload.get("last_modified", "")),
        indexed_at=str(payload.get("indexed_at", "")),
        summary=summary,
        monthly_rent_amount_numeric=_optional_float(
            payload.get("monthly_rent_amount_numeric")
        ),
    )


def _answer_summary_aggregate_question(
    question: str,
    summaries: list[LeaseSummaryRecord],
) -> RAGChatResponse | None:
    mode = _rent_aggregate_mode(question)
    if mode is None:
        return None

    candidates = [
        summary
        for summary in summaries
        if summary.monthly_rent_amount_numeric is not None
    ]
    missing = [
        summary.filename or summary.key
        for summary in summaries
        if summary.monthly_rent_amount_numeric is None
    ]

    if not summaries:
        return RAGChatResponse(
            question=question,
            answer=(
                "I do not have indexed lease summaries available for the selected "
                "leases, so I cannot compare monthly rent values."
            ),
            citations=[],
        )

    if not candidates:
        return RAGChatResponse(
            question=question,
            answer=(
                "The indexed lease summaries do not contain enough monthly rent "
                "information to answer that comparison."
            ),
            citations=[_summary_citation(summary) for summary in summaries],
        )

    selected = min(candidates, key=lambda item: item.monthly_rent_amount_numeric)
    adjective = "cheapest"
    if mode == "highest":
        selected = max(candidates, key=lambda item: item.monthly_rent_amount_numeric)
        adjective = "highest"

    rent_text = selected.summary.extraction.monthly_rent_amount or "the extracted rent"
    answer = (
        f"The {adjective} monthly rent I found is {rent_text} in "
        f"{selected.filename} ({selected.key}). I compared "
        f"{len(candidates)} indexed lease summaries with parseable monthly rent."
    )
    if missing:
        answer += " I could not include these leases because their monthly rent was missing or unparseable: "
        answer += ", ".join(missing) + "."

    return RAGChatResponse(
        question=question,
        answer=answer,
        citations=[_summary_citation(selected)],
    )


def _rent_aggregate_mode(question: str) -> str | None:
    cleaned = question.lower()
    rent_terms = {"rent", "rental", "lease", "leases", "price", "cost"}
    if not any(term in cleaned for term in rent_terms):
        return None
    if any(
        term in cleaned
        for term in ("cheapest", "lowest", "least expensive", "minimum", "min ")
    ):
        return "lowest"
    if any(
        term in cleaned
        for term in ("highest", "most expensive", "maximum", "max ")
    ):
        return "highest"
    return None


def _summary_context(summary: LeaseSummaryRecord) -> str:
    extraction = summary.summary.extraction
    payload = {
        "key": summary.key,
        "filename": summary.filename,
        "tenant_name": extraction.tenant_name,
        "landlord_name": extraction.landlord_name,
        "property_address": extraction.property_address,
        "lease_start_date": extraction.lease_start_date,
        "lease_end_date": extraction.lease_end_date,
        "monthly_rent_amount": extraction.monthly_rent_amount,
        "monthly_rent_amount_numeric": summary.monthly_rent_amount_numeric,
        "rent_payment_due_date": extraction.rent_payment_due_date,
        "security_deposit_amount": extraction.security_deposit_amount,
        "notice_period_to_vacate": extraction.notice_period_to_vacate,
        "tenant_obligations": extraction.tenant_obligations,
        "landlord_obligations": extraction.landlord_obligations,
        "unusual_clauses": extraction.unusual_clauses,
        "plain_english_summary": extraction.plain_english_summary,
        "warnings": summary.summary.warnings,
    }
    return json.dumps(payload, ensure_ascii=False)


def _summary_snippet(summary: LeaseSummaryRecord) -> str:
    extraction = summary.summary.extraction
    parts = [
        f"monthly_rent_amount: {extraction.monthly_rent_amount}",
        f"property_address: {extraction.property_address}",
        f"lease_start_date: {extraction.lease_start_date}",
        f"lease_end_date: {extraction.lease_end_date}",
        f"security_deposit_amount: {extraction.security_deposit_amount}",
    ]
    return "; ".join(part for part in parts if not part.endswith(": None"))


def _normalise_money_amount(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"(\d[\d,]*(?:\.\d+)?)", value)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _col_has_prefix(col_obj: Any, prefix: str) -> bool:
    """Check if a legacy collection (no s3_key in collection metadata) belongs to prefix."""
    try:
        peek = col_obj.peek(limit=1)
        for m in (peek.get("metadatas") or []):
            if isinstance(m, dict) and str(m.get("s3_prefix", "")) == prefix:
                return True
    except Exception:
        pass
    return False


def _collection_name_for_key(s3_key: str) -> str:
    # Derive a stable, valid Chroma collection name from the S3 key.
    # Chroma rules: 3-63 chars, alphanumeric/underscore/hyphen, start/end alphanumeric.
    digest = hashlib.sha256(s3_key.encode("utf-8")).hexdigest()[:12]
    stem = PurePosixPath(s3_key).stem
    readable = re.sub(r"[^a-zA-Z0-9]", "_", stem)[:20].strip("_")
    name = f"lease_{digest}_{readable}" if readable else f"lease_{digest}"
    name = name[:63].rstrip("_")
    return name


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


def _report_index_progress(
    progress_callback: IndexProgressCallback | None,
    current: int,
    total: int,
    message: str,
    current_key: str | None = None,
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(current, total, message, current_key)
    except Exception:
        return


def _normalize_prefix(prefix: str | None) -> str:
    return (prefix or "").strip().strip("/")
