from datetime import UTC, datetime

import pytest

from app.config import Settings
from app.rag_service import (
    AzureEmbeddingClient,
    LeaseChunk,
    RAGConfigurationError,
    RAGInvalidKeyError,
    RAGLeaseNotIndexedError,
    RAGService,
    split_text_into_chunks,
)
from app.schemas import (
    RAGChatAnswer,
    RAGChatMessage,
    RAGCitation,
    RAGStatusResponse,
    S3LeaseFile,
)


class FakeS3Storage:
    def __init__(self, files: dict[str, str]) -> None:
        self.files = files

    def list_lease_files(self) -> list[S3LeaseFile]:
        return [
            S3LeaseFile(
                key=key,
                filename=key.rsplit("/", 1)[-1],
                size=len(text.encode("utf-8")),
                last_modified=datetime(2026, 1, 1, tzinfo=UTC),
            )
            for key, text in self.files.items()
        ]

    def get_file(self, key: str) -> tuple[str, bytes]:
        return key.rsplit("/", 1)[-1], self.files[key].encode("utf-8")


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[float(len(text.split()))] for text in texts]


class FakeVectorStore:
    def __init__(self) -> None:
        self.reset_prefixes: list[str] = []
        self.chunks: list[LeaseChunk] = []
        self.embeddings: list[list[float]] = []
        self.queries: list[dict[str, object]] = []
        self.query_chunks: list[LeaseChunk] = []
        self.chunks_by_key: dict[str, list[LeaseChunk]] = {}

    def reset_prefix(self, prefix: str) -> None:
        self.reset_prefixes.append(prefix)

    def upsert_chunks(
        self,
        chunks: list[LeaseChunk],
        embeddings: list[list[float]],
    ) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        self.chunks_by_key = {}
        for chunk in chunks:
            self.chunks_by_key.setdefault(chunk.key, []).append(chunk)

    def status(self) -> RAGStatusResponse:
        return RAGStatusResponse(
            collection_name="lease_chunks",
            indexed_lease_count=1,
            chunk_count=len(self.chunks),
            last_indexed_at=None,
        )

    def query(
        self,
        embedding: list[float],
        top_k: int,
        lease_keys: list[str] | None = None,
    ) -> list[LeaseChunk]:
        self.queries.append(
            {
                "embedding": embedding,
                "top_k": top_k,
                "lease_keys": lease_keys,
            }
        )
        return self.query_chunks

    def chunks_for_key(self, key: str) -> list[LeaseChunk]:
        return self.chunks_by_key.get(key, [])


class FakeChatClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def answer(
        self,
        question: str,
        history: list[RAGChatMessage],
        chunks: list[LeaseChunk],
    ) -> RAGChatAnswer:
        self.calls.append(
            {
                "question": question,
                "history": history,
                "chunks": chunks,
            }
        )
        return RAGChatAnswer(
            answer="Rent is due on the first day.",
            citations=[
                RAGCitation(
                    key="model-made-up-key",
                    filename="model-made-up-file.txt",
                    snippet="Model supplied citation text.",
                    chunk_index=99,
                )
            ],
        )


def lease_text(word_count: int, marker: str) -> str:
    return " ".join(f"{marker}{index}" for index in range(word_count))


def chunk(key: str, text: str, index: int) -> LeaseChunk:
    return LeaseChunk(
        key=key,
        filename=key.rsplit("/", 1)[-1],
        text=text,
        chunk_index=index,
        s3_prefix="sample_leases",
        source_extension=".txt",
        size=100,
        last_modified="",
        indexed_at="",
    )


def make_service(
    vector_store: FakeVectorStore | None = None,
    embedding_client: FakeEmbeddingClient | None = None,
    chat_client: FakeChatClient | None = None,
) -> RAGService:
    return RAGService(
        vector_store=vector_store or FakeVectorStore(),
        embedding_client=embedding_client or FakeEmbeddingClient(),
        chat_client=chat_client or FakeChatClient(),
        s3_prefix="sample_leases",
    )


def test_split_text_into_overlapping_chunks():
    chunks = split_text_into_chunks(
        " ".join(str(index) for index in range(10)),
        chunk_words=4,
        overlap_words=1,
    )

    assert chunks == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]


def test_indexing_downloads_s3_leases_chunks_embeds_and_upserts():
    vector_store = FakeVectorStore()
    embedding_client = FakeEmbeddingClient()
    service = make_service(vector_store=vector_store, embedding_client=embedding_client)
    s3_storage = FakeS3Storage(
        {
            "sample_leases/lease_a.txt": lease_text(450, "a"),
            "sample_leases/lease_b.txt": lease_text(120, "b"),
        }
    )

    response = service.index_s3_leases(s3_storage)

    assert vector_store.reset_prefixes == ["sample_leases"]
    assert response.indexed_lease_count == 2
    assert response.indexed_chunk_count == len(vector_store.chunks)
    assert response.failed_files == []
    assert len(vector_store.embeddings) == len(vector_store.chunks)
    assert [chunk.key for chunk in vector_store.chunks].count(
        "sample_leases/lease_a.txt"
    ) == 2
    assert all(chunk.s3_prefix == "sample_leases" for chunk in vector_store.chunks)
    assert sum(len(batch) for batch in embedding_client.batches) == len(
        vector_store.chunks
    )


def test_indexing_records_parse_failures():
    vector_store = FakeVectorStore()
    service = make_service(vector_store=vector_store)
    s3_storage = FakeS3Storage(
        {
            "sample_leases/lease_a.txt": lease_text(120, "a"),
            "sample_leases/notes.rtf": "unsupported",
        }
    )

    response = service.index_s3_leases(s3_storage)

    assert response.indexed_lease_count == 1
    assert response.failed_files == ["sample_leases/notes.rtf"]
    assert [chunk.key for chunk in vector_store.chunks] == [
        "sample_leases/lease_a.txt"
    ]


def test_search_returns_matching_lease_snippets():
    vector_store = FakeVectorStore()
    vector_store.query_chunks = [
        LeaseChunk(
            key="sample_leases/lease_a.txt",
            filename="lease_a.txt",
            text="Monthly rent is 1,500 pounds.",
            chunk_index=0,
            s3_prefix="sample_leases",
            source_extension=".txt",
            size=100,
            last_modified="",
            indexed_at="",
            score=0.82,
        )
    ]
    service = make_service(vector_store=vector_store)

    response = service.search("What is the rent?", top_k=3)

    assert vector_store.queries == [
        {"embedding": [4.0], "top_k": 3, "lease_keys": None}
    ]
    assert response.matches[0].key == "sample_leases/lease_a.txt"
    assert response.matches[0].snippet == "Monthly rent is 1,500 pounds."


def test_lease_text_from_index_rebuilds_overlapping_chunks_in_order():
    vector_store = FakeVectorStore()
    expected_words = [f"w{index}" for index in range(500)]
    first_chunk = " ".join(expected_words[:400])
    second_chunk = " ".join(expected_words[320:])
    vector_store.chunks_by_key["sample_leases/lease_a.txt"] = [
        chunk("sample_leases/lease_a.txt", second_chunk, 1),
        chunk("sample_leases/lease_a.txt", first_chunk, 0),
    ]
    service = make_service(vector_store=vector_store)

    text = service.lease_text_from_index("sample_leases/lease_a.txt")

    assert text.split() == expected_words


def test_lease_text_from_index_returns_404_style_error_when_missing():
    service = make_service(vector_store=FakeVectorStore())

    with pytest.raises(RAGLeaseNotIndexedError, match="Indexed lease was not found"):
        service.lease_text_from_index("sample_leases/missing.txt")


def test_lease_text_from_index_blocks_keys_outside_prefix():
    service = make_service(vector_store=FakeVectorStore())

    with pytest.raises(RAGInvalidKeyError, match="configured S3_PREFIX"):
        service.lease_text_from_index("other_prefix/lease_a.txt")


def test_chat_filters_by_selected_keys_and_passes_history_to_chat_client():
    vector_store = FakeVectorStore()
    vector_store.query_chunks = [
        LeaseChunk(
            key="sample_leases/lease_a.txt",
            filename="lease_a.txt",
            text="Rent is due on the first day.",
            chunk_index=1,
            s3_prefix="sample_leases",
            source_extension=".txt",
            size=100,
            last_modified="",
            indexed_at="",
            score=0.9,
        )
    ]
    chat_client = FakeChatClient()
    service = make_service(vector_store=vector_store, chat_client=chat_client)
    history = [
        RAGChatMessage(role="user", content="What is the rent?"),
        RAGChatMessage(role="assistant", content="The rent is 1,500 pounds."),
    ]

    response = service.chat(
        question="When is it due?",
        lease_keys=["sample_leases/lease_a.txt"],
        history=history,
        top_k=5,
    )

    assert vector_store.queries == [
        {
            "embedding": [4.0],
            "top_k": 5,
            "lease_keys": ["sample_leases/lease_a.txt"],
        }
    ]
    assert chat_client.calls[0]["history"] == history
    assert response.answer == "Rent is due on the first day."
    assert response.citations == [
        RAGCitation(
            key="sample_leases/lease_a.txt",
            filename="lease_a.txt",
            snippet="Rent is due on the first day.",
            chunk_index=1,
        )
    ]


def test_embedding_client_requires_embedding_deployment():
    settings = Settings(
        AZURE_OPENAI_API_KEY="test-key",
        AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com",
        AZURE_OPENAI_API_VERSION="2024-08-01-preview",
        AZURE_OPENAI_DEPLOYMENT="chat-deployment",
        AZURE_OPENAI_EMBEDDING_DEPLOYMENT=None,
    )

    client = AzureEmbeddingClient(settings)

    with pytest.raises(RAGConfigurationError, match="AZURE_OPENAI_EMBEDDING_DEPLOYMENT"):
        client.embed_texts(["lease text"])
