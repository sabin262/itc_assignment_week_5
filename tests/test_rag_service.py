from datetime import UTC, datetime

import pytest

from app.config import Settings
from app.rag_service import (
    AzureEmbeddingClient,
    CHUNK_WORDS,
    LeaseChunk,
    LeaseSummaryRecord,
    LeaseSummaryStore,
    RAGConfigurationError,
    RAGInvalidKeyError,
    RAGLeaseNotIndexedError,
    RAGService,
    split_text_into_chunks,
)
from app.schemas import (
    GuardrailResult,
    LeaseExtraction,
    RAGChatAnswer,
    RAGChatMessage,
    RAGCitation,
    RAGStatusResponse,
    S3LeaseFile,
    SummariseResponse,
    VerificationItem,
    VerificationStatus,
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
        self.fail_on_call = False

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.fail_on_call:
            raise AssertionError("Embedding client should not have been called.")
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
        summaries: list[LeaseSummaryRecord],
    ) -> RAGChatAnswer:
        self.calls.append(
            {
                "question": question,
                "history": history,
                "chunks": chunks,
                "summaries": summaries,
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


class FakeSummaryStore:
    def __init__(self) -> None:
        self.reset_prefixes: list[str] = []
        self.records: list[LeaseSummaryRecord] = []

    def reset_prefix(self, prefix: str) -> None:
        self.reset_prefixes.append(prefix)
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


class FakeLeaseService:
    def __init__(self, failed_markers: set[str] | None = None) -> None:
        self.failed_markers = failed_markers or set()
        self.calls: list[str] = []

    def summarise(self, text: str) -> SummariseResponse:
        self.calls.append(text)
        for marker in self.failed_markers:
            if marker in text:
                raise RuntimeError("Summary failed.")
        rent = "1,500 pounds"
        if "cheap" in text:
            rent = "1,250 pounds"
        if "expensive" in text:
            rent = "1,875 pounds"
        return summary_response(rent)


def lease_text(word_count: int, marker: str) -> str:
    return " ".join(f"{marker}{index}" for index in range(word_count))


def lease_sentence_text(marker: str, rent: str = "1,500 pounds") -> str:
    base = (
        "This residential lease is between tenant Alex Rivera and landlord Morgan "
        "Properties for 12 Garden Street. The lease starts on 1 January 2026 and "
        f"monthly rent is {rent}. The rent is due on the first day of each month. "
        "The security deposit is 1,500 pounds. The tenant must keep the home clean "
        "and report maintenance problems. The landlord must make repairs and give "
        "notice before entry. The tenant may not create excessive noise and must "
        "follow guest rules. Either party must give two months notice before the "
        "end of the tenancy. "
    )
    return (base + f"This marker is {marker}. ") * 2


def summary_response(rent: str | None = "1,500 pounds") -> SummariseResponse:
    return SummariseResponse(
        extraction=LeaseExtraction(
            tenant_name="Alex Rivera",
            landlord_name="Morgan Properties",
            property_address="12 Garden Street",
            lease_start_date="1 January 2026",
            lease_end_date="31 December 2026",
            monthly_rent_amount=rent,
            rent_payment_due_date="first day of each month",
            security_deposit_amount="1,500 pounds",
            notice_period_to_vacate="two months",
            tenant_obligations=["Keep the home clean."],
            landlord_obligations=["Make repairs."],
            unusual_clauses=None,
            plain_english_summary="Alex rents the property for one year.",
        ),
        verification=GuardrailResult(
            overall_supported=True,
            checks=[
                VerificationItem(
                    field_name="monthly_rent_amount",
                    status=VerificationStatus.supported,
                    extracted_value=rent,
                    evidence=str(rent) if rent else None,
                )
            ],
        ),
        warnings=[],
    )


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


def summary_record(
    key: str,
    rent: str | None,
    numeric_rent: float | None,
) -> LeaseSummaryRecord:
    return LeaseSummaryRecord(
        key=key,
        filename=key.rsplit("/", 1)[-1],
        s3_prefix="sample_leases",
        size=100,
        last_modified="",
        indexed_at="2026-01-01T00:00:00+00:00",
        summary=summary_response(rent),
        monthly_rent_amount_numeric=numeric_rent,
    )


def make_service(
    vector_store: FakeVectorStore | None = None,
    embedding_client: FakeEmbeddingClient | None = None,
    chat_client: FakeChatClient | None = None,
    summary_store: FakeSummaryStore | None = None,
) -> RAGService:
    return RAGService(
        vector_store=vector_store or FakeVectorStore(),
        embedding_client=embedding_client or FakeEmbeddingClient(),
        chat_client=chat_client or FakeChatClient(),
        s3_prefix="sample_leases",
        summary_store=summary_store or FakeSummaryStore(),
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
    summary_store = FakeSummaryStore()
    lease_service = FakeLeaseService()
    service = make_service(
        vector_store=vector_store,
        embedding_client=embedding_client,
        summary_store=summary_store,
    )
    s3_storage = FakeS3Storage(
        {
            "sample_leases/lease_a.txt": lease_text(CHUNK_WORDS + 50, "a"),
            "sample_leases/lease_b.txt": lease_sentence_text("cheap", "1,250 pounds"),
        }
    )
    progress_updates = []

    response = service.index_s3_leases(
        s3_storage,
        lease_service,
        progress_callback=lambda current, total, message, key: progress_updates.append(
            (current, total, message, key)
        ),
    )

    assert vector_store.reset_prefixes == ["sample_leases"]
    assert summary_store.reset_prefixes == ["sample_leases"]
    assert response.indexed_lease_count == 2
    assert response.indexed_chunk_count == len(vector_store.chunks)
    assert response.failed_files == []
    assert response.summarised_lease_count == 2
    assert response.summary_failed_files == []
    assert len(vector_store.embeddings) == len(vector_store.chunks)
    assert [chunk.key for chunk in vector_store.chunks].count(
        "sample_leases/lease_a.txt"
    ) == 2
    assert {record.key for record in summary_store.records} == {
        "sample_leases/lease_a.txt",
        "sample_leases/lease_b.txt",
    }
    assert summary_store.records[1].monthly_rent_amount_numeric == 1250
    assert all(chunk.s3_prefix == "sample_leases" for chunk in vector_store.chunks)
    assert sum(len(batch) for batch in embedding_client.batches) == len(
        vector_store.chunks
    )
    assert progress_updates[0] == (
        0,
        4,
        "Preparing indexed lease store.",
        None,
    )
    assert progress_updates[-1] == (4, 4, "Indexing completed.", None)
    assert any(update[3] == "sample_leases/lease_a.txt" for update in progress_updates)


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


def test_indexing_records_summary_failures_without_blocking_chunk_indexing():
    vector_store = FakeVectorStore()
    summary_store = FakeSummaryStore()
    lease_service = FakeLeaseService(failed_markers={"bad_summary"})
    service = make_service(vector_store=vector_store, summary_store=summary_store)
    s3_storage = FakeS3Storage(
        {
            "sample_leases/lease_a.txt": lease_sentence_text("cheap", "1,250 pounds"),
            "sample_leases/lease_b.txt": lease_sentence_text("bad_summary"),
        }
    )

    response = service.index_s3_leases(s3_storage, lease_service)

    assert response.indexed_lease_count == 2
    assert len(vector_store.chunks) == 2
    assert response.summarised_lease_count == 1
    assert response.summary_failed_files == ["sample_leases/lease_b.txt"]
    assert [record.key for record in summary_store.records] == [
        "sample_leases/lease_a.txt"
    ]


def test_summary_store_persists_records_and_resets_by_prefix(tmp_path):
    store = LeaseSummaryStore(str(tmp_path))
    store.upsert_summaries(
        [
            summary_record("sample_leases/lease_a.txt", "1,250 pounds", 1250),
            LeaseSummaryRecord(
                key="other_prefix/lease_b.txt",
                filename="lease_b.txt",
                s3_prefix="other_prefix",
                size=100,
                last_modified="",
                indexed_at="2026-01-01T00:00:00+00:00",
                summary=summary_response("1,875 pounds"),
                monthly_rent_amount_numeric=1875,
            ),
        ]
    )

    reloaded = LeaseSummaryStore(str(tmp_path))

    assert reloaded.count("sample_leases") == 1
    assert reloaded.list_summaries("sample_leases")[0].summary.extraction.monthly_rent_amount == "1,250 pounds"

    reloaded.reset_prefix("sample_leases")

    assert reloaded.count("sample_leases") == 0
    assert reloaded.count("other_prefix") == 1


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


def test_chat_answers_cheapest_rent_from_summaries_without_vector_query():
    vector_store = FakeVectorStore()
    embedding_client = FakeEmbeddingClient()
    embedding_client.fail_on_call = True
    chat_client = FakeChatClient()
    summary_store = FakeSummaryStore()
    summary_store.records = [
        summary_record("sample_leases/lease_a.txt", "1,875 pounds", 1875),
        summary_record("sample_leases/lease_b.txt", "1,250 pounds", 1250),
    ]
    service = make_service(
        vector_store=vector_store,
        embedding_client=embedding_client,
        chat_client=chat_client,
        summary_store=summary_store,
    )

    response = service.chat(
        question="What is the cheapest rent in the leases?",
        lease_keys=[],
        history=[],
        top_k=5,
    )

    assert "1,250 pounds" in response.answer
    assert "lease_b.txt" in response.answer
    assert response.citations == [
        RAGCitation(
            key="sample_leases/lease_b.txt",
            filename="lease_b.txt",
            snippet=(
                "monthly_rent_amount: 1,250 pounds; property_address: "
                "12 Garden Street; lease_start_date: 1 January 2026; "
                "lease_end_date: 31 December 2026; security_deposit_amount: "
                "1,500 pounds"
            ),
            chunk_index=-1,
            source_type="summary",
        )
    ]
    assert vector_store.queries == []
    assert chat_client.calls == []


def test_chat_cheapest_rent_respects_selected_lease_filter():
    summary_store = FakeSummaryStore()
    summary_store.records = [
        summary_record("sample_leases/lease_a.txt", "1,875 pounds", 1875),
        summary_record("sample_leases/lease_b.txt", "1,250 pounds", 1250),
    ]
    service = make_service(summary_store=summary_store)

    response = service.chat(
        question="Which selected lease has the lowest rent?",
        lease_keys=["sample_leases/lease_a.txt"],
        history=[],
        top_k=5,
    )

    assert "1,875 pounds" in response.answer
    assert "lease_a.txt" in response.answer


def test_chat_cheapest_rent_discloses_unparseable_summary_values():
    summary_store = FakeSummaryStore()
    summary_store.records = [
        summary_record("sample_leases/lease_a.txt", "1,250 pounds", 1250),
        summary_record("sample_leases/lease_b.txt", None, None),
    ]
    service = make_service(summary_store=summary_store)

    response = service.chat(
        question="What is the cheapest rent?",
        lease_keys=[],
        history=[],
        top_k=5,
    )

    assert "1,250 pounds" in response.answer
    assert "lease_b.txt" in response.answer


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
