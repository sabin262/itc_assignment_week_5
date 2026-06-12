from io import BytesIO

from docx import Document
from fastapi.testclient import TestClient

from app.document_parser import extract_text_from_file
from app.main import (
    app,
    get_chat_history_store_factory,
    get_lease_service_factory,
    get_rag_service_factory,
    get_s3_storage_factory,
)
from app.chat_history import ChatHistoryConfigurationError, ChatHistoryNotFoundError
from app.chat_history import ChatHistoryError
from app.rag_service import (
    RAGConfigurationError,
    RAGInvalidKeyError,
    RAGLeaseNotIndexedError,
)
from app.schemas import (
    GuardrailResult,
    LeaseComparison,
    LeaseDifference,
    LeaseExtraction,
    RAGChatResponse,
    RAGChatSessionListResponse,
    RAGChatSessionResponse,
    RAGChatSessionSummary,
    RAGChatStoredMessage,
    RAGCitation,
    RAGIndexResponse,
    RAGSearchMatch,
    RAGSearchResponse,
    RAGStatusResponse,
    S3LeaseFile,
    SummariseResponse,
    VerificationItem,
    VerificationStatus,
)
from app.s3_storage import S3InvalidKeyError, S3ObjectNotFoundError


def make_lease_text(marker: str = "standard") -> str:
    base = (
        "This residential lease agreement is made between tenant Alex Rivera and "
        "landlord Morgan Properties for the apartment at 12 Garden Street, London. "
        "The tenancy starts on 1 January 2026 and ends on 31 December 2026. "
        "Monthly rent is 1,500 pounds and must be paid on the first day of each month. "
        "The tenant paid a security deposit of 1,500 pounds. "
        "The tenant must keep the premises clean, report maintenance issues promptly, "
        "avoid excessive noise, follow guest rules, and keep pets only with written consent. "
        "The landlord must complete necessary repairs, maintain building systems, and give "
        "at least twenty four hours notice before entering except in emergencies. "
        "Either party must give two months notice before vacating at the end of the term. "
    )
    return (base + f"This clause marker is {marker}. ") * 2


def make_docx_bytes(text: str) -> bytes:
    document = Document()
    for paragraph in text.split(". "):
        document.add_paragraph(paragraph)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


class FakeLeaseService:
    def summarise(self, lease_text: str) -> SummariseResponse:
        extraction = LeaseExtraction(
            tenant_name="Alex Rivera",
            landlord_name=None,
            property_address="12 Garden Street, London",
            lease_start_date="1 January 2026",
            lease_end_date="31 December 2026",
            monthly_rent_amount="1,500 pounds",
            rent_payment_due_date="first day of each month",
            security_deposit_amount="1,500 pounds",
            notice_period_to_vacate="two months",
            tenant_obligations=["Keep the premises clean."],
            landlord_obligations=["Give notice before entering."],
            unusual_clauses=None,
            plain_english_summary="Alex rents the property for one year.",
        )
        verification = GuardrailResult(
            overall_supported=False,
            checks=[
                VerificationItem(
                    field_name="tenant_name",
                    status=VerificationStatus.supported,
                    extracted_value="Alex Rivera",
                    evidence="tenant Alex Rivera",
                ),
                VerificationItem(
                    field_name="landlord_name",
                    status=VerificationStatus.missing_from_extraction,
                    extracted_value=None,
                    explanation="No landlord name extracted.",
                ),
                VerificationItem(
                    field_name="property_address",
                    status=VerificationStatus.unsupported,
                    extracted_value="12 Garden Street, London",
                    explanation="Address was not found in the provided text.",
                ),
            ],
        )
        return SummariseResponse(
            extraction=extraction,
            verification=verification,
            warnings=["property_address was flagged as unsupported by the source lease."],
        )

    def compare(self, lease_a: str, lease_b: str):
        return {
            "lease_a": self.summarise(lease_a),
            "lease_b": self.summarise(lease_b),
            "comparison": LeaseComparison(
                summary="Lease B has a higher rent and longer notice period.",
                differences=[
                    LeaseDifference(
                        field_name="monthly_rent_amount",
                        lease_a_value="1,500 pounds",
                        lease_b_value="1,700 pounds",
                        difference="Lease B costs 200 pounds more per month.",
                        practical_impact="Lease A is cheaper month to month.",
                    ),
                    LeaseDifference(
                        field_name="notice_period_to_vacate",
                        lease_a_value="two months",
                        lease_b_value="three months",
                        difference="Lease B requires one extra month of notice.",
                        practical_impact="Lease B gives the tenant less flexibility.",
                    ),
                ],
            ),
        }


class FakeS3LeaseStorage:
    def __init__(self) -> None:
        self._files = {
            "sample_leases/valid_lease_a.txt": make_lease_text("s3-a").encode("utf-8"),
            "sample_leases/valid_lease_b.txt": make_lease_text("s3-b").encode("utf-8"),
        }

    def list_lease_files(self) -> list[S3LeaseFile]:
        return [
            S3LeaseFile(
                key="sample_leases/valid_lease_a.txt",
                filename="valid_lease_a.txt",
                size=len(self._files["sample_leases/valid_lease_a.txt"]),
                last_modified=None,
            ),
            S3LeaseFile(
                key="sample_leases/valid_lease_b.txt",
                filename="valid_lease_b.txt",
                size=len(self._files["sample_leases/valid_lease_b.txt"]),
                last_modified=None,
            ),
        ]

    def get_file(self, key: str) -> tuple[str, bytes]:
        if not key.startswith("sample_leases/"):
            raise S3InvalidKeyError("S3 key must be inside the configured S3_PREFIX.")
        if not key.endswith((".txt", ".pdf", ".docx")):
            raise S3InvalidKeyError("Unsupported file type. Use .docx, .pdf, .txt.")
        if key not in self._files:
            raise S3ObjectNotFoundError(f"S3 lease file was not found: {key}")
        return key.rsplit("/", 1)[-1], self._files[key]


class FakeRAGService:
    def status(self) -> RAGStatusResponse:
        return RAGStatusResponse(
            collection_name="lease_chunks",
            indexed_lease_count=2,
            chunk_count=4,
            last_indexed_at="2026-01-01T00:00:00+00:00",
            indexed_summary_count=2,
        )

    def index_s3_leases(
        self,
        s3_storage,
        lease_service,
        progress_callback=None,
    ) -> RAGIndexResponse:
        assert len(s3_storage.list_lease_files()) == 2
        assert lease_service.summarise(make_lease_text()).extraction.tenant_name == "Alex Rivera"
        if progress_callback:
            progress_callback(1, 3, "Processing valid_lease_a.txt.", "sample_leases/valid_lease_a.txt")
            progress_callback(3, 3, "Indexing completed.", None)
        return RAGIndexResponse(
            indexed_lease_count=2,
            indexed_chunk_count=4,
            skipped_files=[],
            failed_files=[],
            summarised_lease_count=2,
            summary_failed_files=[],
        )

    def search(self, question: str, top_k: int) -> RAGSearchResponse:
        assert question == "What is the monthly rent?"
        assert top_k == 3
        return RAGSearchResponse(
            question=question,
            matches=[
                RAGSearchMatch(
                    key="sample_leases/valid_lease_a.txt",
                    filename="valid_lease_a.txt",
                    snippet="Monthly rent is 1,500 pounds.",
                    score=0.91,
                    chunk_index=0,
                )
            ],
        )

    def lease_text_from_index(self, key: str) -> str:
        if not key.startswith("sample_leases/"):
            raise RAGInvalidKeyError("S3 key must be inside the configured S3_PREFIX.")
        if key == "sample_leases/missing.txt":
            raise RAGLeaseNotIndexedError(f"Indexed lease was not found: {key}")
        return make_lease_text(f"indexed-{key}")

    def chat(self, question: str, lease_keys, history, top_k: int) -> RAGChatResponse:
        assert question == "When is rent due?"
        assert lease_keys == ["sample_leases/valid_lease_a.txt"]
        assert top_k == 5
        assert [item.content for item in history] == ["What is the rent?"]
        return RAGChatResponse(
            question=question,
            answer="Rent is due on the first day of each month.",
            citations=[
                RAGCitation(
                    key="sample_leases/valid_lease_a.txt",
                    filename="valid_lease_a.txt",
                    snippet="Rent must be paid on the first day of each month.",
                    chunk_index=0,
                )
            ],
        )


class FakeChatHistoryStore:
    sessions: dict[str, dict[str, object]] = {}
    fail_configuration = False
    fail_save = False

    @classmethod
    def reset(cls) -> None:
        cls.sessions = {}
        cls.fail_configuration = False
        cls.fail_save = False

    def _raise_if_unconfigured(self) -> None:
        if self.fail_configuration:
            raise ChatHistoryConfigurationError(
                "CHAT_HISTORY_TABLE_NAME is not configured."
            )

    def save_exchange(
        self,
        *,
        session_id,
        question: str,
        lease_keys: list[str],
        response: RAGChatResponse,
    ) -> tuple[str, str]:
        self._raise_if_unconfigured()
        if self.fail_save:
            raise ChatHistoryError("Could not save chat history: test failure.")
        session_id = session_id or f"session-{len(self.sessions) + 1}"
        saved_at = f"2026-01-01T00:00:0{len(self.sessions) + 1}+00:00"
        session = self.sessions.setdefault(
            session_id,
            {
                "session_id": session_id,
                "title": question[:80],
                "lease_keys": lease_keys,
                "created_at": saved_at,
                "updated_at": saved_at,
                "messages": [],
            },
        )
        session["lease_keys"] = lease_keys
        session["updated_at"] = saved_at
        session["messages"].extend(
            [
                RAGChatStoredMessage(
                    role="user",
                    content=question,
                    created_at=saved_at,
                ),
                RAGChatStoredMessage(
                    role="assistant",
                    content=response.answer,
                    citations=response.citations,
                    verification=response.verification,
                    warnings=response.warnings,
                    created_at=saved_at,
                ),
            ]
        )
        return session_id, saved_at

    def list_sessions(self) -> RAGChatSessionListResponse:
        self._raise_if_unconfigured()
        sessions = sorted(
            self.sessions.values(),
            key=lambda session: str(session["updated_at"]),
            reverse=True,
        )
        return RAGChatSessionListResponse(
            sessions=[
                RAGChatSessionSummary(
                    session_id=str(session["session_id"]),
                    title=str(session["title"]),
                    lease_keys=list(session["lease_keys"]),
                    message_count=len(session["messages"]),
                    created_at=str(session["created_at"]),
                    updated_at=str(session["updated_at"]),
                )
                for session in sessions
            ]
        )

    def get_session(self, session_id: str) -> RAGChatSessionResponse:
        self._raise_if_unconfigured()
        session = self.sessions.get(session_id)
        if session is None:
            raise ChatHistoryNotFoundError(
                f"Saved chat session was not found: {session_id}"
            )
        return RAGChatSessionResponse(
            session_id=str(session["session_id"]),
            title=str(session["title"]),
            lease_keys=list(session["lease_keys"]),
            message_count=len(session["messages"]),
            created_at=str(session["created_at"]),
            updated_at=str(session["updated_at"]),
            messages=list(session["messages"]),
        )

    def delete_session(self, session_id: str) -> None:
        self._raise_if_unconfigured()
        if session_id not in self.sessions:
            raise ChatHistoryNotFoundError(
                f"Saved chat session was not found: {session_id}"
            )
        del self.sessions[session_id]


def override_service_factory():
    return FakeLeaseService


def override_s3_storage_factory():
    return FakeS3LeaseStorage


def override_rag_service_factory():
    return FakeRAGService


def override_chat_history_store_factory():
    return FakeChatHistoryStore


app.dependency_overrides[get_lease_service_factory] = override_service_factory
app.dependency_overrides[get_s3_storage_factory] = override_s3_storage_factory
app.dependency_overrides[get_rag_service_factory] = override_rag_service_factory
app.dependency_overrides[get_chat_history_store_factory] = (
    override_chat_history_store_factory
)
client = TestClient(app)


def test_summarise_rejects_short_lease_text():
    response = client.post("/summarise-text", json={"lease_text": "Too short."})

    assert response.status_code == 422
    assert "Lease text must contain at least 100 words." in response.text


def test_summarise_returns_extraction_and_guardrail_warnings():
    response = client.post("/summarise-text", json={"lease_text": make_lease_text()})

    assert response.status_code == 200
    body = response.json()
    assert body["extraction"]["tenant_name"] == "Alex Rivera"
    assert body["extraction"]["landlord_name"] is None
    assert body["warnings"] == [
        "property_address was flagged as unsupported by the source lease."
    ]


def test_summarise_file_accepts_text_upload():
    response = client.post(
        "/summarise",
        files={
            "file": (
                "lease.txt",
                make_lease_text().encode("utf-8"),
                "text/plain",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["extraction"]["tenant_name"] == "Alex Rivera"


def test_compare_files_accepts_mixed_docx_and_text_uploads():
    response = client.post(
        "/compare",
        files={
            "lease_a": (
                "lease_a.docx",
                make_docx_bytes(make_lease_text("a")),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            "lease_b": (
                "lease_b.txt",
                make_lease_text("b").encode("utf-8"),
                "text/plain",
            ),
        },
    )

    assert response.status_code == 200
    assert response.json()["comparison"]["differences"][0]["field_name"] == "monthly_rent_amount"


def test_summarise_file_rejects_unsupported_file_type():
    response = client.post(
        "/summarise",
        files={"file": ("lease.rtf", b"unsupported content", "application/rtf")},
    )

    assert response.status_code == 422
    assert "Unsupported file type" in response.text


def test_docx_parser_extracts_paragraph_text():
    extracted = extract_text_from_file("lease.docx", make_docx_bytes(make_lease_text()))

    assert "tenant Alex Rivera" in extracted
    assert "landlord Morgan Properties" in extracted


def test_openapi_docs_available_for_smoke_check():
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["info"]["title"] == "Smart Lease Summariser"


def test_compare_rejects_if_either_lease_is_short():
    response = client.post(
        "/compare-text",
        json={"lease_a": make_lease_text("a"), "lease_b": "Too short."},
    )

    assert response.status_code == 422
    assert "Lease text must contain at least 100 words." in response.text


def test_compare_returns_structured_differences():
    response = client.post(
        "/compare-text",
        json={"lease_a": make_lease_text("a"), "lease_b": make_lease_text("b")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["comparison"]["summary"] == "Lease B has a higher rent and longer notice period."
    assert body["comparison"]["differences"][0]["field_name"] == "monthly_rent_amount"


def test_s3_leases_lists_available_files():
    response = client.get("/s3/leases")

    assert response.status_code == 200
    body = response.json()
    assert [item["key"] for item in body] == [
        "sample_leases/valid_lease_a.txt",
        "sample_leases/valid_lease_b.txt",
    ]


def test_summarise_s3_lease():
    response = client.post(
        "/summarise-s3",
        json={"key": "sample_leases/valid_lease_a.txt"},
    )

    assert response.status_code == 200
    assert response.json()["extraction"]["tenant_name"] == "Alex Rivera"


def test_compare_s3_leases():
    response = client.post(
        "/compare-s3",
        json={
            "lease_a_key": "sample_leases/valid_lease_a.txt",
            "lease_b_key": "sample_leases/valid_lease_b.txt",
        },
    )

    assert response.status_code == 200
    assert response.json()["comparison"]["differences"][0]["field_name"] == "monthly_rent_amount"


def test_summarise_indexed_lease():
    response = client.post(
        "/summarise-indexed",
        json={"key": "sample_leases/valid_lease_a.txt"},
    )

    assert response.status_code == 200
    assert response.json()["extraction"]["tenant_name"] == "Alex Rivera"


def test_compare_indexed_leases():
    response = client.post(
        "/compare-indexed",
        json={
            "lease_a_key": "sample_leases/valid_lease_a.txt",
            "lease_b_key": "sample_leases/valid_lease_b.txt",
        },
    )

    assert response.status_code == 200
    assert response.json()["comparison"]["differences"][0]["field_name"] == "monthly_rent_amount"


def test_summarise_s3_returns_404_when_key_missing():
    response = client.post(
        "/summarise-s3",
        json={"key": "sample_leases/missing.txt"},
    )

    assert response.status_code == 404
    assert "S3 lease file was not found" in response.text


def test_summarise_s3_rejects_unsupported_file_type():
    response = client.post(
        "/summarise-s3",
        json={"key": "sample_leases/lease.rtf"},
    )

    assert response.status_code == 422
    assert "Unsupported file type" in response.text


def test_summarise_s3_blocks_keys_outside_configured_prefix():
    response = client.post(
        "/summarise-s3",
        json={"key": "other_prefix/valid_lease_a.txt"},
    )

    assert response.status_code == 422
    assert "configured S3_PREFIX" in response.text


def test_summarise_indexed_returns_404_when_lease_not_indexed():
    response = client.post(
        "/summarise-indexed",
        json={"key": "sample_leases/missing.txt"},
    )

    assert response.status_code == 404
    assert "Indexed lease was not found" in response.text


def test_summarise_indexed_blocks_keys_outside_configured_prefix():
    response = client.post(
        "/summarise-indexed",
        json={"key": "other_prefix/valid_lease_a.txt"},
    )

    assert response.status_code == 422
    assert "configured S3_PREFIX" in response.text


def test_rag_status_returns_index_state():
    response = client.get("/rag/status")

    assert response.status_code == 200
    body = response.json()
    assert body["collection_name"] == "lease_chunks"
    assert body["indexed_lease_count"] == 2
    assert body["chunk_count"] == 4
    assert body["indexed_summary_count"] == 2


def test_rag_index_indexes_s3_leases():
    response = client.post("/rag/index")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"running", "completed"}
    assert body["job_id"]
    assert "progress_percent" in body

    status_response = client.get("/rag/index/status")
    assert status_response.status_code == 200
    status_body = status_response.json()
    assert status_body["status"] == "completed"
    assert status_body["progress_percent"] == 1.0
    assert status_body["message"] == "Indexing completed."
    result = status_body["result"]
    assert result["indexed_lease_count"] == 2
    assert result["indexed_chunk_count"] == 4
    assert result["summarised_lease_count"] == 2
    assert result["summary_failed_files"] == []


def test_rag_search_returns_matching_lease_snippets():
    response = client.post(
        "/rag/search",
        json={"question": "What is the monthly rent?", "top_k": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["matches"][0]["key"] == "sample_leases/valid_lease_a.txt"
    assert body["matches"][0]["snippet"] == "Monthly rent is 1,500 pounds."


def test_rag_chat_filters_to_selected_lease_and_accepts_history():
    FakeChatHistoryStore.reset()
    response = client.post(
        "/rag/chat",
        json={
            "question": "When is rent due?",
            "lease_keys": ["sample_leases/valid_lease_a.txt"],
            "history": [{"role": "user", "content": "What is the rent?"}],
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Rent is due on the first day of each month."
    assert body["citations"][0]["key"] == "sample_leases/valid_lease_a.txt"
    assert body["session_id"] == "session-1"
    assert body["saved_at"] == "2026-01-01T00:00:01+00:00"


def test_rag_chat_returns_answer_when_history_save_fails():
    FakeChatHistoryStore.reset()
    FakeChatHistoryStore.fail_save = True
    try:
        response = client.post(
            "/rag/chat",
            json={
                "question": "When is rent due?",
                "lease_keys": ["sample_leases/valid_lease_a.txt"],
                "history": [{"role": "user", "content": "What is the rent?"}],
                "top_k": 5,
            },
        )
    finally:
        FakeChatHistoryStore.reset()

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Rent is due on the first day of each month."
    assert body["session_id"] is None
    assert body["saved_at"] is None
    assert body["warnings"] == [
        "Chat history could not be saved: Could not save chat history: test failure."
    ]


def test_rag_chat_appends_to_existing_saved_session():
    FakeChatHistoryStore.reset()
    first_response = client.post(
        "/rag/chat",
        json={
            "question": "When is rent due?",
            "lease_keys": ["sample_leases/valid_lease_a.txt"],
            "history": [{"role": "user", "content": "What is the rent?"}],
            "top_k": 5,
        },
    )
    session_id = first_response.json()["session_id"]

    second_response = client.post(
        "/rag/chat",
        json={
            "session_id": session_id,
            "question": "When is rent due?",
            "lease_keys": ["sample_leases/valid_lease_a.txt"],
            "history": [{"role": "user", "content": "What is the rent?"}],
            "top_k": 5,
        },
    )

    assert second_response.status_code == 200
    assert second_response.json()["session_id"] == session_id
    session = FakeChatHistoryStore.sessions[session_id]
    assert len(session["messages"]) == 4


def test_rag_chat_sessions_can_be_listed_loaded_and_deleted():
    FakeChatHistoryStore.reset()
    response = client.post(
        "/rag/chat",
        json={
            "question": "When is rent due?",
            "lease_keys": ["sample_leases/valid_lease_a.txt"],
            "history": [{"role": "user", "content": "What is the rent?"}],
            "top_k": 5,
        },
    )
    session_id = response.json()["session_id"]

    list_response = client.get("/rag/chat/sessions")
    assert list_response.status_code == 200
    assert list_response.json()["sessions"][0]["session_id"] == session_id

    load_response = client.get(f"/rag/chat/sessions/{session_id}")
    assert load_response.status_code == 200
    loaded = load_response.json()
    assert loaded["lease_keys"] == ["sample_leases/valid_lease_a.txt"]
    assert [message["role"] for message in loaded["messages"]] == [
        "user",
        "assistant",
    ]
    assert loaded["messages"][1]["citations"][0]["key"] == "sample_leases/valid_lease_a.txt"

    delete_response = client.delete(f"/rag/chat/sessions/{session_id}")
    assert delete_response.status_code == 204
    missing_response = client.get(f"/rag/chat/sessions/{session_id}")
    assert missing_response.status_code == 404


def test_rag_chat_history_missing_table_returns_clear_error():
    FakeChatHistoryStore.reset()
    FakeChatHistoryStore.fail_configuration = True

    try:
        response = client.get("/rag/chat/sessions")
    finally:
        FakeChatHistoryStore.reset()

    assert response.status_code == 500
    assert "CHAT_HISTORY_TABLE_NAME" in response.text


def test_rag_config_errors_return_clear_500_response():
    def override_broken_rag_service_factory():
        def broken_factory():
            raise RAGConfigurationError(
                "Azure OpenAI embedding deployment is not configured."
            )

        return broken_factory

    app.dependency_overrides[get_rag_service_factory] = (
        override_broken_rag_service_factory
    )
    try:
        response = client.get("/rag/status")
    finally:
        app.dependency_overrides[get_rag_service_factory] = (
            override_rag_service_factory
        )

    assert response.status_code == 500
    assert "embedding deployment is not configured" in response.text
