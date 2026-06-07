from io import BytesIO

from docx import Document
from fastapi.testclient import TestClient

from app.document_parser import extract_text_from_file
from app.main import app, get_lease_service_factory
from app.schemas import (
    GuardrailResult,
    LeaseComparison,
    LeaseDifference,
    LeaseExtraction,
    SummariseResponse,
    VerificationItem,
    VerificationStatus,
)


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


def override_service_factory():
    return FakeLeaseService


app.dependency_overrides[get_lease_service_factory] = override_service_factory
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
