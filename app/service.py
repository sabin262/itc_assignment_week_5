from app.llm_client import AzureLeaseLLMClient
from app.schemas import CompareResponse, SummariseResponse, VerificationStatus


class LeaseSummariserService:
    def __init__(self, llm_client: AzureLeaseLLMClient):
        self._llm_client = llm_client

    def summarise(self, lease_text: str) -> SummariseResponse:
        extraction = self._llm_client.extract(lease_text)
        verification = self._llm_client.verify(lease_text, extraction)
        return SummariseResponse(
            extraction=extraction,
            verification=verification,
            warnings=_build_warnings(verification),
        )

    def compare(self, lease_a: str, lease_b: str) -> CompareResponse:
        lease_a_summary = self.summarise(lease_a)
        lease_b_summary = self.summarise(lease_b)
        comparison = self._llm_client.compare(
            lease_a_summary.extraction,
            lease_b_summary.extraction,
        )
        return CompareResponse(
            lease_a=lease_a_summary,
            lease_b=lease_b_summary,
            comparison=comparison,
        )


def _build_warnings(verification) -> list[str]:
    warnings: list[str] = []
    for check in verification.checks:
        if check.status == VerificationStatus.unsupported:
            warnings.append(
                f"{check.field_name} was flagged as unsupported by the source lease."
            )
    return warnings

