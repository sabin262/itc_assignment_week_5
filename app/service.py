from app.llm_client import AzureLeaseLLMClient
from app.schemas import CompareResponse, SummariseResponse, VerificationStatus
from typing import Any 


class LeaseSummariserService:
    def __init__(self, llm_client: AzureLeaseLLMClient, langfuse: Any | None = None):
        self._llm_client = llm_client
        self._langfuse = langfuse

    def summarise(self, lease_text: str) -> SummariseResponse:
        trace = self._langfuse.trace(name="summarise") if self._langfuse else None
        try:
            extraction = self._llm_client.extract(lease_text, trace=trace)
            verification = self._llm_client.verify(lease_text, extraction, trace=trace)
            return SummariseResponse(
                extraction=extraction,
                verification=verification,
                warnings=_build_warnings(verification),
            )
        finally:
            if self._langfuse:
                self._langfuse.flush()

    def compare(self, lease_a: str, lease_b: str) -> CompareResponse:
        trace = self._langfuse.trace(name="compare") if self._langfuse else None
        try:
            lease_a_summary = self.summarise(lease_a)
            lease_b_summary = self.summarise(lease_b)
            comparison = self._llm_client.compare(
                lease_a_summary.extraction,
                lease_b_summary.extraction,
                trace=trace,
            )
            return CompareResponse(
                lease_a=lease_a_summary,
                lease_b=lease_b_summary,
                comparison=comparison,
            )
        finally:
            if self._langfuse:
                self._langfuse.flush()


def _build_warnings(verification) -> list[str]:
    warnings: list[str] = []
    for check in verification.checks:
        if check.status == VerificationStatus.unsupported:
            warnings.append(
                f"{check.field_name} was flagged as unsupported by the source lease."
            )
    return warnings

