from contextlib import nullcontext

from frontend import renderers


class RecordingStreamlit:
    def __init__(self) -> None:
        self.calls = []

    def subheader(self, text: str) -> None:
        self.calls.append(("subheader", text))

    def columns(self, count: int | list[int]):
        self.calls.append(("columns", count))
        column_count = count if isinstance(count, int) else len(count)
        return [nullcontext() for _ in range(column_count)]

    def download_button(self, **kwargs) -> None:
        self.calls.append(("download_button", kwargs.get("label")))

    def markdown(self, text: str) -> None:
        self.calls.append(("markdown", text))

    def write(self, value) -> None:
        self.calls.append(("write", value))

    def warning(self, text: str) -> None:
        self.calls.append(("warning", text))

    def success(self, text: str) -> None:
        self.calls.append(("success", text))

    def info(self, text: str) -> None:
        self.calls.append(("info", text))

    def dataframe(self, data, **kwargs) -> None:
        self.calls.append(("dataframe", data.to_dict("records")))

    def expander(self, label: str):
        self.calls.append(("expander", label))
        return nullcontext()

    def json(self, value) -> None:
        self.calls.append(("json", value))


def test_summary_response_renders_grounding_before_extraction(monkeypatch):
    fake_st = RecordingStreamlit()
    monkeypatch.setattr(renderers, "st", fake_st)

    renderers.render_summary_response(
        {
            "extraction": {
                "tenant_name": "Alex Rivera",
                "plain_english_summary": "Alex rents the property for one year.",
            },
            "verification": {
                "overall_supported": False,
                "checks": [
                    {
                        "field_name": "tenant_name",
                        "status": "unsupported",
                        "evidence": None,
                        "explanation": "Not found in source text.",
                    }
                ],
            },
            "warnings": ["tenant_name was flagged as unsupported by the source lease."],
        }
    )

    dataframe_calls = [payload for name, payload in fake_st.calls if name == "dataframe"]

    assert ("markdown", "#### Grounding Check") in fake_st.calls
    assert dataframe_calls[0][0]["field"] == "tenant_name"
    assert dataframe_calls[1][0]["field"] == "Tenant"
    warning_call = (
        "warning",
        "- tenant_name was flagged as unsupported by the source lease.",
    )
    summary_heading_call = ("markdown", "#### Plain-English Summary")
    assert fake_st.calls.index(warning_call) < fake_st.calls.index(summary_heading_call)
