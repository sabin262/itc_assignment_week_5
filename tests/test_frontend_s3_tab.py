from contextlib import nullcontext

from frontend import tabs


class RecordingStreamlit:
    def __init__(self) -> None:
        self.calls = []
        self.buttons: dict[str, bool] = {}
        self.selectbox_values: dict[str, object] = {}

    def info(self, text: str) -> None:
        self.calls.append(("info", text))

    def subheader(self, text: str) -> None:
        self.calls.append(("subheader", text))

    def selectbox(self, label: str, options, format_func=None, index=0, key=None):
        self.calls.append(("selectbox", label, key))
        return self.selectbox_values.get(key, options[index])

    def button(self, label: str, **_kwargs) -> bool:
        self.calls.append(("button", label))
        return self.buttons.get(label, False)

    def spinner(self, text: str):
        self.calls.append(("spinner", text))
        return nullcontext()

    def columns(self, count: int):
        self.calls.append(("columns", count))
        return [nullcontext() for _ in range(count)]


def s3_leases() -> list[dict[str, object]]:
    return [
        {
            "key": "sample_leases/valid_lease_a.txt",
            "filename": "valid_lease_a.txt",
            "size": 100,
            "last_modified": None,
        },
        {
            "key": "sample_leases/valid_lease_b.txt",
            "filename": "valid_lease_b.txt",
            "size": 100,
            "last_modified": None,
        },
    ]


def test_s3_tab_handles_empty_list(monkeypatch):
    fake_st = RecordingStreamlit()
    monkeypatch.setattr(tabs, "st", fake_st)
    monkeypatch.setattr(tabs, "call_api_get", lambda path: [])

    tabs.render_s3_leases_tab()

    assert ("info", "No S3 lease files were returned.") in fake_st.calls


def test_s3_tab_summarise_sends_selected_key(monkeypatch):
    fake_st = RecordingStreamlit()
    fake_st.buttons["Summarise S3 Lease"] = True
    monkeypatch.setattr(tabs, "st", fake_st)
    monkeypatch.setattr(tabs, "call_api_get", lambda path: s3_leases())

    calls = []

    def fake_call_api(path, payload):
        calls.append((path, payload))
        return {"ok": True}

    monkeypatch.setattr(tabs, "call_api", fake_call_api)
    monkeypatch.setattr(
        tabs,
        "render_summary_response",
        lambda response, heading="Result": fake_st.calls.append(("summary", heading)),
    )

    tabs.render_s3_leases_tab()

    assert calls == [
        ("/summarise-s3", {"key": "sample_leases/valid_lease_a.txt"})
    ]
    assert ("summary", "S3 Summary") in fake_st.calls


def test_s3_tab_compare_sends_selected_keys(monkeypatch):
    leases = s3_leases()
    fake_st = RecordingStreamlit()
    fake_st.buttons["Compare S3 Leases"] = True
    fake_st.selectbox_values["s3_compare_lease_a"] = leases[0]
    fake_st.selectbox_values["s3_compare_lease_b"] = leases[1]
    monkeypatch.setattr(tabs, "st", fake_st)
    monkeypatch.setattr(tabs, "call_api_get", lambda path: leases)

    calls = []

    def fake_call_api(path, payload):
        calls.append((path, payload))
        return {"ok": True}

    monkeypatch.setattr(tabs, "call_api", fake_call_api)
    monkeypatch.setattr(
        tabs,
        "render_compare_response",
        lambda response: fake_st.calls.append(("compare", response)),
    )

    tabs.render_s3_leases_tab()

    assert calls == [
        (
            "/compare-s3",
            {
                "lease_a_key": "sample_leases/valid_lease_a.txt",
                "lease_b_key": "sample_leases/valid_lease_b.txt",
            },
        )
    ]
    assert ("compare", {"ok": True}) in fake_st.calls
