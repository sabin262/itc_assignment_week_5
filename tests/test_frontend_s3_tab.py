from contextlib import nullcontext

from frontend import tabs


class RecordingPlaceholder:
    def __init__(self, calls) -> None:
        self.calls = calls

    def markdown(self, text: str) -> None:
        self.calls.append(("placeholder_markdown", text))

    def empty(self) -> None:
        self.calls.append(("placeholder_empty",))


class RecordingStreamlit:
    def __init__(self) -> None:
        self.calls = []
        self.buttons: dict[str, bool] = {}
        self.selectbox_values: dict[str, object] = {}
        self.multiselect_values: dict[str, list[object]] = {}
        self.text_inputs: dict[str, str] = {}
        self.chat_inputs: dict[str, str] = {}
        self.toggles: dict[str, bool] = {}
        self.radios: dict[str, str] = {}
        self.session_state: dict[str, object] = {}

    def info(self, text: str) -> None:
        self.calls.append(("info", text))

    def selectbox(self, label: str, options, format_func=None, index=0, key=None):
        self.calls.append(("selectbox", label, key))
        return self.selectbox_values.get(key, options[index])

    def multiselect(self, label: str, options, format_func=None, key=None):
        self.calls.append(("multiselect", label, key))
        return self.multiselect_values.get(key, [])

    def radio(self, label: str, options, horizontal=False, key=None):
        self.calls.append(("radio", label, key))
        return self.radios.get(key, options[0])

    def button(self, label: str, **_kwargs) -> bool:
        self.calls.append(("button", label))
        return self.buttons.get(label, False)

    def spinner(self, text: str):
        self.calls.append(("spinner", text))
        return nullcontext()

    def columns(self, count: int):
        self.calls.append(("columns", count))
        return [nullcontext() for _ in range(count)]

    def container(self, **kwargs):
        self.calls.append(("container", kwargs))
        return nullcontext()

    def chat_message(self, role: str):
        self.calls.append(("chat_message", role))
        return nullcontext()

    def chat_input(self, placeholder: str, key=None) -> str | None:
        self.calls.append(("chat_input", placeholder, key))
        return self.chat_inputs.get(key)

    def empty(self):
        self.calls.append(("empty",))
        return RecordingPlaceholder(self.calls)

    def metric(self, label: str, value) -> None:
        self.calls.append(("metric", label, value))

    def progress(self, value, text=None) -> None:
        self.calls.append(("progress", value, text))

    def caption(self, text: str) -> None:
        self.calls.append(("caption", text))

    def success(self, text: str) -> None:
        self.calls.append(("success", text))

    def warning(self, text: str) -> None:
        self.calls.append(("warning", text))

    def error(self, text: str) -> None:
        self.calls.append(("error", text))

    def text_input(self, label: str, key=None) -> str:
        self.calls.append(("text_input", label, key))
        return self.text_inputs.get(key, "")

    def toggle(self, label: str, value=False, key=None) -> bool:
        self.calls.append(("toggle", label, key))
        return self.toggles.get(key, value)

    def markdown(self, text: str) -> None:
        self.calls.append(("markdown", text))

    def write(self, value) -> None:
        self.calls.append(("write", value))


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


def test_s3_summarise_tab_handles_empty_list(monkeypatch):
    fake_st = RecordingStreamlit()
    monkeypatch.setattr(tabs, "st", fake_st)
    monkeypatch.setattr(tabs, "call_api_get", lambda path: [])

    tabs.render_s3_summarise_tab()

    assert ("info", "No S3 lease files were returned.") in fake_st.calls


def test_s3_summarise_sends_selected_key(monkeypatch):
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

    tabs.render_s3_summarise_tab()

    assert calls == [
        ("/summarise-s3", {"key": "sample_leases/valid_lease_a.txt"})
    ]
    assert ("summary", "S3 Summary") in fake_st.calls


def test_s3_summarise_indexed_sends_selected_key_to_indexed_endpoint(monkeypatch):
    fake_st = RecordingStreamlit()
    fake_st.radios["s3_summarise_source"] = "Indexed leases"
    fake_st.buttons["Summarise Indexed Lease"] = True
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

    tabs.render_s3_summarise_tab()

    assert calls == [
        ("/summarise-indexed", {"key": "sample_leases/valid_lease_a.txt"})
    ]
    assert ("summary", "Indexed Summary") in fake_st.calls


def test_s3_compare_sends_selected_keys(monkeypatch):
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

    tabs.render_s3_compare_tab()

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


def test_s3_compare_indexed_sends_selected_keys_to_indexed_endpoint(monkeypatch):
    leases = s3_leases()
    fake_st = RecordingStreamlit()
    fake_st.radios["s3_compare_source"] = "Indexed leases"
    fake_st.buttons["Compare Indexed Leases"] = True
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

    tabs.render_s3_compare_tab()

    assert calls == [
        (
            "/compare-indexed",
            {
                "lease_a_key": "sample_leases/valid_lease_a.txt",
                "lease_b_key": "sample_leases/valid_lease_b.txt",
            },
        )
    ]
    assert ("compare", {"ok": True}) in fake_st.calls


def test_s3_index_calls_rag_index(monkeypatch):
    fake_st = RecordingStreamlit()
    fake_st.buttons["Index S3 Leases"] = True
    monkeypatch.setattr(tabs, "st", fake_st)
    def fake_call_api_get(path):
        if path == "/rag/index/status":
            return {"status": "idle"}
        return {
            "collection_name": "lease_chunks",
            "indexed_lease_count": 0,
            "chunk_count": 0,
            "indexed_summary_count": 0,
            "last_indexed_at": None,
        }

    monkeypatch.setattr(tabs, "call_api_get", fake_call_api_get)

    calls = []

    def fake_call_api(path, payload):
        calls.append((path, payload))
        return {
            "job_id": "job-1",
            "status": "completed",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:00:05+00:00",
            "error": None,
            "result": {
                "indexed_lease_count": 2,
                "indexed_chunk_count": 4,
                "skipped_files": [],
                "failed_files": [],
                "summarised_lease_count": 2,
                "summary_failed_files": [],
            },
        }

    monkeypatch.setattr(tabs, "call_api", fake_call_api)

    tabs.render_s3_index_tab()

    assert calls == [("/rag/index", {})]
    assert (
        "success",
        "Indexed 2 leases into 4 chunks and stored 2 summaries.",
    ) in fake_st.calls


def test_s3_index_running_job_shows_progress_bar(monkeypatch):
    fake_st = RecordingStreamlit()
    monkeypatch.setattr(tabs, "st", fake_st)
    monkeypatch.setattr(tabs.time, "sleep", lambda _seconds: None)

    def fake_call_api_get(path):
        if path == "/rag/index/status":
            return {
                "job_id": "job-1",
                "status": "running",
                "started_at": "2026-01-01T00:00:00+00:00",
                "finished_at": None,
                "result": None,
                "error": None,
                "progress_current": 2,
                "progress_total": 5,
                "progress_percent": 0.4,
                "message": "Processing lease_b.txt.",
                "current_key": "sample_leases/lease_b.txt",
            }
        return {
            "collection_name": "lease_chunks",
            "indexed_lease_count": 1,
            "chunk_count": 2,
            "indexed_summary_count": 1,
            "last_indexed_at": None,
        }

    monkeypatch.setattr(tabs, "call_api_get", fake_call_api_get)

    tabs.render_s3_index_tab()

    assert (
        "progress",
        0.4,
        "Processing lease_b.txt. (2/5)",
    ) in fake_st.calls
    assert (
        "caption",
        "Current file: sample_leases/lease_b.txt",
    ) in fake_st.calls


def test_s3_search_sends_question_to_rag_search(monkeypatch):
    fake_st = RecordingStreamlit()
    fake_st.buttons["Search Indexed Leases"] = True
    fake_st.text_inputs["rag_search_question"] = "What is the rent?"
    monkeypatch.setattr(tabs, "st", fake_st)

    calls = []

    def fake_call_api(path, payload):
        calls.append((path, payload))
        return {
            "question": payload["question"],
            "matches": [
                {
                    "key": "sample_leases/valid_lease_a.txt",
                    "filename": "valid_lease_a.txt",
                    "snippet": "Rent is 1,500 pounds.",
                    "score": 0.8,
                    "chunk_index": 0,
                }
            ],
        }

    monkeypatch.setattr(tabs, "call_api", fake_call_api)

    tabs.render_s3_search_tab()

    assert calls == [
        ("/rag/search", {"question": "What is the rent?", "top_k": 5})
    ]
    assert ("write", "Rent is 1,500 pounds.") in fake_st.calls


def test_s3_chat_sends_selected_keys_and_preserves_history(monkeypatch):
    leases = s3_leases()
    fake_st = RecordingStreamlit()
    fake_st.multiselect_values["rag_chat_lease_keys"] = [leases[1]]
    fake_st.chat_inputs["rag_chat_question"] = "When is rent due?"
    fake_st.session_state["rag_chat_history"] = [
        {"role": "user", "content": "What is the rent?"},
        {"role": "assistant", "content": "Rent is 1,500 pounds."},
    ]
    monkeypatch.setattr(tabs, "st", fake_st)
    monkeypatch.setattr(tabs, "call_api_get", lambda path: leases)

    calls = []

    def fake_call_api(path, payload):
        calls.append((path, payload))
        return {
            "question": payload["question"],
            "answer": "Rent is due on the first day of each month.",
            "citations": [
                {
                    "key": "sample_leases/valid_lease_b.txt",
                    "filename": "valid_lease_b.txt",
                    "snippet": "Rent is due on the first day.",
                    "chunk_index": 0,
                }
            ],
        }

    monkeypatch.setattr(tabs, "call_api", fake_call_api)

    tabs.render_s3_chat_tab()

    assert calls == [
        (
            "/rag/chat",
            {
                "question": "When is rent due?",
                "lease_keys": ["sample_leases/valid_lease_b.txt"],
                "history": [
                    {"role": "user", "content": "What is the rent?"},
                    {"role": "assistant", "content": "Rent is 1,500 pounds."},
                ],
                "top_k": 5,
            },
        )
    ]
    assert fake_st.session_state["rag_chat_history"][-2:] == [
        {"role": "user", "content": "When is rent due?"},
        {
            "role": "assistant",
            "content": "Rent is due on the first day of each month.",
            "citations": [
                {
                    "key": "sample_leases/valid_lease_b.txt",
                    "filename": "valid_lease_b.txt",
                    "snippet": "Rent is due on the first day.",
                    "chunk_index": 0,
                }
            ],
        },
    ]
    assert ("container", {"height": 520, "border": True}) in fake_st.calls
    assert ("chat_message", "user") in fake_st.calls
    assert ("chat_message", "assistant") in fake_st.calls
    assert (
        "placeholder_markdown",
        "_Searching indexed lease text..._",
    ) in fake_st.calls
    assert (
        "placeholder_markdown",
        "Rent is due on the first day of each month.",
    ) in fake_st.calls
    assert ("markdown", "#### Sources") not in fake_st.calls


def test_s3_chat_toggle_shows_sources_after_assistant_response(monkeypatch):
    leases = s3_leases()
    fake_st = RecordingStreamlit()
    fake_st.toggles["rag_chat_show_sources"] = True
    fake_st.session_state["rag_chat_history"] = [
        {
            "role": "assistant",
            "content": "Pets are limited to one indoor cat.",
            "citations": [
                {
                    "key": "sample_leases/valid_lease_b.txt",
                    "filename": "valid_lease_b.txt",
                    "snippet": "One indoor cat is permitted.",
                    "chunk_index": 2,
                }
            ],
        },
    ]
    monkeypatch.setattr(tabs, "st", fake_st)
    monkeypatch.setattr(tabs, "call_api_get", lambda path: leases)

    tabs.render_s3_chat_tab()

    assistant_call = ("markdown", "Pets are limited to one indoor cat.")
    sources_call = ("markdown", "#### Sources")
    assert assistant_call in fake_st.calls
    assert sources_call in fake_st.calls
    assert fake_st.calls.index(assistant_call) < fake_st.calls.index(sources_call)
    assert ("write", "One indoor cat is permitted.") in fake_st.calls


def test_s3_chat_displays_grounding_warnings(monkeypatch):
    leases = s3_leases()
    fake_st = RecordingStreamlit()
    fake_st.chat_inputs["rag_chat_question"] = "Can I sublet?"
    monkeypatch.setattr(tabs, "st", fake_st)
    monkeypatch.setattr(tabs, "call_api_get", lambda path: leases)

    def fake_call_api(path, payload):
        return {
            "question": payload["question"],
            "answer": "I could not verify the generated answer against the indexed lease context.",
            "citations": [],
            "verification": {
                "overall_supported": False,
                "checks": [
                    {
                        "field_name": "answer",
                        "status": "unsupported",
                        "extracted_value": "Subletting is allowed.",
                        "evidence": None,
                        "explanation": "The context does not mention subletting.",
                    }
                ],
            },
            "warnings": [
                "answer was flagged as unsupported by the indexed lease context."
            ],
        }

    monkeypatch.setattr(tabs, "call_api", fake_call_api)

    tabs.render_s3_chat_tab()

    assert (
        "warning",
        "- answer was flagged as unsupported by the indexed lease context.",
    ) in fake_st.calls
    assert fake_st.session_state["rag_chat_history"][-1]["warnings"] == [
        "answer was flagged as unsupported by the indexed lease context."
    ]
