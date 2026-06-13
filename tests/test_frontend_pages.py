from contextlib import nullcontext

from frontend import streamlit_app


class Sidebar:
    def __init__(self, selected_page: str) -> None:
        self.selected_page = selected_page

    def radio(self, label: str, options):
        return self.selected_page


class RecordingStreamlit:
    def __init__(self, selected_page: str) -> None:
        self.calls = []
        self.sidebar = Sidebar(selected_page)

    def title(self, text: str) -> None:
        self.calls.append(("title", text))

    def tabs(self, labels):
        self.calls.append(("tabs", labels))
        return [nullcontext() for _ in labels]


def test_local_page_renders_only_local_tabs(monkeypatch):
    fake_st = RecordingStreamlit("Local Leases")
    rendered = []
    monkeypatch.setattr(streamlit_app, "st", fake_st)
    monkeypatch.setattr(streamlit_app, "configure_page", lambda: None)
    monkeypatch.setattr(
        streamlit_app,
        "render_summarise_tab",
        lambda: rendered.append("local_summarise"),
    )
    monkeypatch.setattr(
        streamlit_app,
        "render_compare_tab",
        lambda: rendered.append("local_compare"),
    )
    monkeypatch.setattr(
        streamlit_app,
        "render_s3_summarise_tab",
        lambda: rendered.append("s3_summarise"),
    )

    streamlit_app.main()

    assert ("tabs", ["Summarise", "Compare"]) in fake_st.calls
    assert rendered == ["local_summarise", "local_compare"]


def test_s3_page_renders_only_s3_tabs(monkeypatch):
    fake_st = RecordingStreamlit("S3 Leases")
    rendered = []
    monkeypatch.setattr(streamlit_app, "st", fake_st)
    monkeypatch.setattr(streamlit_app, "configure_page", lambda: None)
    monkeypatch.setattr(
        streamlit_app,
        "render_summarise_tab",
        lambda: rendered.append("local_summarise"),
    )
    monkeypatch.setattr(
        streamlit_app,
        "render_s3_summarise_tab",
        lambda: rendered.append("s3_summarise"),
    )
    monkeypatch.setattr(
        streamlit_app,
        "render_s3_compare_tab",
        lambda: rendered.append("s3_compare"),
    )
    monkeypatch.setattr(
        streamlit_app,
        "render_s3_index_tab",
        lambda: rendered.append("s3_index"),
    )
    monkeypatch.setattr(
        streamlit_app,
        "render_s3_chat_tab",
        lambda: rendered.append("s3_chat"),
    )

    streamlit_app.main()

    assert (
        "tabs",
        ["Summarise", "Compare", "Index", "Upload & Index", "Chat"],
    ) in fake_st.calls
    assert rendered == [
        "s3_summarise",
        "s3_compare",
        "s3_index",
        "s3_chat",
    ]
