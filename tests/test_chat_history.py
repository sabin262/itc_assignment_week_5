from copy import deepcopy
from types import SimpleNamespace

from botocore.exceptions import ClientError
import pytest

from app.chat_history import (
    ChatHistoryError,
    ChatHistoryConfigurationError,
    ChatHistoryNotFoundError,
    DynamoDBChatHistoryStore,
)
from app.schemas import RAGChatResponse, RAGCitation


class FakeDynamoTable:
    key_schema = [
        {"AttributeName": "pk", "KeyType": "HASH"},
        {"AttributeName": "sk", "KeyType": "RANGE"},
    ]

    def __init__(self) -> None:
        self.items: dict[tuple[str, str], dict[str, object]] = {}

    def get_item(self, Key):
        item = self.items.get((Key["pk"], Key["sk"]))
        return {"Item": deepcopy(item)} if item else {}

    def put_item(self, Item):
        self.items[(Item["pk"], Item["sk"])] = deepcopy(Item)

    def delete_item(self, Key):
        self.items.pop((Key["pk"], Key["sk"]), None)

    def query(self, **kwargs):
        if kwargs.get("IndexName") == "gsi1":
            gsi_value = _condition_value(kwargs["KeyConditionExpression"])
            items = [
                deepcopy(item)
                for item in self.items.values()
                if item.get("gsi1pk") == gsi_value
            ]
            items.sort(
                key=lambda item: str(item.get("gsi1sk", "")),
                reverse=not kwargs.get("ScanIndexForward", True),
            )
            return {"Items": items[: kwargs.get("Limit", len(items))]}

        pk = _condition_value(kwargs["KeyConditionExpression"])
        items = [
            deepcopy(item)
            for (item_pk, _sk), item in self.items.items()
            if item_pk == pk
        ]
        items.sort(key=lambda item: str(item.get("sk", "")))
        return {"Items": items}

    def batch_writer(self):
        table = self

        class FakeBatchWriter:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def delete_item(self, Key):
                table.items.pop((Key["pk"], Key["sk"]), None)

        return FakeBatchWriter()


class FakeSingleKeyDynamoTable:
    key_schema = [
        {"AttributeName": "pk", "KeyType": "HASH"},
    ]

    def __init__(self) -> None:
        self.items: dict[str, dict[str, object]] = {}

    def get_item(self, Key):
        if set(Key) != {"pk"}:
            raise ClientError(
                {
                    "Error": {
                        "Code": "ValidationException",
                        "Message": "The provided key element does not match the schema",
                    }
                },
                "GetItem",
            )
        item = self.items.get(Key["pk"])
        return {"Item": deepcopy(item)} if item else {}

    def put_item(self, Item):
        self.items[Item["pk"]] = deepcopy(Item)

    def query(self, **kwargs):
        if kwargs.get("IndexName") == "gsi1":
            gsi_value = _condition_value(kwargs["KeyConditionExpression"])
            items = [
                deepcopy(item)
                for item in self.items.values()
                if item.get("gsi1pk") == gsi_value
            ]
            items.sort(
                key=lambda item: str(item.get("gsi1sk", "")),
                reverse=not kwargs.get("ScanIndexForward", True),
            )
            return {"Items": items[: kwargs.get("Limit", len(items))]}

        pk = _condition_value(kwargs["KeyConditionExpression"])
        item = self.items.get(pk)
        return {"Items": [deepcopy(item)] if item else []}

    def delete_item(self, Key):
        if set(Key) != {"pk"}:
            raise ClientError(
                {
                    "Error": {
                        "Code": "ValidationException",
                        "Message": "The provided key element does not match the schema",
                    }
                },
                "DeleteItem",
            )
        self.items.pop(Key["pk"], None)


def _condition_value(condition) -> str:
    values = getattr(condition, "_values")
    return values[1]


def test_dynamodb_chat_history_store_saves_lists_loads_and_deletes(monkeypatch):
    table = FakeDynamoTable()
    resource_calls = []
    times = [
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T00:01:00+00:00",
    ]

    def fake_resource(service, **kwargs):
        resource_calls.append((service, kwargs))
        return SimpleNamespace(Table=lambda _name: table)

    monkeypatch.setenv("REGION_NAME", "eu-west-2")
    monkeypatch.setattr(
        "app.chat_history.boto3.resource",
        fake_resource,
    )
    monkeypatch.setattr("app.chat_history._utc_now", lambda: times.pop(0))
    monkeypatch.setattr(
        "app.chat_history.uuid4",
        lambda: "session-one",
    )
    store = DynamoDBChatHistoryStore("lease-chat-history")
    assert resource_calls == [("dynamodb", {"region_name": "eu-west-2"})]

    first_response = RAGChatResponse(
        question="When is rent due?",
        answer="Rent is due on the first day.",
        citations=[
            RAGCitation(
                key="sample_leases/lease_a.txt",
                filename="lease_a.txt",
                snippet="Rent is due on the first day.",
                chunk_index=0,
            )
        ],
    )
    session_id, saved_at = store.save_exchange(
        session_id=None,
        question="When is rent due?",
        lease_keys=["sample_leases/lease_a.txt"],
        response=first_response,
    )
    store.save_exchange(
        session_id=session_id,
        question="What is the rent?",
        lease_keys=["sample_leases/lease_a.txt"],
        response=RAGChatResponse(
            question="What is the rent?",
            answer="Rent is 1,500 pounds.",
            citations=[],
        ),
    )

    assert session_id == "session-one"
    assert saved_at == "2026-01-01T00:00:00+00:00"
    sessions = store.list_sessions().sessions
    assert sessions[0].session_id == "session-one"
    assert sessions[0].message_count == 4
    assert sessions[0].updated_at == "2026-01-01T00:01:00+00:00"

    loaded = store.get_session("session-one")
    assert loaded.title == "When is rent due?"
    assert [message.role for message in loaded.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert loaded.messages[1].citations[0].key == "sample_leases/lease_a.txt"

    store.delete_session("session-one")
    with pytest.raises(ChatHistoryNotFoundError):
        store.get_session("session-one")


def test_dynamodb_chat_history_store_supports_single_key_table(monkeypatch):
    table = FakeSingleKeyDynamoTable()
    times = [
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T00:01:00+00:00",
    ]
    monkeypatch.setenv("REGION_NAME", "eu-west-2")
    monkeypatch.setattr(
        "app.chat_history.boto3.resource",
        lambda _service, **_kwargs: SimpleNamespace(Table=lambda _name: table),
    )
    monkeypatch.setattr("app.chat_history._utc_now", lambda: times.pop(0))
    monkeypatch.setattr("app.chat_history.uuid4", lambda: "session-one")
    store = DynamoDBChatHistoryStore("lease-chat-history")

    session_id, _saved_at = store.save_exchange(
        session_id=None,
        question="When is rent due?",
        lease_keys=["sample_leases/lease_a.txt"],
        response=RAGChatResponse(
            question="When is rent due?",
            answer="Rent is due on the first day.",
            citations=[
                RAGCitation(
                    key="sample_leases/lease_a.txt",
                    filename="lease_a.txt",
                    snippet="Rent is due on the first day.",
                    chunk_index=0,
                )
            ],
        ),
    )
    store.save_exchange(
        session_id=session_id,
        question="What is the rent?",
        lease_keys=["sample_leases/lease_a.txt"],
        response=RAGChatResponse(
            question="What is the rent?",
            answer="Rent is 1,500 pounds.",
            citations=[],
        ),
    )

    assert list(table.items) == ["CHAT#session-one"]
    sessions = store.list_sessions().sessions
    assert sessions[0].session_id == "session-one"
    assert sessions[0].message_count == 4

    loaded = store.get_session("session-one")
    assert [message.role for message in loaded.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert loaded.messages[1].citations[0].key == "sample_leases/lease_a.txt"

    store.delete_session("session-one")
    assert table.items == {}


def test_dynamodb_chat_history_store_requires_table_name():
    with pytest.raises(ChatHistoryConfigurationError, match="CHAT_HISTORY_TABLE_NAME"):
        DynamoDBChatHistoryStore(None)


def test_dynamodb_chat_history_store_requires_region(monkeypatch):
    monkeypatch.delenv("REGION_NAME", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    with pytest.raises(ChatHistoryConfigurationError, match="REGION_NAME"):
        DynamoDBChatHistoryStore("lease-chat-history")


def test_dynamodb_chat_history_store_reports_dynamodb_error_details(monkeypatch):
    class FailingTable:
        def query(self, **_kwargs):
            raise ClientError(
                {
                    "Error": {
                        "Code": "ValidationException",
                        "Message": "The table does not have the specified index: gsi1",
                    }
                },
                "Query",
            )

    monkeypatch.setenv("REGION_NAME", "eu-west-2")
    monkeypatch.setattr(
        "app.chat_history.boto3.resource",
        lambda _service, **_kwargs: SimpleNamespace(
            Table=lambda _name: FailingTable()
        ),
    )
    store = DynamoDBChatHistoryStore("lease-chat-history")

    with pytest.raises(ChatHistoryError) as exc_info:
        store.list_sessions()

    assert "ValidationException" in str(exc_info.value)
    assert "gsi1" in str(exc_info.value)
