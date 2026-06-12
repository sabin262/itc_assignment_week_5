from __future__ import annotations

from datetime import UTC, datetime
import os
from typing import Any
from uuid import uuid4

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import BotoCoreError, ClientError

from app.schemas import (
    RAGChatResponse,
    RAGChatSessionListResponse,
    RAGChatSessionResponse,
    RAGChatSessionSummary,
    RAGChatStoredMessage,
    validate_chat_session_id,
)


SHARED_SCOPE_KEY = "SCOPE#shared"
SESSION_PK_PREFIX = "CHAT#"
META_SK = "META"
MESSAGE_SK_PREFIX = "MSG#"


class ChatHistoryError(RuntimeError):
    """Raised when saved chat history cannot be read or written."""


class ChatHistoryConfigurationError(ChatHistoryError):
    """Raised when chat history storage is not configured."""


class ChatHistoryNotFoundError(ChatHistoryError):
    """Raised when a saved chat session does not exist."""


class DynamoDBChatHistoryStore:
    def __init__(self, table_name: str | None, region_name: str | None = None):
        if not table_name:
            raise ChatHistoryConfigurationError(
                "CHAT_HISTORY_TABLE_NAME is not configured."
            )
        self._table = boto3.resource(
            "dynamodb",
            region_name=_dynamodb_region_name(region_name),
        ).Table(table_name)
        self._uses_sort_key: bool | None = None

    def save_exchange(
        self,
        *,
        session_id: str | None,
        question: str,
        lease_keys: list[str],
        response: RAGChatResponse,
    ) -> tuple[str, str]:
        session_id = validate_chat_session_id(session_id) if session_id else str(uuid4())
        now = _utc_now()
        pk = _session_pk(session_id)

        try:
            if self._table_uses_sort_key():
                self._save_exchange_composite_key(
                    pk=pk,
                    session_id=session_id,
                    question=question,
                    lease_keys=lease_keys,
                    response=response,
                    now=now,
                )
            else:
                self._save_exchange_single_key(
                    pk=pk,
                    session_id=session_id,
                    question=question,
                    lease_keys=lease_keys,
                    response=response,
                    now=now,
                )
        except (BotoCoreError, ClientError, ValueError) as exc:
            if _is_key_schema_mismatch(exc):
                self._uses_sort_key = False
                try:
                    self._save_exchange_single_key(
                        pk=pk,
                        session_id=session_id,
                        question=question,
                        lease_keys=lease_keys,
                        response=response,
                        now=now,
                    )
                    return session_id, now
                except (BotoCoreError, ClientError, ValueError) as fallback_exc:
                    raise _dynamodb_error(
                        "Could not save chat history",
                        fallback_exc,
                    ) from fallback_exc
            raise _dynamodb_error("Could not save chat history", exc) from exc

        return session_id, now

    def _table_uses_sort_key(self) -> bool:
        if self._uses_sort_key is None:
            self._uses_sort_key = _table_uses_sort_key(self._table)
        return self._uses_sort_key

    def _save_exchange_composite_key(
        self,
        *,
        pk: str,
        session_id: str,
        question: str,
        lease_keys: list[str],
        response: RAGChatResponse,
        now: str,
    ) -> None:
        meta_item = self._get_meta_item(pk)
        if meta_item is None:
            created_at = now
            title = _chat_title(question)
            message_count = 0
        else:
            created_at = str(meta_item.get("created_at") or now)
            title = str(meta_item.get("title") or _chat_title(question))
            message_count = int(meta_item.get("message_count") or 0)

        user_sequence = message_count + 1
        assistant_sequence = message_count + 2
        self._table.put_item(
            Item=_message_item(
                pk=pk,
                sequence=user_sequence,
                role="user",
                content=question,
                created_at=now,
            )
        )
        self._table.put_item(
            Item=_message_item(
                pk=pk,
                sequence=assistant_sequence,
                role="assistant",
                content=response.answer,
                created_at=now,
                citations=[
                    citation.model_dump(mode="json")
                    for citation in response.citations
                ],
                verification=(
                    response.verification.model_dump(mode="json")
                    if response.verification is not None
                    else None
                ),
                warnings=response.warnings,
            )
        )
        self._table.put_item(
            Item={
                "pk": pk,
                "sk": META_SK,
                "session_id": session_id,
                "title": title,
                "lease_keys": lease_keys,
                "created_at": created_at,
                "updated_at": now,
                "message_count": assistant_sequence,
                "gsi1pk": SHARED_SCOPE_KEY,
                "gsi1sk": f"{now}#{session_id}",
            }
        )

    def _save_exchange_single_key(
        self,
        *,
        pk: str,
        session_id: str,
        question: str,
        lease_keys: list[str],
        response: RAGChatResponse,
        now: str,
    ) -> None:
        item = self._get_session_item(pk)
        if item is None:
            created_at = now
            title = _chat_title(question)
            messages: list[dict[str, Any]] = []
        else:
            created_at = str(item.get("created_at") or now)
            title = str(item.get("title") or _chat_title(question))
            messages = [
                message
                for message in item.get("messages") or []
                if isinstance(message, dict)
            ]

        user_sequence = len(messages) + 1
        assistant_sequence = len(messages) + 2
        messages.extend(
            [
                _stored_message_payload(
                    sequence=user_sequence,
                    role="user",
                    content=question,
                    created_at=now,
                ),
                _stored_message_payload(
                    sequence=assistant_sequence,
                    role="assistant",
                    content=response.answer,
                    created_at=now,
                    citations=[
                        citation.model_dump(mode="json")
                        for citation in response.citations
                    ],
                    verification=(
                        response.verification.model_dump(mode="json")
                        if response.verification is not None
                        else None
                    ),
                    warnings=response.warnings,
                ),
            ]
        )
        self._table.put_item(
            Item={
                "pk": pk,
                "session_id": session_id,
                "title": title,
                "lease_keys": lease_keys,
                "created_at": created_at,
                "updated_at": now,
                "message_count": len(messages),
                "messages": messages,
                "gsi1pk": SHARED_SCOPE_KEY,
                "gsi1sk": f"{now}#{session_id}",
            }
        )

    def _get_meta_item(self, pk: str) -> dict[str, Any] | None:
        response = self._table.get_item(Key={"pk": pk, "sk": META_SK})
        item = response.get("Item")
        return item if isinstance(item, dict) else None

    def _get_session_item(self, pk: str) -> dict[str, Any] | None:
        response = self._table.get_item(Key={"pk": pk})
        item = response.get("Item")
        return item if isinstance(item, dict) else None

    def list_sessions(self, limit: int = 50) -> RAGChatSessionListResponse:
        try:
            response = self._table.query(
                IndexName="gsi1",
                KeyConditionExpression=Key("gsi1pk").eq(SHARED_SCOPE_KEY),
                ScanIndexForward=False,
                Limit=limit,
            )
        except (BotoCoreError, ClientError) as exc:
            raise _dynamodb_error("Could not list saved chat sessions", exc) from exc

        sessions = [
            _session_summary_from_item(item)
            for item in response.get("Items", [])
            if _is_session_summary_item(item)
        ]
        return RAGChatSessionListResponse(sessions=sessions)

    def get_session(self, session_id: str) -> RAGChatSessionResponse:
        session_id = validate_chat_session_id(session_id)
        pk = _session_pk(session_id)
        try:
            if self._table_uses_sort_key():
                return self._get_session_composite_key(session_id, pk)
            return self._get_session_single_key(session_id, pk)
        except (BotoCoreError, ClientError) as exc:
            if _is_key_schema_mismatch(exc):
                self._uses_sort_key = False
                try:
                    return self._get_session_single_key(session_id, pk)
                except (BotoCoreError, ClientError) as fallback_exc:
                    raise _dynamodb_error(
                        "Could not load saved chat session",
                        fallback_exc,
                    ) from fallback_exc
            raise _dynamodb_error("Could not load saved chat session", exc) from exc

    def _get_session_composite_key(
        self,
        session_id: str,
        pk: str,
    ) -> RAGChatSessionResponse:
        response = self._table.query(
            KeyConditionExpression=Key("pk").eq(pk),
        )
        items = response.get("Items", [])
        meta = next((item for item in items if item.get("sk") == META_SK), None)
        if meta is None:
            raise ChatHistoryNotFoundError(f"Saved chat session was not found: {session_id}")

        messages = [
            _stored_message_from_item(item)
            for item in sorted(items, key=lambda item: str(item.get("sk", "")))
            if str(item.get("sk", "")).startswith(MESSAGE_SK_PREFIX)
        ]
        summary = _session_summary_from_item(meta)
        return RAGChatSessionResponse(
            **summary.model_dump(mode="json"),
            messages=messages,
        )

    def _get_session_single_key(
        self,
        session_id: str,
        pk: str,
    ) -> RAGChatSessionResponse:
        item = self._get_session_item(pk)
        if item is None:
            raise ChatHistoryNotFoundError(
                f"Saved chat session was not found: {session_id}"
            )
        summary = _session_summary_from_item(item)
        return RAGChatSessionResponse(
            **summary.model_dump(mode="json"),
            messages=[
                _stored_message_from_item(message)
                for message in item.get("messages") or []
                if isinstance(message, dict)
            ],
        )

    def delete_session(self, session_id: str) -> None:
        session_id = validate_chat_session_id(session_id)
        pk = _session_pk(session_id)
        try:
            if self._table_uses_sort_key():
                self._delete_session_composite_key(session_id, pk)
            else:
                self._delete_session_single_key(session_id, pk)
        except ChatHistoryNotFoundError:
            raise
        except (BotoCoreError, ClientError) as exc:
            if _is_key_schema_mismatch(exc):
                self._uses_sort_key = False
                try:
                    self._delete_session_single_key(session_id, pk)
                    return
                except (BotoCoreError, ClientError) as fallback_exc:
                    raise _dynamodb_error(
                        "Could not delete saved chat session",
                        fallback_exc,
                    ) from fallback_exc
            raise _dynamodb_error("Could not delete saved chat session", exc) from exc

    def _delete_session_composite_key(self, session_id: str, pk: str) -> None:
        response = self._table.query(
            KeyConditionExpression=Key("pk").eq(pk),
        )
        items = response.get("Items", [])
        if not any(item.get("sk") == META_SK for item in items):
            raise ChatHistoryNotFoundError(
                f"Saved chat session was not found: {session_id}"
            )

        with self._table.batch_writer() as batch:
            for item in items:
                batch.delete_item(Key={"pk": item["pk"], "sk": item["sk"]})

    def _delete_session_single_key(self, session_id: str, pk: str) -> None:
        if self._get_session_item(pk) is None:
            raise ChatHistoryNotFoundError(
                f"Saved chat session was not found: {session_id}"
            )
        self._table.delete_item(Key={"pk": pk})


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _dynamodb_region_name(region_name: str | None = None) -> str:
    configured_region = (
        region_name
        or os.getenv("REGION_NAME")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
    )
    if not configured_region:
        raise ChatHistoryConfigurationError(
            "DynamoDB AWS region is not configured. Set REGION_NAME in .env."
        )
    return configured_region


def _dynamodb_error(operation: str, exc: Exception) -> ChatHistoryError:
    if isinstance(exc, ClientError):
        error = exc.response.get("Error", {})
        code = str(error.get("Code") or "ClientError")
        message = str(error.get("Message") or exc)
        return ChatHistoryError(f"{operation}: DynamoDB {code}: {message}")
    return ChatHistoryError(f"{operation}: {exc}")


def _is_key_schema_mismatch(exc: Exception) -> bool:
    if not isinstance(exc, ClientError):
        return False
    error = exc.response.get("Error", {})
    code = str(error.get("Code") or "")
    message = str(error.get("Message") or "").lower()
    return (
        code == "ValidationException"
        and "provided key element does not match the schema" in message
    )


def _table_uses_sort_key(table: Any) -> bool:
    try:
        key_schema = table.key_schema
    except AttributeError:
        return True
    except (BotoCoreError, ClientError):
        return True
    if not isinstance(key_schema, list):
        return True
    return any(
        isinstance(key, dict) and key.get("KeyType") == "RANGE"
        for key in key_schema
    )


def _session_pk(session_id: str) -> str:
    return f"{SESSION_PK_PREFIX}{session_id}"


def _chat_title(question: str) -> str:
    title = " ".join(question.split()) or "Lease chat"
    if len(title) <= 80:
        return title
    return f"{title[:77].rstrip()}..."


def _message_item(
    *,
    pk: str,
    sequence: int,
    role: str,
    content: str,
    created_at: str,
    citations: list[dict[str, Any]] | None = None,
    verification: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "pk": pk,
        "sk": f"{MESSAGE_SK_PREFIX}{sequence:06d}",
        "sequence": sequence,
        "role": role,
        "content": content,
        "citations": citations or [],
        "verification": verification,
        "warnings": warnings or [],
        "created_at": created_at,
    }


def _stored_message_payload(
    *,
    sequence: int,
    role: str,
    content: str,
    created_at: str,
    citations: list[dict[str, Any]] | None = None,
    verification: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "role": role,
        "content": content,
        "citations": citations or [],
        "verification": verification,
        "warnings": warnings or [],
        "created_at": created_at,
    }


def _is_session_summary_item(item: dict[str, Any]) -> bool:
    return item.get("sk") == META_SK or isinstance(item.get("messages"), list)


def _session_summary_from_item(item: dict[str, Any]) -> RAGChatSessionSummary:
    return RAGChatSessionSummary(
        session_id=str(item.get("session_id") or "").strip(),
        title=str(item.get("title") or "Lease chat"),
        lease_keys=[str(key) for key in item.get("lease_keys") or []],
        message_count=int(item.get("message_count") or 0),
        created_at=str(item.get("created_at") or ""),
        updated_at=str(item.get("updated_at") or ""),
    )


def _stored_message_from_item(item: dict[str, Any]) -> RAGChatStoredMessage:
    return RAGChatStoredMessage(
        role=item.get("role"),
        content=str(item.get("content") or ""),
        citations=item.get("citations") or [],
        verification=item.get("verification"),
        warnings=item.get("warnings") or [],
        created_at=item.get("created_at"),
    )
