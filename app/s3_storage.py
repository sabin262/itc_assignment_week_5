from pathlib import PurePosixPath
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.document_parser import SUPPORTED_EXTENSIONS
from app.schemas import S3LeaseFile


class S3StorageError(RuntimeError):
    """Raised when S3 lease storage cannot complete an operation."""


class S3ConfigurationError(S3StorageError):
    """Raised when required S3 settings are missing."""


class S3InvalidKeyError(S3StorageError):
    """Raised when an S3 key is not allowed for lease loading."""


class S3ObjectNotFoundError(S3StorageError):
    """Raised when a requested S3 lease object does not exist."""


class S3LeaseStorage:
    def __init__(
        self,
        bucket: str | None,
        prefix: str = "sample_leases",
        client: Any | None = None,
    ):
        self._bucket = bucket
        self._prefix = _normalize_prefix(prefix)
        self._client = client

    def list_lease_files(self) -> list[S3LeaseFile]:
        bucket = self._require_bucket()
        lease_files: list[S3LeaseFile] = []

        try:
            paginator = self._s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=self._list_prefix)
            for page in pages:
                for item in page.get("Contents", []):
                    key = item.get("Key", "")
                    if not key or key.endswith("/") or not _is_supported_key(key):
                        continue

                    lease_files.append(
                        S3LeaseFile(
                            key=key,
                            filename=PurePosixPath(key).name,
                            size=item.get("Size", 0),
                            last_modified=item.get("LastModified"),
                        )
                    )
        except (BotoCoreError, ClientError) as exc:
            raise S3StorageError(
                _s3_error_message("Could not list S3 lease files", exc)
            ) from exc

        return lease_files

    def upload_file(self, filename: str, content: bytes) -> str:
        """Upload a file under the configured S3 prefix and return its S3 key."""
        bucket = self._require_bucket()
        safe_name = PurePosixPath(filename).name
        key = f"{self._prefix}/{safe_name}" if self._prefix else safe_name
        print(f"[S3] uploading {len(content)} bytes -> s3://{bucket}/{key}")
        try:
            self._s3_client.put_object(Bucket=bucket, Key=key, Body=content)
        except (BotoCoreError, ClientError) as exc:
            raise S3StorageError(
                _s3_error_message(f"Could not upload file to S3: {safe_name}", exc)
            ) from exc
        print(f"[S3] upload complete: {key!r}")
        return key

    def get_file(self, key: str) -> tuple[str, bytes]:
        bucket = self._require_bucket()
        validated_key = self._validate_key(key)

        try:
            response = self._s3_client.get_object(Bucket=bucket, Key=validated_key)
            content = response["Body"].read()
        except ClientError as exc:
            if _is_not_found_error(exc):
                raise S3ObjectNotFoundError(
                    f"S3 lease file was not found: {validated_key}"
                ) from exc
            raise S3StorageError(
                _s3_error_message(
                    f"Could not download S3 lease file: {validated_key}",
                    exc,
                )
            ) from exc
        except BotoCoreError as exc:
            raise S3StorageError(
                _s3_error_message(
                    f"Could not download S3 lease file: {validated_key}",
                    exc,
                )
            ) from exc

        return PurePosixPath(validated_key).name, content

    @property
    def _s3_client(self) -> Any:
        if self._client is None:
            self._client = boto3.client("s3")
        return self._client

    @property
    def _list_prefix(self) -> str:
        if not self._prefix:
            return ""
        return f"{self._prefix}/"

    def _require_bucket(self) -> str:
        if not self._bucket:
            raise S3ConfigurationError("S3 bucket is not configured. Set S3_BUCKET_NAME.")
        return self._bucket

    def _validate_key(self, key: str) -> str:
        normalized_key = key.strip().lstrip("/")
        if not normalized_key:
            raise S3InvalidKeyError("S3 key is required.")
        if "\\" in normalized_key:
            raise S3InvalidKeyError("S3 key must use forward slashes.")
        if self._prefix and not normalized_key.startswith(f"{self._prefix}/"):
            raise S3InvalidKeyError(
                "S3 key must be inside the configured S3_PREFIX."
            )
        if not _is_supported_key(normalized_key):
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise S3InvalidKeyError(f"Unsupported file type. Use {supported}.")
        return normalized_key


def _normalize_prefix(prefix: str | None) -> str:
    return (prefix or "").strip().strip("/")


def _is_supported_key(key: str) -> bool:
    return PurePosixPath(key).suffix.lower() in SUPPORTED_EXTENSIONS


def _is_not_found_error(exc: ClientError) -> bool:
    code = str(exc.response.get("Error", {}).get("Code", ""))
    return code in {"404", "NoSuchKey", "NotFound"}


def _s3_error_message(operation: str, exc: Exception) -> str:
    if isinstance(exc, ClientError):
        error = exc.response.get("Error", {})
        code = str(error.get("Code") or "ClientError")
        message = str(error.get("Message") or exc)
        return f"{operation}: S3 {code}: {message}"
    return f"{operation}: {exc}"
