from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from pathlib import Path, PurePosixPath

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECRET_NAME = "dev/ds-may26/LeaseSummariser"


def _fetch_secrets() -> dict:
    region = os.getenv("AWS_REGION", os.getenv("AWS_REGION", "eu-west-2"))
    client = boto3.client("secretsmanager", region_name=region)
    try:
        response = client.get_secret_value(SecretId=SECRET_NAME)
        return json.loads(response["SecretString"])
    except Exception as e:
        print(f"Warning: could not fetch secrets from AWS Secrets Manager: {e}", file=sys.stderr)
        return {}

CONTENT_TYPES_BY_SUFFIX = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pdf": "application/pdf",
    ".txt": "text/plain",
}


def env_value(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload all sample lease files to an S3 prefix. AWS credentials are "
            "loaded from .env; all other config is fetched from AWS Secrets Manager."
        )
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the .env file (used for AWS credentials only). Defaults to .env at the project root.",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Folder containing lease files. Defaults to PDF_SOURCE_DIR from Secrets Manager or sample_leases.",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="S3 bucket name. Defaults to S3_BUCKET_NAME from AWS Secrets Manager.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="S3 folder/prefix. Defaults to S3_PREFIX from AWS Secrets Manager or sample_leases.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without writing to S3.",
    )
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_s3_prefix(prefix: str) -> str:
    return prefix.strip().strip("/")


def find_files(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file()
    )


def s3_key_for(file_path: Path, source_dir: Path, prefix: str) -> str:
    relative_path = file_path.relative_to(source_dir).as_posix()
    if not prefix:
        return relative_path
    return str(PurePosixPath(prefix) / relative_path)


def content_type_for(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix in CONTENT_TYPES_BY_SUFFIX:
        return CONTENT_TYPES_BY_SUFFIX[suffix]
    return mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"


def build_s3_client():
    access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = env_value("AWS_SESSION_TOKEN")
    region_name = env_value("AWS_REGION", "AWS_DEFAULT_REGION")

    missing = [
        name
        for name, value in (
            ("AWS_ACCESS_KEY_ID", access_key_id),
            ("AWS_SECRET_ACCESS_KEY", secret_access_key),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "Missing required AWS credential setting(s) in .env: "
            + ", ".join(missing)
        )

    session = boto3.session.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        aws_session_token=session_token,
        region_name=region_name,
    )
    return session.client("s3")


def main() -> int:
    args = parse_args()

    env_path = resolve_project_path(args.env_file)
    if not env_path.exists():
        print(f"Could not find env file: {env_path}", file=sys.stderr)
        return 1

    # Load .env for AWS credentials only (needed to connect to Secrets Manager)
    load_dotenv(env_path, override=True)

    # Fetch all remaining config from AWS Secrets Manager
    secrets = _fetch_secrets()

    source_dir = resolve_project_path(
        args.source_dir
        or secrets.get("PDF_SOURCE_DIR")
        or secrets.get("S3_SOURCE_DIR")
        or "sample_leases"
    )
    bucket = (
        args.bucket
        or secrets.get("S3_BUCKET_NAME")
        or secrets.get("S3_BUCKET")
        or secrets.get("AWS_S3_BUCKET")
    )
    prefix = normalize_s3_prefix(
        args.prefix
        if args.prefix is not None
        else secrets.get("S3_PREFIX", "sample_leases")
    )

    if not bucket:
        print(
            "Missing S3 bucket. Set S3_BUCKET_NAME in .env or pass --bucket.",
            file=sys.stderr,
        )
        return 1

    if not source_dir.exists() or not source_dir.is_dir():
        print(f"Source folder does not exist: {source_dir}", file=sys.stderr)
        return 1

    files = find_files(source_dir)
    if not files:
        print(f"No files found in {source_dir}")
        return 0

    uploads = [
        (file_path, s3_key_for(file_path, source_dir, prefix))
        for file_path in files
    ]
    destination = f"s3://{bucket}/{prefix}/" if prefix else f"s3://{bucket}/"

    if args.dry_run:
        print(f"Dry run: {len(uploads)} file(s) would be uploaded to {destination}")
        for file_path, key in uploads:
            print(f"{file_path} -> s3://{bucket}/{key}")
        return 0

    try:
        s3 = build_s3_client()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for file_path, key in uploads:
        print(f"Uploading {file_path.name} -> s3://{bucket}/{key}")
        try:
            s3.upload_file(
                str(file_path),
                bucket,
                key,
                ExtraArgs={"ContentType": content_type_for(file_path)},
            )
        except (BotoCoreError, ClientError) as exc:
            print(f"Failed to upload {file_path}: {exc}", file=sys.stderr)
            return 1

    print(f"Uploaded {len(uploads)} file(s) to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
