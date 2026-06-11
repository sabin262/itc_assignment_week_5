import json
import os
from functools import lru_cache

import boto3
from botocore.exceptions import ClientError
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

SECRET_NAME = "dev/ds-may26/LeaseSummariser"


@lru_cache
def _fetch_secrets() -> dict:
    region = os.getenv("REGION_NAME", "eu-west-2")
    client = boto3.client("secretsmanager", region_name=region)
    try:
        response = client.get_secret_value(SecretId=SECRET_NAME)
        return json.loads(response["SecretString"])
    except ClientError as e:
        raise RuntimeError(f"Failed to fetch secrets from AWS Secrets Manager: {e}") from e


class Settings(BaseSettings):
    azure_openai_api_key: str = Field(alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field(alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version: str = Field(alias="AZURE_OPENAI_API_VERSION")
    azure_openai_deployment: str = Field(alias="AZURE_OPENAI_DEPLOYMENT")
    azure_openai_embedding_deployment: str | None = Field(
        default=None,
        alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    )
    azure_openai_generation_api: str | None = Field(
        default=None,
        alias="AZURE_OPENAI_GENERATION_API",
    )

    model_config = SettingsConfigDict(
        extra="ignore",
        populate_by_name=True,
    )


class S3Settings(BaseSettings):
    s3_bucket_name: str | None = Field(default=None, alias="S3_BUCKET_NAME")
    s3_prefix: str = Field(default="sample_leases", alias="S3_PREFIX")
    s3_pdf_file_name: str | None = Field(default=None, alias="S3_PDF_FILE_NAME")
    pdf_source_dir: str | None = Field(default=None, alias="PDF_SOURCE_DIR")

    model_config = SettingsConfigDict(
        extra="ignore",
        populate_by_name=True,
    )


class RAGSettings(BaseSettings):
    chroma_persist_dir: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(
        default="lease_chunks",
        alias="CHROMA_COLLECTION_NAME",
    )

    model_config = SettingsConfigDict(
        extra="ignore",
        populate_by_name=True,
    )


class LangfuseSettings(BaseSettings):
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_base_url: str | None = Field(default=None, alias="LANGFUSE_BASE_URL")

    model_config = SettingsConfigDict(
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings(**_fetch_secrets())


@lru_cache
def get_s3_settings() -> S3Settings:
    return S3Settings(**_fetch_secrets())


@lru_cache
def get_rag_settings() -> RAGSettings:
    return RAGSettings(**_fetch_secrets())


@lru_cache
def get_langfuse_settings() -> LangfuseSettings:
    return LangfuseSettings(**_fetch_secrets())
