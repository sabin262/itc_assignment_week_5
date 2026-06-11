import json
import os
from functools import lru_cache

import boto3
from botocore.exceptions import ClientError
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

SECRET_NAME = "dev/ds-may26/LeaseSummariser"


def _fetch_azure_secrets() -> dict:
    region = os.getenv("REGION_NAME", "eu-west-2")
    client = boto3.client("secretsmanager", region_name=region)
    try:
        response = client.get_secret_value(SecretId=SECRET_NAME)
        return json.loads(response["SecretString"])
    except ClientError as e:
        raise RuntimeError(f"Failed to fetch Azure secrets from AWS Secrets Manager: {e}") from e


class Settings(BaseSettings):
    azure_openai_api_key: str = Field(alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field(alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version: str = Field(alias="AZURE_OPENAI_API_VERSION")
    azure_openai_deployment: str = Field(alias="AZURE_OPENAI_DEPLOYMENT")

    model_config = SettingsConfigDict(
        extra="ignore",
        populate_by_name=True,
    )


class S3Settings(BaseSettings):
    s3_bucket_name: str | None = Field(default=None, alias="S3_BUCKET_NAME")
    s3_prefix: str = Field(default="sample_leases", alias="S3_PREFIX")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    secrets = _fetch_azure_secrets()
    return Settings(
        AZURE_OPENAI_API_KEY=secrets["AZURE_OPENAI_API_KEY"],
        AZURE_OPENAI_ENDPOINT=secrets["AZURE_OPENAI_ENDPOINT"],
        AZURE_OPENAI_API_VERSION=secrets["AZURE_OPENAI_API_VERSION"],
        AZURE_OPENAI_DEPLOYMENT=secrets["AZURE_OPENAI_DEPLOYMENT"],
    )


@lru_cache
def get_s3_settings() -> S3Settings:
    return S3Settings()

