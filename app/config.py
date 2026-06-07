from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    azure_openai_api_key: str = Field(alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field(alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version: str = Field(alias="AZURE_OPENAI_API_VERSION")
    azure_openai_deployment: str = Field(alias="AZURE_OPENAI_DEPLOYMENT")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

