from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WARDROBE_", env_file=".env")

    database_url: str = "sqlite:///./wardrobe.db"
    storage_dir: Path = Path("./storage")
    # "stub" until the Phase 1 model service exists, then "http"
    analyzer: str = "stub"
    ml_service_url: str = "http://localhost:8001"


settings = Settings()
