import uuid
from pathlib import Path

from app.config import settings


def save(data: bytes, suffix: str) -> str:
    key = f"{uuid.uuid4().hex}{suffix}"
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    (settings.storage_dir / key).write_bytes(data)
    return key


def path_for(key: str) -> Path:
    return settings.storage_dir / key


def delete(key: str) -> None:
    path_for(key).unlink(missing_ok=True)
