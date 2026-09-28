import uuid
from datetime import UTC, datetime
from enum import StrEnum

from sqlmodel import JSON, Column, Field, SQLModel


class ItemSource(StrEnum):
    photo = "photo"  # user photographed the garment
    online = "online"  # screenshot / product image from a purchase


class Item(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    source: ItemSource
    original_image: str  # storage key of the upload
    catalog_image: str | None = None  # cleaned shop-style image (Phase 2)
    category: str = "unknown"
    attributes: dict = Field(default_factory=dict, sa_column=Column(JSON))
    confidence: float = 0.0
    analyzer_version: str = ""  # which model produced the labels
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ItemRead(SQLModel):
    id: uuid.UUID
    source: ItemSource
    category: str
    attributes: dict
    confidence: float
    analyzer_version: str
    image_url: str
    created_at: datetime
