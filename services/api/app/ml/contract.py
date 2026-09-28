"""The boundary between the app and the ML models.

The app only depends on these types. The Phase 1 model service must return
JSON matching `AnalyzeResponse` from `POST /v1/analyze` (multipart field `image`).
"""

from typing import Protocol

from pydantic import BaseModel


class Garment(BaseModel):
    category: str  # e.g. "shirt", "trousers", "sneakers"
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2, normalised 0-1
    attributes: dict[str, str] = {}  # e.g. {"color": "navy", "pattern": "striped"}


class AnalyzeResponse(BaseModel):
    model_version: str
    garments: list[Garment]  # one photo can contain several garments


class GarmentAnalyzer(Protocol):
    def analyze(self, image: bytes, filename: str) -> AnalyzeResponse: ...
