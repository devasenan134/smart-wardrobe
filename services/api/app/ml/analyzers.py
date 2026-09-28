import httpx

from app.config import settings
from app.ml.contract import AnalyzeResponse, Garment, GarmentAnalyzer


class StubAnalyzer:
    """Placeholder until the real model is served."""

    def analyze(self, image: bytes, filename: str) -> AnalyzeResponse:
        return AnalyzeResponse(
            model_version="stub-0",
            garments=[Garment(category="unknown", confidence=0.0, bbox=(0, 0, 1, 1))],
        )


class HttpAnalyzer:
    def __init__(self, base_url: str):
        self.client = httpx.Client(base_url=base_url, timeout=30)

    def analyze(self, image: bytes, filename: str) -> AnalyzeResponse:
        resp = self.client.post("/v1/analyze", files={"image": (filename, image)})
        resp.raise_for_status()
        return AnalyzeResponse.model_validate(resp.json())


def get_analyzer() -> GarmentAnalyzer:
    if settings.analyzer == "http":
        return HttpAnalyzer(settings.ml_service_url)
    return StubAnalyzer()
