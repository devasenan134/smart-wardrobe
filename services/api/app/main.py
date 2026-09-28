import io
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from sqlmodel import Session, select

from app import storage
from app.db import get_session, init_db
from app.ml.analyzers import get_analyzer
from app.ml.contract import GarmentAnalyzer
from app.models import Item, ItemRead, ItemSource

MAX_UPLOAD_BYTES = 15 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Smart Wardrobe API", lifespan=lifespan)


def to_read(item: Item) -> ItemRead:
    return ItemRead(**item.model_dump(), image_url=f"/items/{item.id}/image")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/items", response_model=list[ItemRead], status_code=201)
def create_items(
    image: UploadFile = File(...),
    source: ItemSource = Form(ItemSource.photo),
    session: Session = Depends(get_session),
    analyzer: GarmentAnalyzer = Depends(get_analyzer),
):
    """Upload one photo; creates one wardrobe item per garment detected in it."""
    data = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large")
    try:
        fmt = Image.open(io.BytesIO(data)).format
    except UnidentifiedImageError:
        raise HTTPException(415, "Not an image")

    key = storage.save(data, f".{fmt.lower()}")
    result = analyzer.analyze(data, image.filename or key)
    items = [
        Item(
            source=source,
            original_image=key,
            category=g.category,
            attributes=g.attributes,
            confidence=g.confidence,
            analyzer_version=result.model_version,
        )
        for g in result.garments
    ]
    session.add_all(items)
    session.commit()
    for i in items:
        session.refresh(i)
    return [to_read(i) for i in items]


@app.get("/items", response_model=list[ItemRead])
def list_items(session: Session = Depends(get_session)):
    items = session.exec(select(Item).order_by(Item.created_at.desc())).all()
    return [to_read(i) for i in items]


def get_item(item_id: uuid.UUID, session: Session = Depends(get_session)) -> Item:
    item = session.get(Item, item_id)
    if not item:
        raise HTTPException(404, "Item not found")
    return item


@app.get("/items/{item_id}", response_model=ItemRead)
def read_item(item: Item = Depends(get_item)):
    return to_read(item)


@app.get("/items/{item_id}/image")
def item_image(item: Item = Depends(get_item)):
    return FileResponse(storage.path_for(item.catalog_image or item.original_image))


@app.delete("/items/{item_id}", status_code=204)
def delete_item(item: Item = Depends(get_item), session: Session = Depends(get_session)):
    session.delete(item)
    session.commit()
    # several items can share one photo; drop the file once nothing references it
    still_used = session.exec(select(Item).where(Item.original_image == item.original_image)).first()
    if not still_used:
        storage.delete(item.original_image)
