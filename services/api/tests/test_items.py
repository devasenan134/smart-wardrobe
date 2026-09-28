import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("WARDROBE_DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("WARDROBE_STORAGE_DIR", str(tmp_path / "storage"))
    import importlib

    import app.config, app.db, app.storage, app.ml.analyzers, app.main

    for mod in (app.config, app.db, app.storage, app.ml.analyzers, app.main):
        importlib.reload(mod)
    with TestClient(app.main.app) as c:
        yield c


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "navy").save(buf, "PNG")
    return buf.getvalue()


def test_upload_list_delete(client):
    resp = client.post("/items", files={"image": ("shirt.png", png_bytes())}, data={"source": "online"})
    assert resp.status_code == 201
    [item] = resp.json()
    assert item["source"] == "online"
    assert item["analyzer_version"] == "stub-0"

    assert len(client.get("/items").json()) == 1
    assert client.get(item["image_url"]).status_code == 200

    assert client.delete(f"/items/{item['id']}").status_code == 204
    assert client.get(f"/items/{item['id']}").status_code == 404


def test_rejects_non_image(client):
    resp = client.post("/items", files={"image": ("x.png", b"not an image")})
    assert resp.status_code == 415
