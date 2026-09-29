"""Download Fashionpedia into ml/data/fashionpedia/.

By default only the annotation JSONs are fetched (~557 MB), which is enough for
EDA on class counts, masks and attributes. Images are opt-in (~3.6 GB zipped).

    uv run python scripts/download_fashionpedia.py              # annotations
    uv run python scripts/download_fashionpedia.py --images     # + images, unzipped
    uv run python scripts/download_fashionpedia.py --dry-run    # show what would happen

Files that already exist with the right size are skipped. Interrupted downloads
resume from the .part file.

Source: https://github.com/cvdfoundation/fashionpedia
License: annotations CC BY 4.0; images are Flickr, each with its own license
(see docs/decisions.md).
"""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://s3.amazonaws.com/ifashionist-dataset"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "fashionpedia"

ANNOTATIONS = [
    "annotations/instances_attributes_val2020.json",  # ~15 MB, 1.2k images
    "annotations/instances_attributes_train2020.json",  # ~540 MB, 45k images
]
IMAGES = [
    "images/val_test2020.zip",  # ~240 MB
    "images/train2020.zip",  # ~3.3 GB
]


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as resp:
        return int(resp.headers["Content-Length"])


def download(url: str, dest: Path) -> None:
    total = remote_size(url)
    if dest.exists() and dest.stat().st_size == total:
        print(f"skip  {dest.name} (already downloaded)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0

    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    with urllib.request.urlopen(req) as resp, open(part, "ab" if done else "wb") as f:
        if done and resp.status != 206:  # server ignored Range, start over
            f.truncate(0)
            done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"\r      {dest.name}: {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
    print()

    if part.stat().st_size != total:
        sys.exit(f"error: {dest.name} is incomplete, run the script again to resume")
    part.rename(dest)
    print(f"done  {dest.name}")


def unzip(archive: Path) -> None:
    marker = archive.with_suffix(".unzipped")
    if marker.exists():
        print(f"skip  unzip {archive.name}")
        return
    print(f"unzip {archive.name} ...")
    with zipfile.ZipFile(archive) as z:
        z.extractall(archive.parent)
    marker.touch()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", action="store_true", help="also download and unzip the images (~3.6 GB)")
    parser.add_argument("--dry-run", action="store_true", help="list files and sizes, download nothing")
    args = parser.parse_args()

    files = ANNOTATIONS + (IMAGES if args.images else [])
    print(f"target: {OUT_DIR}")

    if args.dry_run:
        for name in files:
            print(f"  {name}  {remote_size(f'{BASE_URL}/{name}') / 1e6:,.0f} MB")
        return

    for name in files:
        dest = OUT_DIR / name
        download(f"{BASE_URL}/{name}", dest)
        if dest.suffix == ".zip":
            unzip(dest)


if __name__ == "__main__":
    main()
