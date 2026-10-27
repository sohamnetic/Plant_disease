"""Render build step: download the ONNX models (deploy_models.zip, a GitHub release asset)
and unpack them to deploy_models/. Standard library only.

  python src/fetch_models.py            # uses MODELS_URL / MODELS_SHA256 (defaults below)
  python src/fetch_models.py --pack     # (locally) zip deploy_models/ and print its SHA-256
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "deploy_models"
ZIP_NAME = "deploy_models.zip"
MODELS_URL = os.environ.get(
    "MODELS_URL", f"https://github.com/sohamnetic/Plant_disease/releases/download/models-v1/{ZIP_NAME}")
MODELS_SHA256 = os.environ.get("MODELS_SHA256", "")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch() -> None:
    if (DEST / "classifier" / "calibration.json").is_file():
        print("models already present:", DEST)
        return
    with tempfile.TemporaryDirectory() as tmp:
        z = Path(tmp) / ZIP_NAME
        print("downloading", MODELS_URL, flush=True)
        with urllib.request.urlopen(MODELS_URL, timeout=600) as r, z.open("wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        digest = sha256(z)
        if MODELS_SHA256 and digest != MODELS_SHA256:
            sys.exit(f"checksum mismatch: got {digest}, expected {MODELS_SHA256}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(DEST)
    print("models ready:", DEST, "sha256", digest)


def pack() -> None:
    out = ROOT / ZIP_NAME
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(DEST.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(DEST).as_posix())
    print(f"{out}  {out.stat().st_size / 2**20:.1f} MB  sha256 {sha256(out)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", action="store_true")
    pack() if ap.parse_args().pack else fetch()
