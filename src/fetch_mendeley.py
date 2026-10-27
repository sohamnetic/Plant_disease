"""Download a public Mendeley Data dataset into data_external/raw/<id>/ (gitignored).

Walks the dataset's folder tree through the public API, downloads every file,
and unzips archives in place. Already-downloaded files are skipped.

  python src/fetch_mendeley.py c2hrwntzgh --version 2 --only "Original Cauliflower Dataset.zip"
  python src/fetch_mendeley.py x26px3xnmy --version 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

API = "https://data.mendeley.com/public-api/datasets"
RAW_DIR = cfg.PROJECT_ROOT / "data_external" / "raw"
# The API returns 403 for urllib's default User-Agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (research dataset download)"}


def _open(url: str, timeout: int, attempts: int = 5):
    """Retry transient server errors (the API returns occasional 502/503)."""
    for i in range(attempts):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            code = getattr(exc, "code", None)
            if i == attempts - 1 or (code is not None and code < 500):
                raise
            time.sleep(2 ** i)


def _get_json(url: str):
    with _open(url, 60) as r:
        return json.load(r)


def _folders(ds: str, version: int) -> list[dict]:
    try:
        return _get_json(f"{API}/{ds}/folders/{version}")
    except Exception:
        return []


def _files(ds: str, version: int, folder_id: str) -> list[dict]:
    q = urllib.parse.urlencode({"folder_id": folder_id, "version": version})
    return _get_json(f"{API}/{ds}/files?{q}")


def _download(url: str, dest: Path, size: int) -> None:
    if dest.is_file() and dest.stat().st_size == size:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    # Large archives stall mid-transfer: resume from the partial file with HTTP Range.
    for _ in range(20):
        have = tmp.stat().st_size if tmp.is_file() else 0
        if have >= size:
            break
        headers = {**HEADERS, "Range": f"bytes={have}-"} if have else HEADERS
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                with tmp.open(mode) as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            print(f"  {dest.name}: interrupted at {have / 1e6:.0f} MB ({exc}); resuming")
            time.sleep(3)
    if tmp.stat().st_size != size:
        raise RuntimeError(f"{dest.name}: got {tmp.stat().st_size} of {size} bytes")
    tmp.replace(dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--only", action="append", default=[], help="download only these filenames")
    args = ap.parse_args()

    root = RAW_DIR / args.dataset
    folders = _folders(args.dataset, args.version)
    by_id = {f["id"]: f for f in folders}

    def rel_path(folder_id: str) -> Path:
        parts = []
        while folder_id in by_id:
            parts.append(by_id[folder_id]["name"].strip())
            folder_id = by_id[folder_id].get("parent_id")
        return Path(*reversed(parts)) if parts else Path()

    todo = ["root"] + [f["id"] for f in folders]
    n = 0
    for fid in todo:
        for f in _files(args.dataset, args.version, fid):
            name = f["filename"]
            if args.only and name not in args.only:
                continue
            dest = root / (rel_path(fid) if fid != "root" else Path()) / name
            _download(f["content_details"]["download_url"], dest, f["size"])
            n += 1
            if dest.suffix.lower() == ".zip":
                out = dest.with_suffix("")
                if not out.exists():
                    with zipfile.ZipFile(dest) as z:
                        z.extractall(out)
    print(f"{args.dataset}: {n} files -> {root}")


if __name__ == "__main__":
    main()
