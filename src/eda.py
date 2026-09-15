"""Dataset audit / EDA for leaf-disease classification.

Discovers every image on disk, never hardcodes counts, does not delete or
relabel files, and writes a leakage-safe frozen split (seed 42).

Run from the project root:
    python src/eda.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import imagehash
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


def ensure_dirs() -> None:
    cfg.EDA_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    cfg.SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def rel_to_root(path: Path) -> str:
    return path.relative_to(cfg.PROJECT_ROOT).as_posix()


def infer_source_type(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    joined = "/".join(parts)
    if "original dataset" in joined:
        return "original"
    if "augmented dataset" in joined:
        return "augmented"
    return "unknown"


def parse_filename_number(name: str) -> int | None:
    matches = re.findall(r"(\d+)", name)
    if not matches:
        return None
    return int(matches[-1])


def filename_pattern(name: str) -> str:
    return re.sub(r"\d+", "N", name)


def discover_images() -> pd.DataFrame:
    rows = []
    if not cfg.DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {cfg.DATASET_ROOT}")

    for path in sorted(cfg.DATASET_ROOT.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        parent_name = path.parent.name
        class_id = cfg.FOLDER_TO_CLASS.get(parent_name)
        rows.append(
            {
                "filepath": rel_to_root(path),
                "abs_path": str(path),
                "filename": path.name,
                "parent_folder": parent_name,
                "class": class_id if class_id else "UNMAPPED",
                "source_type": infer_source_type(path),
                "extension": ext,
                "supported_format": ext in cfg.IMAGE_EXTENSIONS,
                "file_size_bytes": path.stat().st_size,
                "filename_number": parse_filename_number(path.name),
                "filename_pattern": filename_pattern(path.name),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No files found under dataset/")
    return df


def analyze_image(abs_path: str) -> dict:
    out = {
        "readable": False,
        "corrupt": False,
        "corrupt_reason": "",
        "width": np.nan,
        "height": np.nan,
        "n_channels": np.nan,
        "mode": "",
        "aspect_ratio": np.nan,
        "mean_brightness": np.nan,
        "std_brightness": np.nan,
        "mean_saturation": np.nan,
        "phash": "",
        "dhash": "",
        "md5": "",
    }
    path = Path(abs_path)
    try:
        out["md5"] = hashlib.md5(path.read_bytes()).hexdigest()
    except OSError as exc:
        out["corrupt"] = True
        out["corrupt_reason"] = f"read_bytes:{exc}"
        return out

    try:
        with Image.open(path) as im:
            im.load()
            mode = im.mode
            width, height = im.size
            out["width"] = int(width)
            out["height"] = int(height)
            out["mode"] = mode
            out["aspect_ratio"] = float(width) / float(height) if height else np.nan
            out["n_channels"] = len(im.getbands())
            rgb = ImageOps.exif_transpose(im).convert("RGB")
            arr = np.asarray(rgb, dtype=np.float32)
            gray = arr.mean(axis=2)
            out["mean_brightness"] = float(gray.mean())
            out["std_brightness"] = float(gray.std())
            hsv = rgb.convert("HSV")
            sat = np.asarray(hsv, dtype=np.float32)[:, :, 1]
            out["mean_saturation"] = float(sat.mean())
            out["phash"] = str(imagehash.phash(rgb, hash_size=8))
            out["dhash"] = str(imagehash.dhash(rgb, hash_size=8))
            out["readable"] = True
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        out["corrupt"] = True
        out["corrupt_reason"] = str(exc)
    return out


def enrich_inventory(df: pd.DataFrame) -> pd.DataFrame:
    paths = df["abs_path"].tolist()
    analyses = []
    n = len(paths)
    for i, p in enumerate(paths, start=1):
        analyses.append(analyze_image(p))
        if i % 200 == 0 or i == n:
            print(f"  analyzed {i}/{n}")
    extra = pd.DataFrame(analyses)
    out = pd.concat([df.reset_index(drop=True), extra], axis=1)
    out["pixels"] = out["width"] * out["height"]
    out["very_low_resolution"] = (
        (out["width"] < cfg.VERY_LOW_RES_MIN_SIDE)
        | (out["height"] < cfg.VERY_LOW_RES_MIN_SIDE)
        | (out["pixels"] < cfg.VERY_LOW_RES_MIN_SIDE**2)
    ) & out["readable"]
    out["extremely_small"] = out["very_low_resolution"]
    # "extremely large" relative to this dataset: > 4000 on a side or > 12MP
    out["extremely_large"] = (
        (out["width"] > 4000) | (out["height"] > 4000) | (out["pixels"] > 12_000_000)
    ) & out["readable"]
    out["blank_or_near_blank"] = (out["std_brightness"] < cfg.BLANK_STD_MAX) & out["readable"]
    out["extreme_darkness"] = (out["mean_brightness"] < cfg.EXTREME_DARK_MEAN_MAX) & out["readable"]
    out["extreme_brightness"] = (
        out["mean_brightness"] > cfg.EXTREME_BRIGHT_MEAN_MIN
    ) & out["readable"]
    out["unusual_aspect_ratio"] = (
        (out["aspect_ratio"] > cfg.UNUSUAL_ASPECT_MAX)
        | (out["aspect_ratio"] < 1.0 / cfg.UNUSUAL_ASPECT_MAX)
    ) & out["readable"]
    out["wrong_color_mode"] = out["readable"] & ~out["mode"].isin(["RGB", "RGBA", "P", "L"])
    # RGBA / palette / grayscale are notable even if openable
    out["non_rgb"] = out["readable"] & (out["mode"] != "RGB")
    out["high_saturation_flag"] = (
        (out["mean_saturation"] >= cfg.HIGH_SATURATION_MEAN) & out["readable"]
    )
    out["augmentation_artifact_suspect"] = (
        (out["source_type"] == "augmented")
        & out["readable"]
        & (
            out["high_saturation_flag"]
            | (out["std_brightness"] > 90)
            | out["unusual_aspect_ratio"]
        )
    )
    out["unsupported_format"] = ~out["supported_format"]
    return out


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def hamming_hex(a: str, b: str) -> int:
    return bin(int(str(a), 16) ^ int(str(b), 16)).count("1")


def pair_is_near_duplicate(ph_a: str, ph_b: str, dh_a: str, dh_b: str) -> tuple[bool, int, int]:
    if not ph_a or not ph_b:
        return False, 99, 99
    ph = hamming_hex(ph_a, ph_b)
    dh = hamming_hex(dh_a, dh_b) if dh_a and dh_b else 99
    strong = ph <= cfg.PHASH_NEAR_DUP_MAX
    corroborated = (ph <= cfg.PHASH_LOOSE_MAX) and (dh <= cfg.DHASH_LOOSE_MAX)
    return (strong or corroborated), ph, dh


def _hash_to_uint64(values: list[str]) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros(len(values), dtype=np.uint64)
    valid = np.zeros(len(values), dtype=bool)
    for i, h in enumerate(values):
        if h:
            out[i] = np.uint64(int(str(h), 16))
            valid[i] = True
    return out, valid


def build_duplicate_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    uf = UnionFind(n)
    pair_rows = []

    readable_idx = df.index[df["readable"]].to_numpy()
    # Exact byte duplicates first
    md5_to_idxs: dict[str, list[int]] = defaultdict(list)
    for i in readable_idx.tolist():
        md5 = df.at[i, "md5"]
        if md5:
            md5_to_idxs[md5].append(i)
    for md5, idxs in md5_to_idxs.items():
        for a, b in zip(idxs, idxs[1:]):
            uf.union(a, b)
            pair_rows.append(
                {
                    "idx_a": a,
                    "idx_b": b,
                    "match_type": "exact_md5",
                    "phash_hamming": 0,
                    "dhash_hamming": 0,
                    "same_class": df.at[a, "class"] == df.at[b, "class"],
                }
            )

    ph_int, ph_ok = _hash_to_uint64(df["phash"].tolist())
    dh_int, dh_ok = _hash_to_uint64(df["dhash"].tolist())
    classes = df["class"].to_numpy()
    idx = np.arange(n)[ph_ok]
    print(f"  pairwise hash compare on {len(idx)} readable hashes...")
    for k, i in enumerate(idx):
        rest = idx[k + 1 :]
        if rest.size == 0:
            continue
        ph_ham = np.bitwise_count(ph_int[rest] ^ ph_int[i]).astype(np.int16)
        dh_ham = np.bitwise_count(dh_int[rest] ^ dh_int[i]).astype(np.int16)
        strong = ph_ham <= cfg.PHASH_NEAR_DUP_MAX
        corroborated = (ph_ham <= cfg.PHASH_LOOSE_MAX) & (dh_ham <= cfg.DHASH_LOOSE_MAX) & dh_ok[rest] & dh_ok[i]
        hit = np.where(strong | corroborated)[0]
        for t in hit.tolist():
            j = int(rest[t])
            ph = int(ph_ham[t])
            dh = int(dh_ham[t])
            uf.union(int(i), j)
            pair_rows.append(
                {
                    "idx_a": int(i),
                    "idx_b": j,
                    "match_type": "near_hash" if ph > 0 else "phash_identical",
                    "phash_hamming": ph,
                    "dhash_hamming": dh,
                    "same_class": classes[int(i)] == classes[j],
                }
            )
        if (k + 1) % 500 == 0:
            print(f"    compared {k + 1}/{len(idx)}")

    roots = [uf.find(i) for i in range(n)]
    # Compact group ids: g000001 ...
    root_to_gid: dict[int, str] = {}
    next_id = 1
    group_ids = []
    for i, r in enumerate(roots):
        if r not in root_to_gid:
            # singletons still get a unique group so splits stay group-aware
            root_to_gid[r] = f"g{next_id:06d}"
            next_id += 1
        group_ids.append(root_to_gid[r])

    grouped = df.copy()
    grouped["group_id"] = group_ids
    grouped["group_root"] = roots

    # Annotate group size / composition
    stats = grouped.groupby("group_id").agg(
        group_size=("filepath", "count"),
        n_classes=("class", "nunique"),
        n_original=("source_type", lambda s: int((s == "original").sum())),
        n_augmented=("source_type", lambda s: int((s == "augmented").sum())),
        classes=("class", lambda s: "|".join(sorted(set(s)))),
        source_types=("source_type", lambda s: "|".join(sorted(set(s)))),
    )
    grouped = grouped.merge(stats, on="group_id", how="left")
    grouped["is_duplicate_group"] = grouped["group_size"] > 1
    grouped["mixed_class_group"] = grouped["n_classes"] > 1
    grouped["orig_aug_paired_group"] = (grouped["n_original"] > 0) & (
        grouped["n_augmented"] > 0
    )

    pairs = pd.DataFrame(pair_rows)
    return grouped, pairs


def write_duplicate_groups_csv(df: pd.DataFrame) -> pd.DataFrame:
    dup = df[df["group_size"] > 1].copy()
    cols = [
        "group_id",
        "group_size",
        "classes",
        "source_types",
        "n_original",
        "n_augmented",
        "mixed_class_group",
        "orig_aug_paired_group",
        "filepath",
        "filename",
        "class",
        "source_type",
        "filename_number",
        "phash",
        "dhash",
        "md5",
        "width",
        "height",
    ]
    out = dup[cols].sort_values(["group_size", "group_id", "source_type", "filename"], ascending=[False, True, True, True])
    path = cfg.EDA_DIR / "duplicate_groups.csv"
    out.to_csv(path, index=False)
    return out


def quality_flags(row: pd.Series) -> list[str]:
    flags = []
    if row["corrupt"] or not row["readable"]:
        flags.append("corrupt")
    if row.get("unsupported_format"):
        flags.append("unsupported_format")
    if row.get("blank_or_near_blank"):
        flags.append("blank_or_near_blank")
    if row.get("extreme_brightness"):
        flags.append("extreme_brightness")
    if row.get("extreme_darkness"):
        flags.append("extreme_darkness")
    if row.get("very_low_resolution"):
        flags.append("very_low_resolution")
    if row.get("extremely_large"):
        flags.append("extremely_large")
    if row.get("unusual_aspect_ratio"):
        flags.append("unusual_aspect_ratio")
    if row.get("non_rgb"):
        flags.append(f"non_rgb_mode:{row.get('mode')}")
    if row.get("augmentation_artifact_suspect"):
        flags.append("augmentation_artifact_suspect")
    if row.get("is_duplicate_group") and row.get("group_size", 1) > 1:
        flags.append("in_duplicate_or_near_duplicate_group")
    if row.get("mixed_class_group"):
        flags.append("mixed_class_near_duplicate")
    return flags


def write_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    report = df.copy()
    report["flags"] = report.apply(lambda r: ";".join(quality_flags(r)), axis=1)
    report["n_flags"] = report["flags"].apply(lambda s: 0 if s == "" else len(s.split(";")))
    cols = [
        "filepath",
        "filename",
        "class",
        "source_type",
        "group_id",
        "group_size",
        "readable",
        "corrupt",
        "corrupt_reason",
        "mode",
        "width",
        "height",
        "aspect_ratio",
        "mean_brightness",
        "std_brightness",
        "mean_saturation",
        "flags",
        "n_flags",
    ]
    out = report[cols].sort_values(["n_flags", "class", "filepath"], ascending=[False, True, True])
    out.to_csv(cfg.EDA_DIR / "data_quality_report.csv", index=False)
    return out


def count_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cls in cfg.CLASS_ORDER:
        sub = df[df["class"] == cls]
        orig = int((sub["source_type"] == "original").sum())
        aug = int((sub["source_type"] == "augmented").sum())
        rows.append(
            {
                "class": cls,
                "original": orig,
                "augmented": aug,
                "total": orig + aug,
                "pct_of_dataset": 100.0 * (orig + aug) / len(df) if len(df) else 0.0,
            }
        )
    table = pd.DataFrame(rows)
    table.loc["Total"] = {
        "class": "Total",
        "original": int(table["original"].sum()),
        "augmented": int(table["augmented"].sum()),
        "total": int(table["total"].sum()),
        "pct_of_dataset": 100.0,
    }
    return table


def save_count_ascii(table: pd.DataFrame) -> str:
    lines = [
        "Class                 Original   Augmented   Total",
        "---------------------------------------------------",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{str(row['class']):<22}{int(row['original']):8d}{int(row['augmented']):12d}{int(row['total']):9d}"
        )
    text = "\n".join(lines) + "\n"
    (cfg.EDA_DIR / "class_counts_table.txt").write_text(text, encoding="utf-8")
    table.to_csv(cfg.EDA_DIR / "class_counts_table.csv", index=False)
    return text


def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df["class"].value_counts().reindex(cfg.CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index.astype(str), counts.values, color="#2E7D32")
    ax.set_title("Class distribution (all images)")
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    plt.xticks(rotation=20, ha="right")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(cfg.EDA_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)


def plot_original_vs_augmented(df: pd.DataFrame) -> None:
    data = (
        df.groupby(["class", "source_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(cfg.CLASS_ORDER)
        .reindex(columns=["original", "augmented"], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(data.index))
    w = 0.38
    ax.bar(x - w / 2, data["original"], w, label="original", color="#1565C0")
    ax.bar(x + w / 2, data["augmented"], w, label="augmented", color="#EF6C00")
    ax.set_xticks(x)
    ax.set_xticklabels(data.index, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Original vs augmented counts by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.EDA_DIR / "original_vs_augmented.png", dpi=150)
    plt.close(fig)

    # overall pie-ish bar
    fig, ax = plt.subplots(figsize=(5, 5))
    overall = df["source_type"].value_counts().reindex(["original", "augmented"])
    ax.bar(overall.index.astype(str), overall.values, color=["#1565C0", "#EF6C00"])
    ax.set_title("Original vs augmented (overall)")
    for i, v in enumerate(overall.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(cfg.EDA_DIR / "original_vs_augmented_overall.png", dpi=150)
    plt.close(fig)


def _load_thumb(abs_path: str, size: int = 160) -> Image.Image:
    try:
        with Image.open(abs_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((size, size))
            canvas = Image.new("RGB", (size, size), (20, 20, 20))
            x = (size - im.width) // 2
            y = (size - im.height) // 2
            canvas.paste(im, (x, y))
            return canvas
    except Exception:
        canvas = Image.new("RGB", (size, size), (80, 0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, size // 2), "UNREADABLE", fill=(255, 255, 255))
        return canvas


def save_image_grid(
    paths: list[str],
    labels: list[str],
    out_path: Path,
    title: str,
    cols: int = 5,
    thumb: int = 160,
) -> None:
    if not paths:
        return
    n = len(paths)
    rows = int(np.ceil(n / cols))
    pad = 8
    label_h = 36
    cell_w, cell_h = thumb, thumb + label_h
    grid = Image.new(
        "RGB",
        (cols * (cell_w + pad) + pad, rows * (cell_h + pad) + pad + 40),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((pad, 10), title, fill=(0, 0, 0), font=font)
    for i, (p, lab) in enumerate(zip(paths, labels)):
        r, c = divmod(i, cols)
        x = pad + c * (cell_w + pad)
        y = 40 + pad + r * (cell_h + pad)
        grid.paste(_load_thumb(p, thumb), (x, y))
        # wrap-ish filename
        text = lab if len(lab) <= 28 else lab[:25] + "..."
        draw.text((x, y + thumb + 4), text, fill=(20, 20, 20), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def _zip_labels(frame: pd.DataFrame, kind: str) -> list[str]:
    """Build grid captions without itertuples (column name `class` is reserved)."""
    cls = frame["class"].astype(str).tolist()
    src = frame["source_type"].astype(str).str[:3].tolist()
    names = frame["filename"].astype(str).tolist()
    if kind == "src_name":
        return [f"{s}|{n}" for s, n in zip(src, names)]
    if kind == "cls_src_name":
        return [f"{c}|{s}|{n}" for c, s, n in zip(cls, src, names)]
    gids = frame["group_id"].astype(str).tolist()
    if kind == "gid_cls_src_name":
        return [f"{g}|{c}|{s}|{n}" for g, c, s, n in zip(gids, cls, src, names)]
    if kind == "gid_src_name":
        return [f"{g}|{s}|{n}" for g, s, n in zip(gids, src, names)]
    raise ValueError(kind)


def random_class_grids(df: pd.DataFrame, rng: random.Random) -> None:
    readable = df[df["readable"]]
    for cls in cfg.CLASS_ORDER:
        for source in ("original", "augmented"):
            sub = readable[(readable["class"] == cls) & (readable["source_type"] == source)]
            n = min(20, len(sub))
            if n == 0:
                continue
            sample = sub.sample(n=n, random_state=cfg.RANDOM_SEED)
            save_image_grid(
                sample["abs_path"].tolist(),
                _zip_labels(sample, "src_name"),
                cfg.EDA_DIR / f"grid_{cls}_{source}.png",
                title=f"{cls} — {source} (n={n} of {len(sub)})",
            )


def contact_sheet_random(df: pd.DataFrame) -> None:
    readable = df[df["readable"]]
    for cls in cfg.CLASS_ORDER:
        sub = readable[readable["class"] == cls]
        n = min(24, len(sub))
        sample = sub.sample(n=n, random_state=cfg.RANDOM_SEED + 1)
        save_image_grid(
            sample["abs_path"].tolist(),
            _zip_labels(sample, "cls_src_name"),
            cfg.REVIEW_DIR / f"random_{cls}.png",
            title=f"Random sample — {cls}",
            cols=6,
        )


def contact_sheet_suspicious(df: pd.DataFrame) -> None:
    mask = (
        df["corrupt"]
        | df["blank_or_near_blank"]
        | df["extreme_brightness"]
        | df["extreme_darkness"]
        | df["very_low_resolution"]
        | df["unusual_aspect_ratio"]
        | df["augmentation_artifact_suspect"]
        | df["mixed_class_group"]
        | df["non_rgb"]
    )
    sub = df[mask & df["readable"]]
    if sub.empty:
        # still write an empty marker
        (cfg.REVIEW_DIR / "suspicious_images_none.txt").write_text(
            "No readable images matched quality-suspicion heuristics.\n",
            encoding="utf-8",
        )
        return
    # cap for a usable sheet
    sub = sub.sort_values(["n_classes", "std_brightness", "filepath"]).head(60)
    save_image_grid(
        sub["abs_path"].tolist(),
        _zip_labels(sub, "cls_src_name"),
        cfg.REVIEW_DIR / "suspicious_images.png",
        title="Suspicious / quality-flagged images (heuristic, not deleted)",
        cols=6,
    )
    sub[
        [
            "filepath",
            "class",
            "source_type",
            "filename",
            "mode",
            "width",
            "height",
            "mean_brightness",
            "std_brightness",
            "mean_saturation",
            "group_id",
        ]
    ].to_csv(cfg.REVIEW_DIR / "suspicious_images.csv", index=False)


def contact_sheet_duplicate_groups(df: pd.DataFrame) -> None:
    dups = df[(df["group_size"] > 1) & df["readable"]].copy()
    if dups.empty:
        (cfg.REVIEW_DIR / "duplicate_groups_none.txt").write_text(
            "No multi-image groups found.\n", encoding="utf-8"
        )
        return
    # Show the largest groups, up to 12 groups, up to 8 images each
    gids = (
        dups.drop_duplicates("group_id")
        .sort_values("group_size", ascending=False)["group_id"]
        .head(12)
        .tolist()
    )
    paths, labels = [], []
    for gid in gids:
        g = dups[dups["group_id"] == gid].head(8)
        paths.extend(g["abs_path"].tolist())
        labels.extend(_zip_labels(g, "gid_cls_src_name"))
    save_image_grid(
        paths,
        labels,
        cfg.REVIEW_DIR / "duplicate_groups_samples.png",
        title="Sample duplicate / near-duplicate groups (not deleted)",
        cols=8,
        thumb=140,
    )

    # original<->augmented pairing examples
    paired = dups[dups["orig_aug_paired_group"]]
    if not paired.empty:
        gids = (
            paired.drop_duplicates("group_id")
            .sort_values("group_size", ascending=False)["group_id"]
            .head(10)
            .tolist()
        )
        paths, labels = [], []
        for gid in gids:
            g = paired[paired["group_id"] == gid]
            orig = g[g["source_type"] == "original"].head(2)
            aug = g[g["source_type"] == "augmented"].head(6)
            show = pd.concat([orig, aug])
            paths.extend(show["abs_path"].tolist())
            labels.extend(_zip_labels(show, "gid_src_name"))
        save_image_grid(
            paths,
            labels,
            cfg.REVIEW_DIR / "orig_aug_paired_groups.png",
            title="Original <-> augmented paired groups (leakage-relevant)",
            cols=8,
            thumb=140,
        )


def contact_sheet_potential_mislabels(df: pd.DataFrame) -> None:
    mixed = df[df["mixed_class_group"] & df["readable"]]
    if mixed.empty:
        (cfg.REVIEW_DIR / "potential_mislabels_none.txt").write_text(
            "No mixed-class near-duplicate groups found by hash matching.\n",
            encoding="utf-8",
        )
        return
    mixed_show = mixed.head(48)
    paths = mixed_show["abs_path"].tolist()
    labels = _zip_labels(mixed_show, "gid_cls_src_name")
    save_image_grid(
        paths,
        labels,
        cfg.REVIEW_DIR / "potential_mislabels.png",
        title="Potential mislabels: near-duplicate hashes across classes",
        cols=6,
    )
    mixed[
        ["group_id", "filepath", "class", "source_type", "filename", "phash"]
    ].to_csv(cfg.REVIEW_DIR / "potential_mislabels.csv", index=False)


def filename_relationship_report(df: pd.DataFrame) -> dict:
    """Check whether original/augmented filenames share a usable ID scheme."""
    report = {"per_class": {}, "conclusion": ""}
    for cls in cfg.CLASS_ORDER:
        orig = df[(df["class"] == cls) & (df["source_type"] == "original")]
        aug = df[(df["class"] == cls) & (df["source_type"] == "augmented")]
        o_nums = set(orig["filename_number"].dropna().astype(int))
        a_nums = set(aug["filename_number"].dropna().astype(int))
        report["per_class"][cls] = {
            "n_original": int(len(orig)),
            "n_augmented": int(len(aug)),
            "orig_num_min": int(min(o_nums)) if o_nums else None,
            "orig_num_max": int(max(o_nums)) if o_nums else None,
            "aug_num_min": int(min(a_nums)) if a_nums else None,
            "aug_num_max": int(max(a_nums)) if a_nums else None,
            "orig_unique_numbers": len(o_nums),
            "aug_unique_numbers": len(a_nums),
            "number_overlap_count": len(o_nums & a_nums),
            "orig_filename_patterns": orig["filename_pattern"].value_counts().to_dict(),
            "aug_filename_patterns": aug["filename_pattern"].value_counts().to_dict(),
            "ratio_aug_per_orig": (len(aug) / len(orig)) if len(orig) else None,
        }
    report["conclusion"] = (
        "Augmented filename numbers extend well beyond original counts "
        "(e.g. sequential IDs in the augmented folders), so filename numbers "
        "are NOT a safe original<->augmented join key. Grouping uses perceptual "
        "hashes (phash/dhash) plus exact MD5. Same-number original files with "
        "both .jpg and .jpeg are treated as possible duplicates and hashed."
    )
    (cfg.EDA_DIR / "filename_relationship_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def dimension_summary(df: pd.DataFrame) -> dict:
    r = df[df["readable"]]
    summary = {
        "n_readable": int(len(r)),
        "width_min": int(r["width"].min()) if len(r) else None,
        "width_median": float(r["width"].median()) if len(r) else None,
        "width_max": int(r["width"].max()) if len(r) else None,
        "height_min": int(r["height"].min()) if len(r) else None,
        "height_median": float(r["height"].median()) if len(r) else None,
        "height_max": int(r["height"].max()) if len(r) else None,
        "aspect_min": float(r["aspect_ratio"].min()) if len(r) else None,
        "aspect_median": float(r["aspect_ratio"].median()) if len(r) else None,
        "aspect_max": float(r["aspect_ratio"].max()) if len(r) else None,
        "unique_wh": int(r.dropna(subset=["width", "height"]).apply(lambda x: (int(x["width"]), int(x["height"])), axis=1).nunique())
        if len(r)
        else 0,
        "top_resolutions": r.groupby(["width", "height"]).size().sort_values(ascending=False).head(15).to_dict()
        if len(r)
        else {},
        "mode_counts": r["mode"].value_counts().to_dict() if len(r) else {},
        "extension_counts": df["extension"].value_counts().to_dict(),
        "channel_counts": r["n_channels"].value_counts().to_dict() if len(r) else {},
    }
    # stringify tuple keys
    summary["top_resolutions"] = {
        f"{int(w)}x{int(h)}": int(c) for (w, h), c in summary["top_resolutions"].items()
    }
    summary["channel_counts"] = {str(k): int(v) for k, v in summary["channel_counts"].items()}
    (cfg.EDA_DIR / "image_size_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


def plot_size_histograms(df: pd.DataFrame) -> None:
    r = df[df["readable"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].hist(r["width"], bins=30, color="#1565C0")
    axes[0].set_title("Width")
    axes[1].hist(r["height"], bins=30, color="#2E7D32")
    axes[1].set_title("Height")
    axes[2].hist(r["aspect_ratio"].clip(0, 3), bins=30, color="#6A1B9A")
    axes[2].set_title("Aspect ratio (clipped at 3)")
    fig.suptitle("Image dimension distributions")
    fig.tight_layout()
    fig.savefig(cfg.EDA_DIR / "image_dimension_histograms.png", dpi=150)
    plt.close(fig)


def stratified_group_split(df: pd.DataFrame) -> pd.DataFrame:
    """Assign whole groups to train/val/test, stratified by majority class.

    Two-phase (still group-atomic, seed 42):
    1) Assign groups that contain originals so *original* counts per class
       approximate 70/15/15 — needed because Experiment A is original-only
       and augs dominate some classes (especially black_rot).
    2) Assign remaining (augmented-only) groups so *all-image* counts per
       class still approximate 70/15/15, using leftover capacity.
    """
    rng = np.random.RandomState(cfg.RANDOM_SEED)
    group_meta = (
        df.groupby("group_id")
        .agg(
            n=("filepath", "count"),
            n_original=("source_type", lambda s: int((s == "original").sum())),
            majority_class=("class", lambda s: s.value_counts().idxmax()),
            n_classes=("class", "nunique"),
        )
        .reset_index()
    )

    split_of_group: dict[str, str] = {}
    for cls in cfg.CLASS_ORDER:
        sub = group_meta[group_meta["majority_class"] == cls].copy()
        n_cls = int(sub["n"].sum())
        orig_groups = sub[sub["n_original"] > 0]
        aug_only = sub[sub["n_original"] == 0]

        orig_gids = orig_groups["group_id"].tolist()
        orig_sizes = dict(zip(orig_groups["group_id"], orig_groups["n_original"]))
        orig_total = int(sum(orig_sizes.values())) or 1
        # Largest original-groups first so val/test are not starved by a leftover giant cluster.
        orig_gids.sort(key=lambda g: (-orig_sizes[g], g))
        filled_o = {"train": 0.0, "val": 0.0, "test": 0.0}
        targets_o = {
            "train": cfg.TRAIN_FRAC * orig_total,
            "val": cfg.VAL_FRAC * orig_total,
            "test": cfg.TEST_FRAC * orig_total,
        }
        orig_assign: dict[str, str] = {}
        for gid in orig_gids:
            remaining = {k: targets_o[k] - filled_o[k] for k in targets_o}
            choice = max(remaining, key=lambda k: remaining[k])
            orig_assign[gid] = choice
            filled_o[choice] += orig_sizes[gid]
        split_of_group.update(orig_assign)

        filled = {"train": 0, "val": 0, "test": 0}
        n_map = dict(zip(sub["group_id"], sub["n"]))
        for gid, sp in orig_assign.items():
            filled[sp] += int(n_map[gid])
        targets = {
            "train": cfg.TRAIN_FRAC * n_cls,
            "val": cfg.VAL_FRAC * n_cls,
            "test": cfg.TEST_FRAC * n_cls,
        }

        aug_gids = aug_only["group_id"].tolist()
        rng.shuffle(aug_gids)
        for gid in aug_gids:
            remaining = {k: targets[k] - filled[k] for k in targets}
            choice = max(remaining, key=lambda k: remaining[k])
            split_of_group[gid] = choice
            filled[choice] += int(n_map[gid])

    leftover = group_meta[~group_meta["group_id"].isin(split_of_group)]
    for gid in leftover["group_id"].tolist():
        split_of_group[gid] = "train"

    out = df.copy()
    out["split"] = out["group_id"].map(split_of_group)

    nunique = out.groupby("group_id")["split"].nunique()
    if int(nunique.max()) != 1:
        bad = nunique[nunique > 1].index.tolist()
        raise RuntimeError(f"Group leaked across splits: {bad[:10]}")
    return out


def write_splits(df: pd.DataFrame) -> dict:
    cols = ["filepath", "class", "source_type", "group_id"]
    manifest_rows = {}
    for split in ("train", "val", "test"):
        part = df[df["split"] == split][cols].sort_values(["class", "filepath"])
        path = cfg.SPLITS_DIR / f"{split}.csv"
        part.to_csv(path, index=False)
        manifest_rows[split] = {
            "n_images": int(len(part)),
            "n_groups": int(part["group_id"].nunique()),
            "per_class": part["class"].value_counts().reindex(cfg.CLASS_ORDER).fillna(0).astype(int).to_dict(),
            "per_source": part["source_type"].value_counts().to_dict(),
        }

    # freeze a checksum so later experiments can detect accidental regeneration
    checksum_src = pd.concat(
        [
            pd.read_csv(cfg.SPLITS_DIR / "train.csv"),
            pd.read_csv(cfg.SPLITS_DIR / "val.csv"),
            pd.read_csv(cfg.SPLITS_DIR / "test.csv"),
        ]
    )
    payload = checksum_src.sort_values(["filepath"]).to_csv(index=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": cfg.RANDOM_SEED,
        "fractions": {"train": cfg.TRAIN_FRAC, "val": cfg.VAL_FRAC, "test": cfg.TEST_FRAC},
        "method": (
            "Group-atomic split, seed 42. Per class: (1) assign groups that "
            "contain originals by largest-original-count-first greedy fill toward "
            "70/15/15 of original images (prevents val/test starvation when a few "
            "original clusters are large); (2) shuffle augmented-only groups with "
            "RandomState(42) and assign each to the split with the most remaining "
            "all-image capacity toward 70/15/15. Mixed original+augmented groups "
            "stay together."
        ),
        "canonical": True,
        "do_not_regenerate": True,
        "sha256_of_concatenated_csvs": digest,
        "splits": manifest_rows,
        "class_proportions": {},
    }
    for name, part in [
        ("full", df),
        ("train", df[df["split"] == "train"]),
        ("val", df[df["split"] == "val"]),
        ("test", df[df["split"] == "test"]),
    ]:
        vc = part["class"].value_counts().reindex(cfg.CLASS_ORDER).fillna(0)
        manifest["class_proportions"][name] = {
            "counts": vc.astype(int).to_dict(),
            "pct": (100.0 * vc / vc.sum()).round(3).to_dict() if vc.sum() else {},
        }
        orig = part[part["source_type"] == "original"]["class"].value_counts().reindex(cfg.CLASS_ORDER).fillna(0)
        manifest.setdefault("original_only_proportions", {})
        manifest["original_only_proportions"][name] = {
            "counts": orig.astype(int).to_dict(),
            "pct": (100.0 * orig / orig.sum()).round(3).to_dict() if orig.sum() else {},
            "n": int(orig.sum()),
        }
    (cfg.SPLITS_DIR / "SPLIT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (cfg.SPLITS_DIR / "README.md").write_text(
        "This split is CANONICAL (RANDOM_SEED=42, group-aware).\n"
        "Do not regenerate during later experiments. Do not use test.csv "
        "until explicitly instructed.\n",
        encoding="utf-8",
    )
    return manifest


def write_eda_summary(
    df: pd.DataFrame,
    table_text: str,
    dim: dict,
    manifest: dict,
    fname_report: dict,
) -> str:
    n_dup_groups = int((df.drop_duplicates("group_id")["group_size"] > 1).sum())
    n_near_paired = int(
        df.drop_duplicates("group_id")["orig_aug_paired_group"].sum()
    )
    n_exact_md5_groups = int(
        df[df["md5"] != ""]
        .groupby("md5")
        .filter(lambda g: len(g) > 1)["md5"]
        .nunique()
        if df["md5"].ne("").any()
        else 0
    )
    multi = df[df["group_size"] > 1]
    n_images_in_multi = int(len(multi))
    mixed = int(df.drop_duplicates("group_id")["mixed_class_group"].sum())

    quality = {
        "total_images": int(len(df)),
        "corrupt": int((df["corrupt"] | ~df["readable"]).sum()),
        "duplicates_exact_md5_groups": n_exact_md5_groups,
        "near_duplicate_or_duplicate_groups": n_dup_groups,
        "images_in_multi_image_groups": n_images_in_multi,
        "very_low_resolution": int(df["very_low_resolution"].fillna(False).sum()),
        "extreme_brightness": int(df["extreme_brightness"].fillna(False).sum()),
        "extreme_darkness": int(df["extreme_darkness"].fillna(False).sum()),
        "blank_or_near_blank": int(df["blank_or_near_blank"].fillna(False).sum()),
        "unusual_aspect_ratio": int(df["unusual_aspect_ratio"].fillna(False).sum()),
        "non_rgb": int(df["non_rgb"].fillna(False).sum()),
        "unsupported_format": int(df["unsupported_format"].sum()),
        "augmentation_artifact_suspect": int(
            df["augmentation_artifact_suspect"].fillna(False).sum()
        ),
        "mixed_class_near_duplicate_groups": mixed,
        "orig_aug_paired_groups": n_near_paired,
        "unmapped_class_files": int((df["class"] == "UNMAPPED").sum()),
        "unknown_source_type": int((df["source_type"] == "unknown").sum()),
    }
    (cfg.EDA_DIR / "quality_summary.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )

    lines = []
    lines.append("LEAF DISEASE DATASET — EDA SUMMARY")
    lines.append(f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Random seed: {cfg.RANDOM_SEED}")
    lines.append("")
    lines.append(table_text)
    lines.append("")
    lines.append("QUALITY COUNTS (flags only — nothing deleted or relabeled)")
    for k, v in quality.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("IMAGE SIZE")
    lines.append(
        f"  width {dim['width_min']}–{dim['width_max']} (median {dim['width_median']})"
    )
    lines.append(
        f"  height {dim['height_min']}–{dim['height_max']} (median {dim['height_median']})"
    )
    lines.append(
        f"  aspect {dim['aspect_min']:.3f}–{dim['aspect_max']:.3f} (median {dim['aspect_median']:.3f})"
    )
    lines.append(f"  unique WxH: {dim['unique_wh']}")
    lines.append(f"  modes: {dim['mode_counts']}")
    lines.append(f"  extensions: {dim['extension_counts']}")
    lines.append("")
    lines.append("FILENAME vs HASH GROUPING")
    lines.append("  " + fname_report["conclusion"])
    lines.append(
        f"  Multi-image groups: {n_dup_groups} (images in them: {n_images_in_multi})"
    )
    lines.append(f"  Original<->augmented paired groups: {n_near_paired}")
    lines.append(f"  Mixed-class groups (possible mislabels): {mixed}")
    lines.append("")
    lines.append("SPLITS (group-aware, seed 42, CANONICAL)")
    for split in ("train", "val", "test"):
        info = manifest["splits"][split]
        lines.append(
            f"  {split}: {info['n_images']} images / {info['n_groups']} groups"
        )
        lines.append(f"    per class: {info['per_class']}")
        lines.append(f"    per source: {info['per_source']}")
    lines.append("")
    lines.append("CLASS PROPORTIONS (%)")
    for name, block in manifest["class_proportions"].items():
        lines.append(f"  {name}: {block['pct']}")
    lines.append("ORIGINAL-ONLY COUNTS BY SPLIT")
    for name, block in manifest.get("original_only_proportions", {}).items():
        lines.append(f"  {name} n={block['n']}: {block['counts']}")
    text = "\n".join(lines) + "\n"
    (cfg.EDA_DIR / "eda_summary.txt").write_text(text, encoding="utf-8")
    return text


def write_dataset_structure_md(df: pd.DataFrame, dim: dict, fname_report: dict) -> None:
    orig_root = cfg.ORIGINAL_ROOT
    aug_root = cfg.AUGMENTED_ROOT
    md = []
    md.append("# Dataset structure")
    md.append("")
    md.append("Fresh inspection of `D:\\project-1`. Counts are computed from disk, not assumed.")
    md.append("")
    md.append("## Project layout (after this audit)")
    md.append("")
    md.append("```")
    md.append("project-1/")
    md.append("  dataset/")
    md.append("    Original Dataset/Original Dataset/{class}/   # originals")
    md.append("    Augmented Dataset/Augmented Dataset/{class}/ # augmented")
    md.append("  src/")
    md.append("    config.py")
    md.append("    eda.py")
    md.append("  outputs/eda/")
    md.append("  outputs/review/")
    md.append("  splits/")
    md.append("  DATASET_STRUCTURE.md")
    md.append("```")
    md.append("")
    md.append("There was **no existing code, notebooks, requirements, or README** in the")
    md.append("repository at inspection time — only the `dataset/` tree.")
    md.append("")
    md.append("## Nested folders")
    md.append("")
    md.append("The publisher wrapped each subset twice:")
    md.append("")
    md.append(f"- Originals: `{orig_root.relative_to(cfg.PROJECT_ROOT).as_posix()}`")
    md.append(f"- Augmented: `{aug_root.relative_to(cfg.PROJECT_ROOT).as_posix()}`")
    md.append("")
    md.append("Class folders (exact names on disk → canonical ids):")
    md.append("")
    for folder, cid in cfg.FOLDER_TO_CLASS.items():
        md.append(f"- `{folder}` → `{cid}`")
    md.append("")
    md.append("No other class folders were found. Files whose parent folder is not in this")
    md.append("map would be labeled `UNMAPPED` in the inventory.")
    md.append("")
    md.append("## File formats")
    md.append("")
    md.append("| location | typical format | notes |")
    md.append("|---|---|---|")
    md.append("| original / Bacterial spot rot | `.jpg` | single pattern `Bacterial Spot. (N).jpg` |")
    md.append("| original / Black Rot | mostly `.jpeg`, 1× `.jpg`, 1× `.jfif` | `Black rot(N).*`, numbers 2–63 (no 1) |")
    md.append("| original / Downy Mildew | `.jpg` and `.jpeg` | same `Downy Mildew. (N)` stem can exist in both extensions |")
    md.append("| original / No disease | `.jpg` and `.jpeg` | same `No disease. (N)` stem can exist in both extensions |")
    md.append("| all augmented classes | `.png` | two naming styles in some classes (spaced vs concatenated) |")
    md.append("")
    md.append("## Original vs augmented independence")
    md.append("")
    md.append(fname_report["conclusion"])
    md.append("")
    md.append("Per-class filename stats:")
    md.append("")
    md.append("```json")
    md.append(json.dumps(fname_report["per_class"], indent=2, default=str))
    md.append("```")
    md.append("")
    md.append("Therefore: **do not join original N to augmented N by filename number.**")
    md.append("Leakage control uses MD5 exact-duplicates and perceptual-hash connected")
    md.append("components (`group_id`). All members of a `group_id` stay in one split.")
    md.append("")
    md.append("## Inventory size")
    md.append("")
    md.append(f"- Total files under `dataset/`: **{len(df)}**")
    md.append(f"- Readable images: **{int(df['readable'].sum())}**")
    md.append(f"- Corrupt / unreadable: **{int((~df['readable']).sum())}**")
    md.append("")
    md.append("## Image geometry (readable only)")
    md.append("")
    md.append(f"- Width: {dim['width_min']}–{dim['width_max']} (median {dim['width_median']})")
    md.append(f"- Height: {dim['height_min']}–{dim['height_max']} (median {dim['height_median']})")
    md.append(
        f"- Aspect ratio: {dim['aspect_min']:.4f}–{dim['aspect_max']:.4f} (median {dim['aspect_median']:.4f})"
    )
    md.append(f"- Unique resolutions: {dim['unique_wh']}")
    md.append(f"- Color modes: `{dim['mode_counts']}`")
    md.append(f"- Extensions: `{dim['extension_counts']}`")
    md.append("")
    md.append("## What this audit does **not** do")
    md.append("")
    md.append("- No files were deleted, moved, or relabeled.")
    md.append("- No model training was started.")
    md.append("- `splits/test.csv` is written for later use but must not be evaluated yet.")
    md.append("")
    (cfg.PROJECT_ROOT / "DATASET_STRUCTURE.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    print("Discovering files...")
    discovered = discover_images()
    print(f"  found {len(discovered)} files")
    inv_path = cfg.EDA_DIR / "image_inventory.csv"
    reuse = False
    if inv_path.exists():
        cached = pd.read_csv(inv_path)
        if (
            len(cached) == len(discovered)
            and "phash" in cached.columns
            and "group_id" in cached.columns
            and set(cached["filepath"]) == set(discovered["filepath"])
        ):
            print("  reusing existing image_inventory.csv (same file set)")
            df = cached
            reuse = True
    if not reuse:
        print("Reading images, computing hashes and quality metrics...")
        df = enrich_inventory(discovered)
        print("Building duplicate / near-duplicate groups...")
        df, pairs = build_duplicate_groups(df)
        if not pairs.empty:
            pairs.to_csv(cfg.EDA_DIR / "hash_match_pairs.csv", index=False)
        df.to_csv(inv_path, index=False)
        write_duplicate_groups_csv(df)
        write_quality_report(df)
    else:
        write_duplicate_groups_csv(df)
        write_quality_report(df)

    table = count_table(df)
    table_text = save_count_ascii(table)
    print(table_text)

    print("Writing plots and contact sheets...")
    plot_class_distribution(df)
    plot_original_vs_augmented(df)
    plot_size_histograms(df)
    rng = random.Random(cfg.RANDOM_SEED)
    random_class_grids(df, rng)
    contact_sheet_random(df)
    contact_sheet_suspicious(df)
    contact_sheet_duplicate_groups(df)
    contact_sheet_potential_mislabels(df)

    fname_report = filename_relationship_report(df)
    dim = dimension_summary(df)

    print("Creating leakage-safe frozen split...")
    df = stratified_group_split(df)
    df.to_csv(cfg.EDA_DIR / "image_inventory_with_split.csv", index=False)
    manifest = write_splits(df)
    summary = write_eda_summary(df, table_text, dim, manifest, fname_report)
    write_dataset_structure_md(df, dim, fname_report)
    print(summary)
    print("Done. Split is frozen under splits/. Do not train yet.")


if __name__ == "__main__":
    main()
