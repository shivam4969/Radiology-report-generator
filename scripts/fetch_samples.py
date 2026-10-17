#!/usr/bin/env python3
"""Pull a small labelled set of public chest X-rays into samples/ with a labels.csv.

  python scripts/fetch_samples.py                 # NIH ChestX-ray14 sample via Kaggle, 40 images
  python scripts/fetch_samples.py --n 50          # more
  python scripts/fetch_samples.py --rsna          # also add the RSNA pneumonia subset (DICOM -> PNG; needs pydicom)
  python scripts/fetch_samples.py --from-dir ~/Downloads/nih-sample   # already downloaded by hand

Needs the Kaggle CLI configured (pip install kaggle; token in ~/.kaggle/kaggle.json or
KAGGLE_USERNAME / KAGGLE_KEY). Without it, the script prints what to download by hand.
Nothing here is committed: samples/*.png and samples/labels.csv are gitignored except the CC0
demo image. These datasets are for research only; check each one's terms (samples/README.md).
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
LABELS = SAMPLES / "labels.csv"
NIH_SAMPLE = "nih-chest-xrays/sample"  # Kaggle dataset: 5,606 images + sample_labels.csv
RSNA = "rsna-pneumonia-detection-challenge"  # Kaggle competition

# Keep a balanced mix across the labels the local classifier can produce.
WANTED = ["No Finding", "Pneumonia", "Effusion", "Cardiomegaly", "Pneumothorax", "Consolidation", "Nodule", "Mass", "Atelectasis", "Infiltration"]


def have_kaggle() -> bool:
    return shutil.which("kaggle") is not None and (
        (Path.home() / ".kaggle" / "kaggle.json").exists()
        or all(k in __import__("os").environ for k in ("KAGGLE_USERNAME", "KAGGLE_KEY"))
    )


def manual_instructions() -> None:
    print(
        f"""
Kaggle CLI is not configured, so nothing was downloaded. Two ways to get the set by hand:

NIH ChestX-ray14 sample (5,606 PNGs, 112,120-image parent set is public domain from NIH):
  a) Kaggle: https://www.kaggle.com/datasets/nih-chest-xrays/sample  (download, unzip)
     then: python scripts/fetch_samples.py --from-dir <unzipped folder>
  b) NIH Box (full set, 45 GB): https://nihcc.app.box.com/v/ChestXray-NIHCC
     download one images_0xx.tar.gz plus Data_Entry_2017.csv, untar, then --from-dir <folder>

RSNA pneumonia subset (DICOM, needs `pip install pydicom`):
  https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data  (accept the rules first)
  then: python scripts/fetch_samples.py --rsna --from-dir <unzipped folder>

To configure Kaggle instead: pip install kaggle, create an API token at kaggle.com/settings,
save it as ~/.kaggle/kaggle.json (chmod 600), and run this script again.
"""
    )


def kaggle_download(dataset: str, dest: Path, competition: bool = False) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "competitions" if competition else "datasets", "download", "-c" if competition else "-d", dataset, "-p", str(dest)]
    print("[fetch]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    for z in dest.glob("*.zip"):
        print(f"[fetch] unzipping {z.name}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)


def find_file(root: Path, name: str) -> Path | None:
    hits = list(root.rglob(name))
    return hits[0] if hits else None


def pick_nih(src: Path, n: int, rng: random.Random) -> list[tuple[Path, str]]:
    """Return [(png path, label)] from an NIH sample folder (sample_labels.csv or Data_Entry_2017.csv)."""
    csv_path = find_file(src, "sample_labels.csv") or find_file(src, "Data_Entry_2017.csv") or find_file(src, "Data_Entry_2017_v2020.csv")
    if not csv_path:
        raise SystemExit(f"no sample_labels.csv / Data_Entry_2017.csv under {src}")
    by_label: dict[str, list[str]] = {w: [] for w in WANTED}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("Image Index") or row.get("Image_Index")
            labels = (row.get("Finding Labels") or row.get("Finding_Labels") or "").split("|")
            if not name:
                continue
            # single-label rows only, so the agreement check in bench.mjs is fair
            if len(labels) == 1 and labels[0] in by_label:
                by_label[labels[0]].append(name)
    per = max(1, n // len(WANTED))
    chosen: list[tuple[Path, str]] = []
    for label in WANTED:
        names = by_label[label]
        rng.shuffle(names)
        for name in names[:per]:
            p = find_file(src, name)
            if p:
                chosen.append((p, label))
    return chosen[:n]


def pick_rsna(src: Path, n: int, rng: random.Random) -> list[tuple[Path, str]]:
    try:
        import pydicom
        from PIL import Image
    except ImportError:
        raise SystemExit("--rsna needs `pip install pydicom pillow`")
    labels_csv = find_file(src, "stage_2_train_labels.csv")
    if not labels_csv:
        raise SystemExit(f"stage_2_train_labels.csv not found under {src}")
    pos, neg = [], []
    seen = set()
    with open(labels_csv, newline="") as f:
        for row in csv.DictReader(f):
            pid = row["patientId"]
            if pid in seen:
                continue
            seen.add(pid)
            (pos if row["Target"] == "1" else neg).append(pid)
    rng.shuffle(pos)
    rng.shuffle(neg)
    out_dir = SAMPLES / "_rsna_png"
    out_dir.mkdir(exist_ok=True)
    chosen = []
    for pid, label in [(p, "Pneumonia") for p in pos[: n // 2]] + [(p, "No Finding") for p in neg[: n - n // 2]]:
        dcm = find_file(src, f"{pid}.dcm")
        if not dcm:
            continue
        arr = pydicom.dcmread(dcm).pixel_array
        png = out_dir / f"rsna-{pid}.png"
        Image.fromarray(arr).convert("L").save(png)
        chosen.append((png, label))
    return chosen


def write_samples(items: list[tuple[Path, str]]) -> None:
    SAMPLES.mkdir(exist_ok=True)
    existing: dict[str, str] = {}
    if LABELS.exists():
        with open(LABELS, newline="") as f:
            for row in csv.DictReader(f):
                existing[row["filename"]] = row["label"]
    for src, label in items:
        dst = SAMPLES / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        existing[dst.name] = label
    with open(LABELS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        for name in sorted(existing):
            w.writerow([name, existing[name]])
    print(f"[fetch] {len(items)} images added; {LABELS} now has {len(existing)} rows")
    counts: dict[str, int] = {}
    for _, label in items:
        counts[label] = counts.get(label, 0) + 1
    for label, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"   {label:<16} {c}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40, help="images to pull (30 to 50 is plenty for a demo)")
    ap.add_argument("--rsna", action="store_true", help="also pull the RSNA pneumonia subset")
    ap.add_argument("--from-dir", type=Path, help="use an already downloaded folder instead of the Kaggle CLI")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "downloads", help="where Kaggle zips land (gitignored)")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    if args.from_dir:
        src = args.from_dir.expanduser()
        items = pick_rsna(src, args.n, rng) if args.rsna else pick_nih(src, args.n, rng)
        write_samples(items)
        return

    if not have_kaggle():
        manual_instructions()
        sys.exit(2)

    nih_dir = args.cache / "nih-sample"
    if not find_file(nih_dir, "sample_labels.csv"):
        kaggle_download(NIH_SAMPLE, nih_dir)
    write_samples(pick_nih(nih_dir, args.n, rng))

    if args.rsna:
        rsna_dir = args.cache / "rsna"
        if not find_file(rsna_dir, "stage_2_train_labels.csv"):
            kaggle_download(RSNA, rsna_dir, competition=True)
        write_samples(pick_rsna(rsna_dir, max(10, args.n // 2), rng))

    print("\nNext: ENGINE=local npm run bench   (prints per-label agreement from samples/labels.csv)")


if __name__ == "__main__":
    main()
