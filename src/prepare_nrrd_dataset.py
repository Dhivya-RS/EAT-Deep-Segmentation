"""
Converts the NRRD cardiac-CT volumes + EAT ground-truth volumes (Siriapisith et al. 2021,
figshare 10.6084/m9.figshare.17039813) into 2-D PNG slices for this project.

STEP 1 - look at what is inside (always do this first):
    python src/prepare_nrrd_dataset.py --src "C:/Users/me/Downloads/dataset" --inspect

STEP 2 - create the training data (example: 12 patients x 8 slices, non-contrast training folder):
    python src/prepare_nrrd_dataset.py --src "C:/Users/me/Downloads/dataset" ^
           --filter train_NC --patients 12 --slices 8 --clear

Output: data/images/P01_023.png, data/masks/P01_023.png ...   (P01 = patient, 023 = slice number)
        data/pairs.csv  (which source file each slice came from)
        data/dataset_info.json  (pixel spacing / slice thickness read from the NRRD header)

Options
  --filter TEXT       only use volumes whose path contains TEXT (e.g. train_NC, test_NC, train_CE)
  --label-words a,b   words that mark a label/ground-truth file (default: seg,label,mask,gt,...)
  --mask-value N      use only label value N as EAT (default: every value > 0)
  --window LO,HI      intensity window for the PNGs (default: auto)
  --min-pixels N      skip slices with fewer than N EAT pixels (default 50)
  --flip-v            flip slices vertically (if the CT looks upside down)
"""
import argparse
import csv
import json
import os
import re

import cv2
import numpy as np

from config import BASE_DIR, IMAGE_DIR, MASK_DIR, ensure_dirs
from nrrd_io import get_spacing, read_nrrd

LABEL_WORDS = {"seg", "segmentation", "segmentations", "segm", "label", "labels", "mask", "masks",
               "gt", "groundtruth", "ground", "truth", "annotation", "annotations", "annot"}
IMAGE_WORDS = {"image", "images", "img", "imgs", "ct", "volume", "volumes", "raw", "original", "data", "ce", "nc"}


def tokens(s):
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def strip_ext(rel):
    return os.path.splitext(rel.replace("\\", "/"))[0]


def is_label(rel, words):
    return any(t in words for t in tokens(strip_ext(rel)))


def pair_key(rel, words):
    return tuple(t for t in tokens(strip_ext(rel)) if t not in words and t not in IMAGE_WORDS)


def choose_window(vol, arg):
    if arg and arg != "auto":
        lo, hi = (float(x) for x in arg.split(","))
        return lo, hi, "user"
    if float(vol.min()) < -500:                         # looks like Hounsfield units
        return -200.0, 300.0, "HU soft-tissue window"
    lo, hi = np.percentile(vol, [1, 99.5])
    return float(lo), float(hi), "percentile 1-99.5"


def to_uint8(slice2d, lo, hi):
    x = (slice2d.astype(np.float32) - lo) / max(hi - lo, 1e-6)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def find_files(src):
    out = []
    for root, _, files in os.walk(src):
        for f in files:
            if f.lower().endswith((".nrrd", ".nhdr")):
                out.append(os.path.relpath(os.path.join(root, f), src))
    return sorted(out)


def inspect(src, files, words, limit):
    labs = [f for f in files if is_label(f, words)]
    print(f"\nFound {len(files)} NRRD files ({len(files) - len(labs)} image-like, {len(labs)} label-like).")
    folders = {}
    for f in files:
        folders.setdefault(os.path.dirname(f), []).append(f)
    print("\nFolders (first 15):")
    for d, fs in list(folders.items())[:15]:
        print(f"  {d or '.'}  ->  {len(fs)} files, e.g. {os.path.basename(fs[0])}")
    print(f"\nDetails of the first {limit} files:")
    for f in files[:limit]:
        try:
            data, hdr = read_nrrd(os.path.join(src, f))
            sub = data[:, :, ::max(1, data.shape[2] // 8)] if data.ndim == 3 else data
            print(f"  {f}\n      shape={data.shape} dtype={data.dtype} min={data.min()} max={data.max()} "
                  f"distinct values (sampled)={len(np.unique(sub))} spacing={get_spacing(hdr)} "
                  f"-> {'LABEL' if is_label(f, words) else 'image'}")
        except Exception as e:
            print(f"  {f}\n      could not read: {e}")
    print("\nIf label files are wrongly detected, use  --label-words word1,word2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--filter", default=None)
    ap.add_argument("--patients", type=int, default=12)
    ap.add_argument("--slices", type=int, default=8)
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--mask-value", type=int, default=None)
    ap.add_argument("--label-words", default=None)
    ap.add_argument("--window", default="auto")
    ap.add_argument("--min-pixels", type=int, default=50)
    ap.add_argument("--flip-v", action="store_true")
    a = ap.parse_args()

    words = set(LABEL_WORDS)
    if a.label_words:
        words = {w.strip().lower() for w in a.label_words.split(",") if w.strip()}
    files = find_files(a.src)
    if not files:
        raise SystemExit("No .nrrd files found under --src. Check the folder path (extract the zip first).")
    if a.inspect:
        return inspect(a.src, files, words, a.limit)

    ensure_dirs()
    existing = [f for d in (IMAGE_DIR, MASK_DIR) for f in os.listdir(d) if f != ".gitkeep"]
    if existing and not a.clear:
        raise SystemExit(f"data/images or data/masks already has {len(existing)} files. "
                         "Re-run with --clear to replace them.")
    if a.clear:
        for d in (IMAGE_DIR, MASK_DIR):
            for f in os.listdir(d):
                if f != ".gitkeep":
                    os.remove(os.path.join(d, f))

    imgs, labs = {}, {}
    for f in files:
        (labs if is_label(f, words) else imgs).setdefault(pair_key(f, words), []).append(f)
    pairs = [(k, imgs[k][0], labs[k][0]) for k in sorted(imgs)
             if k in labs and len(imgs[k]) == 1 and len(labs[k]) == 1]
    if a.filter:
        pairs = [p for p in pairs if a.filter.lower() in p[1].lower().replace("\\", "/")]
    print(f"Image volumes: {sum(map(len, imgs.values()))}  Label volumes: {sum(map(len, labs.values()))}  "
          f"Matched pairs{' (filter: ' + a.filter + ')' if a.filter else ''}: {len(pairs)}")
    if not pairs:
        raise SystemExit("No image/label pairs matched. Run with --inspect and send me the output, "
                         "or try --label-words.")

    if len(pairs) > a.patients:
        idx = np.unique(np.linspace(0, len(pairs) - 1, a.patients).astype(int))
        pairs = [pairs[i] for i in idx]

    rows, info, win_msg, total = [], None, None, 0
    for pi, (_, ir, lr) in enumerate(pairs, 1):
        vol, hdr = read_nrrd(os.path.join(a.src, ir))
        lab, _ = read_nrrd(os.path.join(a.src, lr))
        if vol.shape != lab.shape or vol.ndim != 3:
            print(f"  skip {ir}: image shape {vol.shape} != label shape {lab.shape}")
            continue
        if info is None:
            sp = get_spacing(hdr)
            info = {"original_image_size": int(vol.shape[0])}
            if sp and len(sp) >= 3:
                info["pixel_spacing_mm"] = round(float(np.mean(sp[:2])), 4)
                info["slice_thickness_mm"] = round(float(sp[2]), 4)
        lo, hi, how = choose_window(vol, a.window)
        if win_msg is None:
            win_msg = f"intensity window {lo:.0f}..{hi:.0f} ({how}); volume range {vol.min()}..{vol.max()}"
        binary = (lab == a.mask_value) if a.mask_value is not None else (lab > 0)
        zs = [z for z in range(vol.shape[2]) if int(binary[:, :, z].sum()) >= a.min_pixels]
        if not zs:
            print(f"  skip {ir}: no slice with >= {a.min_pixels} EAT pixels")
            continue
        if len(zs) > a.slices:
            zs = [zs[i] for i in np.unique(np.linspace(0, len(zs) - 1, a.slices).astype(int))]
        for z in zs:
            img = to_uint8(vol[:, :, z].T, lo, hi)
            msk = binary[:, :, z].T.astype(np.uint8) * 255
            if a.flip_v:
                img, msk = np.flipud(img), np.flipud(msk)
            name = f"P{pi:02d}_{z:03d}"
            cv2.imwrite(os.path.join(IMAGE_DIR, name + ".png"), img)
            cv2.imwrite(os.path.join(MASK_DIR, name + ".png"), msk)
            rows.append([name, ir, lr, z])
        total += len(zs)
        print(f"  patient P{pi:02d}: {len(zs)} slices  ({ir})")

    if not rows:
        raise SystemExit("Nothing was written. Run with --inspect and send me the output.")
    with open(os.path.join(BASE_DIR, "data", "pairs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["new_name", "source_image", "source_label", "slice_index"])
        w.writerows(rows)
    with open(os.path.join(BASE_DIR, "data", "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"\n{win_msg}")
    print(f"Copied {total} slices from {len({r[1] for r in rows})} patients -> data/images and data/masks")
    print(f"dataset_info.json: {info}")
    if "pixel_spacing_mm" not in info:
        print("Note: no spacing found in the NRRD header; set PIXEL_SPACING_MM / SLICE_THICKNESS_MM in "
              "src/config.py yourself.")


if __name__ == "__main__":
    main()
