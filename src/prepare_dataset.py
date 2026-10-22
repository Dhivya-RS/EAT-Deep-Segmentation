"""
Takes a downloaded dataset folder (e.g. from Kaggle), finds the CT images and their EAT masks,
keeps N pairs that really contain EAT, and copies them into data/images and data/masks with
matching names.

  python src/prepare_dataset.py --src "C:/Users/me/Downloads/kaggle_eat" --n 100

Useful options
  --clear               delete whatever is already in data/images and data/masks first
  --mask-value 1        use only this label value as EAT (for multi-label masks)
  --mask-words a,b      words that identify mask files/folders (default: mask,label,seg,gt,annot)
  --patient-regex RE    regex (group 1 = patient id) applied to the ORIGINAL relative path;
                        output files become  <patient>_0001.png  (then set PATIENT_SPLIT = True)
Supports png / jpg / tif / bmp. DICOM (.dcm) and NIfTI (.nii) must be converted to png first.
"""
import argparse
import csv
import os
import re

import cv2
import numpy as np

from config import BASE_DIR, IMAGE_DIR, MASK_DIR, ensure_dirs

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
OTHER_EXTS = {".dcm", ".nii", ".mhd", ".npy"}


def is_mask(rel, words):
    parts = rel.replace("\\", "/").lower().split("/")
    check = [parts[-1], parts[-2] if len(parts) > 1 else ""]   # file name + parent folder only
    return any(w in c for c in check for w in words)


def make_key(rel, words):
    stem = os.path.splitext(rel.replace("\\", "/"))[0]
    nums = re.findall(r"\d+", stem)
    if nums:
        return tuple(int(n) for n in nums)
    base = os.path.basename(stem).lower()
    return re.sub(r"[_\-\s]*(" + "|".join(words) + r")s?", "", base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--mask-value", type=int, default=None)
    ap.add_argument("--mask-words", default="mask,label,seg,gt,annot")
    ap.add_argument("--min-pixels", type=int, default=20, help="skip masks with fewer EAT pixels")
    ap.add_argument("--patient-regex", default=None)
    a = ap.parse_args()
    words = [w.strip().lower() for w in a.mask_words.split(",") if w.strip()]

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

    imgs, masks, other = {}, {}, 0
    for root, _, files in os.walk(a.src):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            rel = os.path.relpath(os.path.join(root, f), a.src)
            if ext in OTHER_EXTS or f.lower().endswith(".nii.gz"):
                other += 1
            elif ext in IMG_EXTS:
                (masks if is_mask(rel, words) else imgs).setdefault(make_key(rel, words), []).append(rel)
    print(f"Found {sum(map(len, imgs.values()))} image files and {sum(map(len, masks.values()))} mask files.")
    if other:
        print(f"Note: {other} DICOM/NIfTI-type files were ignored (convert them to png first).")

    pairs = [(k, imgs[k][0], masks[k][0]) for k in sorted(imgs, key=str)
             if k in masks and len(imgs[k]) == 1 and len(masks[k]) == 1]
    print(f"Matched pairs: {len(pairs)}")
    if not pairs:
        raise SystemExit("No pairs matched. Check --mask-words, or send me the folder/file names.")

    valid = []
    for k, ir, mr in pairs:
        m = cv2.imread(os.path.join(a.src, mr), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        b = (m == a.mask_value) if a.mask_value is not None else (m > 0)
        if int(b.sum()) >= a.min_pixels:
            valid.append((ir, mr, b))
    print(f"Pairs whose mask contains EAT: {len(valid)}")
    if not valid:
        raise SystemExit("All masks are empty. Try --mask-value or check the mask files.")

    if len(valid) > a.n:                                   # spread the choice over the whole dataset
        idx = np.unique(np.linspace(0, len(valid) - 1, a.n).astype(int))
        valid = [valid[i] for i in idx]

    rows = []
    for i, (ir, mr, b) in enumerate(valid, 1):
        prefix = "case"
        if a.patient_regex:
            mt = re.search(a.patient_regex, ir)
            prefix = re.sub(r"\W+", "", mt.group(1)) if mt else "case"
        name = f"{prefix}_{i:04d}"
        gray = cv2.imread(os.path.join(a.src, ir), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(IMAGE_DIR, name + ".png"), gray)
        cv2.imwrite(os.path.join(MASK_DIR, name + ".png"), b.astype(np.uint8) * 255)
        rows.append([name, ir, mr])

    with open(os.path.join(BASE_DIR, "data", "pairs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["new_name", "original_image", "original_mask"])
        w.writerows(rows)
    print(f"\nCopied {len(rows)} image/mask pairs -> data/images and data/masks  (list: data/pairs.csv)")
    if a.patient_regex:
        print("Tip: set PATIENT_SPLIT = True in src/config.py for a patient-wise split.")


if __name__ == "__main__":
    main()
