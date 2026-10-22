"""Loading, resizing, pairing and splitting the image/mask data."""
import glob
import os
import re

import cv2
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from config import (IMAGE_DIR, IMG_SIZE, MASK_DIR, PATIENT_REGEX, PATIENT_SPLIT, SEED)

EXTS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")


def list_files(folder):
    files = []
    for e in EXTS:
        files += glob.glob(os.path.join(folder, e))
    return sorted(files)


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _digits_key(stem):
    nums = re.findall(r"\d+", stem)
    return tuple(int(n) for n in nums) if nums else None


def pair_files(img_files, mask_files):
    """Pair images with masks: same file name first, else same numbers (img_001 <-> mask_001)."""
    by_stem = {_stem(p): p for p in mask_files}
    by_digits, dup = {}, set()
    for p in mask_files:
        k = _digits_key(_stem(p))
        if k is None:
            continue
        if k in by_digits:
            dup.add(k)
        by_digits[k] = p
    for k in dup:
        by_digits.pop(k, None)

    pairs = []
    for ip in img_files:
        s = _stem(ip)
        if s in by_stem:
            pairs.append((s, ip, by_stem[s]))
            continue
        k = _digits_key(s)
        if k is not None and k in by_digits:
            pairs.append((s, ip, by_digits[k]))
    return pairs


def preprocess_image(gray, size=IMG_SIZE):
    """gray: 2-D uint8 array -> (size, size, 1) float32 in [0, 1]."""
    img = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return img[..., np.newaxis]


def load_image(path):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return preprocess_image(gray)


def load_mask(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.float32)[..., np.newaxis]


def load_dataset():
    img_files, mask_files = list_files(IMAGE_DIR), list_files(MASK_DIR)
    pairs = pair_files(img_files, mask_files)
    if len(pairs) < len(img_files):
        print(f"Warning: {len(img_files) - len(pairs)} images had no matching mask and were skipped.")
    if not pairs:
        raise RuntimeError(
            "No image/mask pairs found. Put images in data/images and masks in data/masks, "
            "or run:  python src/prepare_dataset.py --src <downloaded_folder>")
    X = np.array([load_image(ip) for _, ip, _ in pairs], np.float32)
    Y = np.array([load_mask(mp) for _, _, mp in pairs], np.float32)
    return X, Y, [n for n, _, _ in pairs]


def _random_split(n):
    idx = np.arange(n)
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, random_state=SEED)
    return tr, va, te


def _group_ids(names):
    groups = []
    for n in names:
        m = re.search(PATIENT_REGEX, n)
        groups.append(m.group(1) if m else n)
    return np.array(groups)


def _patient_split(groups):
    idx = np.arange(len(groups))
    try:
        tr, tmp = next(GroupShuffleSplit(1, test_size=0.30, random_state=SEED).split(idx, groups=groups))
        va_r, te_r = next(GroupShuffleSplit(1, test_size=0.50, random_state=SEED)
                          .split(tmp, groups=groups[tmp]))
        print(f"Patient-wise split over {len(set(groups))} patients "
              f"(train {len(set(groups[tr]))} / val {len(set(groups[tmp[va_r]]))} / "
              f"test {len(set(groups[tmp[te_r]]))} patients).")
        return tr, tmp[va_r], tmp[te_r]
    except ValueError:
        print("Warning: not enough patients for a patient-wise split -> using random slice split.")
        return _random_split(len(groups))


def get_splits():
    """70 % train / 15 % validation / 15 % test (fixed seed -> every script sees the same split)."""
    X, Y, names = load_dataset()
    groups = _group_ids(names)
    n_groups = len(set(groups))
    use_patient = PATIENT_SPLIT is True or (PATIENT_SPLIT == "auto" and 5 <= n_groups < len(names))
    tr, va, te = _patient_split(groups) if use_patient else _random_split(len(X))
    if not use_patient:
        print("Random slice split (file names carry no patient ids).")

    def pick(ix):
        return X[ix], Y[ix], [names[i] for i in ix]

    return {"train": pick(tr), "val": pick(va), "test": pick(te)}
