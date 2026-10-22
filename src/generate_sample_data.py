"""
Creates SYNTHETIC cardiac-CT-like slices + EAT masks so the whole pipeline
can be demonstrated without a real dataset.

NOTE: This is demo data only. For real results, copy your real cardiac CT
slices into data/images and the matching EAT masks into data/masks
(same file names) and DO NOT run this script.

Usage:  python src/generate_sample_data.py [--n 200] [--force]
"""
import argparse
import glob
import os

import cv2
import numpy as np

from config import IMAGE_DIR, MASK_DIR, ensure_dirs

SIZE = 512


def make_slice(rng):
    img = np.full((SIZE, SIZE), 12, np.uint8)
    cx, cy = SIZE // 2 + rng.integers(-15, 16), SIZE // 2 + rng.integers(-10, 11)

    # body
    cv2.ellipse(img, (cx, cy), (rng.integers(205, 230), rng.integers(150, 175)), 0, 0, 360, 105, -1)
    # lungs
    cv2.ellipse(img, (cx - 120, cy - 10), (70, 105), 0, 0, 360, 20, -1)
    cv2.ellipse(img, (cx + 120, cy - 10), (70, 105), 0, 0, 360, 20, -1)

    # heart + epicardial fat ring
    hx, hy = cx + rng.integers(-12, 13), cy + rng.integers(0, 20)
    ha, hb = int(rng.integers(55, 75)), int(rng.integers(50, 68))
    t = int(rng.integers(8, 18))                    # fat thickness
    ang = int(rng.integers(0, 360))

    fat = np.zeros((SIZE, SIZE), np.uint8)
    cv2.ellipse(fat, (hx, hy), (ha + t, hb + t), 0, 0, 360, 255, -1)
    cv2.ellipse(fat, (hx, hy), (ha, hb), 0, 0, 360, 0, -1)
    wedge = np.zeros((SIZE, SIZE), np.uint8)        # fat only on part of the ring
    cv2.ellipse(wedge, (hx, hy), (300, 300), 0, ang, ang + int(rng.integers(180, 320)), 255, -1)
    fat = cv2.bitwise_and(fat, wedge)

    cv2.ellipse(img, (hx, hy), (ha, hb), 0, 0, 360, 155, -1)   # heart muscle/blood
    img[fat > 0] = 78                                           # fat = darker

    img = cv2.GaussianBlur(img, (5, 5), 0).astype(np.float32)
    img += rng.normal(0, 6, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, fat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    existing = [f for f in glob.glob(os.path.join(IMAGE_DIR, "*")) if not f.endswith(".gitkeep")]
    if existing and not args.force:
        print(f"data/images already has {len(existing)} files -> skipping sample generation "
              "(use --force to overwrite).")
        return

    rng = np.random.default_rng(42)
    for i in range(1, args.n + 1):
        img, mask = make_slice(rng)
        name = f"sample_{i:04d}.png"
        cv2.imwrite(os.path.join(IMAGE_DIR, name), img)
        cv2.imwrite(os.path.join(MASK_DIR, name), mask)
    print(f"Created {args.n} synthetic image/mask pairs in data/images and data/masks")


if __name__ == "__main__":
    main()
