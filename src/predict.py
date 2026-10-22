"""
Demo script: segment EAT on a CT image (or a folder of slices) and print the quantification.

  python src/predict.py --image data/images/sample_0001.png
  python src/predict.py --folder data/images            (treated as one patient's slice stack)
Add --no-show to only save the output without opening a window.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

from config import MODEL_PATH, PRED_DIR, THRESHOLD, ensure_dirs
from data_loader import list_files, load_image, load_mask, MASK_DIR
from quantify import eat_area_cm2, eat_volume_cm3
from visualize import make_figure


def predict_mask(model, img):
    prob = model.predict(img[np.newaxis], verbose=0)[0, ..., 0]
    return (prob >= THRESHOLD).astype(np.uint8)


def find_gt(image_path):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    for f in list_files(MASK_DIR):
        if os.path.splitext(os.path.basename(f))[0] == stem:
            return load_mask(f)[..., 0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--folder")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    if not args.image and not args.folder:
        ap.error("give --image <file> or --folder <folder>")

    ensure_dirs()
    if not os.path.exists(MODEL_PATH):
        raise SystemExit("Trained model not found. Run:  python src/train.py")
    model = load_model(MODEL_PATH, compile=False)

    if args.image:
        img = load_image(args.image)
        mask = predict_mask(model, img)
        stem = os.path.splitext(os.path.basename(args.image))[0]
        out = os.path.join(PRED_DIR, f"{stem}_demo.png")
        area = eat_area_cm2(mask)
        fig = make_figure(img[..., 0], mask, find_gt(args.image),
                          title=f"{stem}  |  EAT area = {area:.2f} cm2")
        fig.savefig(out, dpi=130)
        print(f"\nImage            : {args.image}")
        print(f"EAT pixels       : {int(mask.sum())}")
        print(f"EAT area         : {area:.2f} cm2")
        print(f"Output saved to  : {out}")
        if not args.no_show:
            plt.show()
    else:
        files = list_files(args.folder)
        masks = []
        for f in files:
            img = load_image(f)
            m = predict_mask(model, img)
            masks.append(m)
            stem = os.path.splitext(os.path.basename(f))[0]
            fig = make_figure(img[..., 0], m, title=stem)
            fig.savefig(os.path.join(PRED_DIR, f"{stem}_demo.png"), dpi=100)
            plt.close(fig)
        vol = eat_volume_cm3(masks)
        print(f"\nSlices processed : {len(files)}")
        print(f"Total EAT volume : {vol:.2f} cm3")
        print(f"Outputs saved to : {PRED_DIR}")


if __name__ == "__main__":
    main()
