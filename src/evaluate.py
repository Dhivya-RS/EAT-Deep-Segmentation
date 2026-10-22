"""Evaluate the trained model on the test split and save results + sample figures.
Usage:  python src/evaluate.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os

from config import BATCH_SIZE, MODEL_PATH, PRED_DIR, RESULTS_CSV, THRESHOLD, ensure_dirs
from data_loader import get_splits
from metrics_np import dice_np, iou_np, precision_np, recall_np
from quantify import eat_area_cm2
from visualize import make_figure


def main(n_samples=8):
    ensure_dirs()
    Xte, Yte, names = get_splits()["test"]
    model = load_model(MODEL_PATH, compile=False)

    prob = model.predict(Xte, batch_size=BATCH_SIZE, verbose=0)
    pred = (prob >= THRESHOLD).astype(np.float32)

    rows = []
    for i, name in enumerate(names):
        gt, pr = Yte[i, ..., 0], pred[i, ..., 0]
        a_gt, a_pr = eat_area_cm2(gt), eat_area_cm2(pr)
        rows.append({
            "image": name,
            "dice": dice_np(gt, pr),
            "iou": iou_np(gt, pr),
            "precision": precision_np(gt, pr),
            "recall": recall_np(gt, pr),
            "gt_eat_area_cm2": a_gt,
            "pred_eat_area_cm2": a_pr,
            "area_error_%": abs(a_pr - a_gt) / (a_gt + 1e-7) * 100,
        })

    df = pd.DataFrame(rows)
    mean_row = {"image": "MEAN"}
    mean_row.update(df.drop(columns="image").mean().to_dict())
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    df.round(4).to_csv(RESULTS_CSV, index=False)

    print("\n===== TEST RESULTS (mean over %d images) =====" % len(names))
    for k in ("dice", "iou", "precision", "recall", "area_error_%"):
        print(f"{k:>14}: {mean_row[k]:.4f}")
    print("Saved:", RESULTS_CSV)

    for i in range(min(n_samples, len(names))):
        fig = make_figure(Xte[i, ..., 0], pred[i, ..., 0], Yte[i, ..., 0],
                          title=f"{names[i]}  |  Dice = {rows[i]['dice']:.3f}")
        fig.savefig(os.path.join(PRED_DIR, f"{names[i]}_result.png"), dpi=130)
        plt.close(fig)
    print("Sample prediction images ->", PRED_DIR)


if __name__ == "__main__":
    main()
