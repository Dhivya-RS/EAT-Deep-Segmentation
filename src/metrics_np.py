"""Plain NumPy metrics used for the final evaluation table."""
import numpy as np

EPS = 1e-7


def dice_np(gt, pr):
    inter = float((gt * pr).sum())
    return (2 * inter + EPS) / (float(gt.sum()) + float(pr.sum()) + EPS)


def iou_np(gt, pr):
    inter = float((gt * pr).sum())
    union = float(gt.sum()) + float(pr.sum()) - inter
    return (inter + EPS) / (union + EPS)


def precision_np(gt, pr):
    tp = float((gt * pr).sum())
    return (tp + EPS) / (float(pr.sum()) + EPS)


def recall_np(gt, pr):
    tp = float((gt * pr).sum())
    return (tp + EPS) / (float(gt.sum()) + EPS)
