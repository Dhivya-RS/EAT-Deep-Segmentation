"""Helpers to draw overlays and comparison figures."""
import matplotlib.pyplot as plt
import numpy as np


def overlay(img, mask, color=(1.0, 0.0, 0.0), alpha=0.5):
    """img: HxW float [0,1]; mask: HxW {0,1}  ->  HxWx3 RGB image."""
    rgb = np.stack([img, img, img], axis=-1).astype(np.float32)
    m = mask > 0
    for c in range(3):
        rgb[..., c][m] = (1 - alpha) * rgb[..., c][m] + alpha * color[c]
    return np.clip(rgb, 0, 1)


def make_figure(img, pred, gt=None, title=None):
    """Original | (ground truth) | predicted mask | overlay."""
    panels = [("Original CT", img, "gray")]
    if gt is not None:
        panels.append(("Ground truth EAT", gt, "gray"))
    panels.append(("Predicted EAT mask", pred, "gray"))
    panels.append(("Overlay (red = EAT)", overlay(img, pred), None))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, (t, data, cmap) in zip(np.atleast_1d(axes), panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig
