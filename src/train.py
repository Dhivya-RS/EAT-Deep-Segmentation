"""Train the U-Net.   Usage:  python src/train.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.optimizers import Adam

from config import (BATCH_SIZE, EPOCHS, GRAPH_DIR, LEARNING_RATE, MODEL_PATH, SEED,
                    TRAIN_LOG_CSV, ensure_dirs)
from data_loader import get_splits
from model import bce_dice_loss, build_unet, dice_coef, iou_coef
import os

np.random.seed(SEED)
tf.random.set_seed(SEED)


def augment(img, mask):
    """Light intensity augmentation (no flips - keeps anatomy orientation realistic)."""
    img = tf.image.random_brightness(img, 0.08)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return tf.clip_by_value(img, 0.0, 1.0), mask


def make_ds(X, Y, training):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if training:
        ds = ds.shuffle(len(X), seed=SEED).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def save_curve(history, key, title, fname):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history[key], label="train")
    plt.plot(history.history["val_" + key], label="validation")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(key)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, fname), dpi=150)
    plt.close()


def main():
    ensure_dirs()
    splits = get_splits()
    Xtr, Ytr, _ = splits["train"]
    Xva, Yva, _ = splits["val"]
    print(f"Train: {len(Xtr)}  Validation: {len(Xva)}  Test: {len(splits['test'][0])}")

    model = build_unet()
    model.compile(optimizer=Adam(LEARNING_RATE), loss=bce_dice_loss,
                  metrics=[dice_coef, iou_coef])

    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor="val_dice_coef", mode="max",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
        EarlyStopping(monitor="val_dice_coef", mode="max", patience=10,
                      restore_best_weights=True),
        CSVLogger(TRAIN_LOG_CSV),
    ]

    history = model.fit(make_ds(Xtr, Ytr, True), validation_data=make_ds(Xva, Yva, False),
                        epochs=EPOCHS, callbacks=callbacks)

    save_curve(history, "loss", "Training vs Validation Loss", "loss_curve.png")
    save_curve(history, "dice_coef", "Dice Coefficient", "dice_curve.png")
    save_curve(history, "iou_coef", "IoU", "iou_curve.png")
    print("\nTraining finished.")
    print("Best model  ->", MODEL_PATH)
    print("Graphs      ->", GRAPH_DIR)


if __name__ == "__main__":
    main()
