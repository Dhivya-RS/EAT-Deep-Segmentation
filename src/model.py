"""U-Net model + Dice/IoU metrics and the combined BCE-Dice loss."""
import tensorflow as tf
from tensorflow.keras import layers, models

from config import IMG_SIZE


def conv_block(x, filters):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    return x


def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 1), base=16):
    inputs = layers.Input(input_shape)

    # encoder
    c1 = conv_block(inputs, base)
    p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, base * 2)
    p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, base * 4)
    p3 = layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, base * 8)
    p4 = layers.MaxPooling2D()(c4)

    # bottleneck
    b = conv_block(p4, base * 16)
    b = layers.Dropout(0.3)(b)

    # decoder
    u4 = layers.Conv2DTranspose(base * 8, 2, strides=2, padding="same")(b)
    c5 = conv_block(layers.concatenate([u4, c4]), base * 8)
    u3 = layers.Conv2DTranspose(base * 4, 2, strides=2, padding="same")(c5)
    c6 = conv_block(layers.concatenate([u3, c3]), base * 4)
    u2 = layers.Conv2DTranspose(base * 2, 2, strides=2, padding="same")(c6)
    c7 = conv_block(layers.concatenate([u2, c2]), base * 2)
    u1 = layers.Conv2DTranspose(base, 2, strides=2, padding="same")(c7)
    c8 = conv_block(layers.concatenate([u1, c1]), base)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c8)
    return models.Model(inputs, outputs, name="EAT_UNet")


# ---------- metrics / loss ----------
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [tf.shape(y_true)[0], -1])
    y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [tf.shape(y_pred)[0], -1])
    inter = tf.reduce_sum(y_true * y_pred, axis=1)
    union = tf.reduce_sum(y_true, axis=1) + tf.reduce_sum(y_pred, axis=1)
    return tf.reduce_mean((2.0 * inter + smooth) / (union + smooth))


def iou_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [tf.shape(y_true)[0], -1])
    y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [tf.shape(y_pred)[0], -1])
    inter = tf.reduce_sum(y_true * y_pred, axis=1)
    union = tf.reduce_sum(y_true, axis=1) + tf.reduce_sum(y_pred, axis=1) - inter
    return tf.reduce_mean((inter + smooth) / (union + smooth))


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + dice_loss(y_true, y_pred)


if __name__ == "__main__":
    build_unet().summary()
