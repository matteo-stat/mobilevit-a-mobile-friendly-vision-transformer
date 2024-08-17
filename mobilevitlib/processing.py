import tensorflow as tf
from typing import Tuple

def data_augmentation(images_batch: tf.Tensor, labels_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    apply random transformations to augment images data

    Args:
        images_batch (tf.Tensor): batch of images data
        labels_batch (tf.Tensor): batch of labels

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: batch of images data, batch of labels
    """
    # random horizontal flip
    images_batch = tf.image.random_flip_left_right(images_batch)

    # random vertical flip
    images_batch = tf.image.random_flip_up_down(images_batch)

    # small hue change
    images_batch = tf.image.random_hue(images_batch, max_delta=0.05)

    # small saturation change
    images_batch = tf.image.random_saturation(images_batch, lower=0.95, upper=1.05) 

    # small contrast change
    images_batch = tf.image.random_contrast(images_batch, lower=0.90, upper=1.10)

    # small brightness change
    images_batch = tf.image.random_brightness(images_batch, max_delta=0.10)

    # clip values out of range
    images_batch = tf.clip_by_value(images_batch, clip_value_min=0.0, clip_value_max=255.0)

    return images_batch, labels_batch
