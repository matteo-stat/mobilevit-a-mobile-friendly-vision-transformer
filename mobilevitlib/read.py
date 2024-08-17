import tensorflow as tf
from typing import Tuple
import random

def read_and_resize_jpeg_image_encode_label(image_file_path: str, label: int, target_resolution: Tuple[int, int] = (256, 256), num_labels: int = 102) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    load png image and label

    Args:
        image_file_path (str): file path name of the jpeg image to load
        label (str): label for the image
        target_resolution (Tuple[int, int]): a tuple containing the target resolution for images (width, height)
        num_labels (int): total number of labels, needed for one hot encoding the label. Defaults to 102.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: image data, label one hot encoded
    """    
    image = tf.io.read_file(image_file_path)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.resize(image, size=target_resolution)

    label = tf.one_hot(label, depth=num_labels, dtype=tf.float32)

    return image, label
