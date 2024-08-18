import tensorflow as tf
import random
from typing import Tuple

def encode_image_and_label(image_file_path: str, label: int, target_resolution: Tuple[int, int] = (256, 256), num_labels: int = 102) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    load jpg image resizing or cropping it to a target resolution and one hot encode a label

    Args:
        image_file_path (str): file path name of the jpeg image to load
        label (str): label for the image
        target_resolution (Tuple[int, int]): a tuple containing the target resolution for images (width, height)
        num_labels (int): total number of labels, needed for one hot encoding the label. Defaults to 102.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: image data, label one hot encoded
    """
    # load image
    image = tf.io.read_file(image_file_path)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, dtype=tf.float32)

    # rescale or randomly crop to target size
    rescaling_factor = random.uniform(1, 1.25)
    image = tf.image.resize(image, size=(int(target_resolution[0]*rescaling_factor), int(target_resolution[1]*rescaling_factor)))
    image = tf.image.random_crop(image, size=(target_resolution[0], target_resolution[1], 3))                            

    # one hot encode label
    label = tf.one_hot(label, depth=num_labels, dtype=tf.float32)

    return image, label

def encode_image(image_file_path: str, target_resolution: Tuple[int, int] = (256, 256)) -> tf.Tensor:
    """
    load jpg image resizing or cropping it to a target resolution and one hot encode a label

    Args:
        image_file_path (str): file path name of the jpeg image to load
        target_resolution (Tuple[int, int]): a tuple containing the target resolution for images (width, height)

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: image data, label one hot encoded
    """
    # load image
    image = tf.io.read_file(image_file_path)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, dtype=tf.float32)

    # rescale or randomly crop to target size
    rescaling_factor = random.uniform(1, 1.25)
    image = tf.image.resize(image, size=(int(target_resolution[0]*rescaling_factor), int(target_resolution[1]*rescaling_factor)))
    image = tf.image.random_crop(image, size=(target_resolution[0], target_resolution[1], 3))                            

    return image

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
