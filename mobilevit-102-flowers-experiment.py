import scipy
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
import mobilevitlib as mvitl
import tensorflow as tf
import keras

# some parameters
NUM_CLASSES = 102
SEED = 1993
BATCH_SIZE = 64

# load labels and subtract 1, so that they are indexes starting from 0
# this is a lot more comfortable if we are gonne one-hot encode them later on
labels = scipy.io.loadmat('data/imagelabels.mat')
labels = labels['labels'][0]
labels = labels - 1

# this is an imbalanced dataset, where some labels have a lot less samples than others
labels_frequency = np.unique(labels, return_counts=True)
print(f"label {labels_frequency[0][labels_frequency[1].argmin()]} has {labels_frequency[1].min()} samples")
print(f"label {labels_frequency[0][labels_frequency[1].argmax()]} has {labels_frequency[1].max()} samples")

# calculate some labels weights for balancing the loss during training
labels_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
labels_weights = {
    int(label): label_weight
    for label, label_weight in np.column_stack((np.unique(labels), labels_weights))
}

# load images paths filenames
image_file_paths = np.array(sorted(glob.iglob('data/jpg/*.jpg')))

# split data into training, validation and test sets
# stratified random sampling can be used to keep the same level of classes imbalance between the sets
image_file_paths_train, image_file_paths_test, labels_train, labels_test = train_test_split(
    image_file_paths,
    labels,
    test_size=0.125,
    random_state=SEED, 
    shuffle=True,
    stratify=labels
)
image_file_paths_train, image_file_paths_val, labels_train, labels_val = train_test_split(
    image_file_paths_train,
    labels_train,
    test_size=0.125,
    random_state=SEED, 
    shuffle=True,
    stratify=labels_train
)

# data pipelines
ds_train = (
    tf.data.Dataset.from_tensor_slices((image_file_paths_train, labels_train))
    .shuffle(buffer_size=len(labels_train))
    .map(mvitl.read.read_and_resize_jpeg_image_encode_label, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size=BATCH_SIZE)
    .map(mvitl.processing.data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
ds_val = (
    tf.data.Dataset.from_tensor_slices((image_file_paths_val, labels_val))
    .shuffle(buffer_size=len(labels_val))
    .map(mvitl.read.read_and_resize_jpeg_image_encode_label, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size=BATCH_SIZE)
    .map(mvitl.processing.data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
ds_test = (
    tf.data.Dataset.from_tensor_slices((image_file_paths_test, labels_test))
    .shuffle(buffer_size=len(labels_test))
    .map(mvitl.read.read_and_resize_jpeg_image_encode_label, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size=BATCH_SIZE)
    .map(mvitl.processing.data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# build mobilevit model
model = mvitl.models.build_mobilevit(network_size='xxs', input_shape=(256, 256, 3), num_classes=NUM_CLASSES)
model.summary()

# compile model
model.compile(
    optimizer=keras.optimizers.AdamW(),
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.CategoricalAccuracy()]
)

# fit the model using validation set to assess model performances on test data
history = model.fit(
    ds_train,
    epochs=50,
    validation_data=ds_val,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.01,
            patience=5,
            verbose=1,
            restore_best_weights=True,
            start_from_epoch=5,
        )
    ],
    class_weight=labels_weights
)

s = 0