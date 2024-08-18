# dependecies
import scipy
import glob
import numpy as np
import mobilevitlib as mvitl
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from sklearn.metrics import classification_report

# set gpu memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=12000)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
  except RuntimeError as e:
    print(e)

# some parameters
NUM_CLASSES = 102
SEED = 1993
BATCH_SIZE = 64

# load labels and subtract 1, so that they are indexes starting from 0
# this is a lot more comfortable for one-hot encode them later on
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
# use stratified random sampling to maintain the same level of class imbalance across all sets
image_file_paths_train, image_file_paths_test, labels_train, labels_test = train_test_split(
    image_file_paths,
    labels,
    test_size=1000,
    random_state=SEED, 
    shuffle=True,
    stratify=labels
)
image_file_paths_train, image_file_paths_val, labels_train, labels_val = train_test_split(
    image_file_paths_train,
    labels_train,
    test_size=689,
    random_state=SEED, 
    shuffle=True,
    stratify=labels_train
)

# data pipelines
ds_train = (
    tf.data.Dataset.from_tensor_slices((image_file_paths_train, labels_train))
    .shuffle(buffer_size=len(labels_train))
    .map(mvitl.processing.encode_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size=BATCH_SIZE)
    .map(mvitl.processing.data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
ds_val = (
    tf.data.Dataset.from_tensor_slices((image_file_paths_val, labels_val))
    .shuffle(buffer_size=len(labels_val))
    .map(mvitl.processing.encode_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size=BATCH_SIZE)
    .map(mvitl.processing.data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
ds_test = (
    tf.data.Dataset.from_tensor_slices(image_file_paths_test)
    .map(mvitl.processing.encode_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size=BATCH_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# build mobilevit model
model = mvitl.models.build_mobilevit(network_size='xs', input_shape=(256, 256, 3), num_classes=NUM_CLASSES)
model.summary()

# compile model
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=2e-4),
    loss=keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1),
    metrics=[keras.metrics.CategoricalAccuracy()]
)

# fit the model using validation set to assess model performances on test data
history = model.fit(
    ds_train,
    epochs=160,
    validation_data=ds_val,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.01,
            patience=10,
            verbose=1,
            restore_best_weights=True,
            start_from_epoch=10,
        ),
    ],
    class_weight=labels_weights
)

# model wit dropouts layers
mvitl.plots.plot_training_history(history.history, figsize=(10, 6), show=True)

# predictions on test data
predictions = []
for batch_test in ds_test:
    predictions.append(model.predict_on_batch(batch_test))
predictions = tf.concat(predictions, axis=0)
predictions = tf.math.argmax(predictions, axis=-1)
predictions = tf.cast(predictions, dtype=tf.uint8)

# whole accuracy on test data
accuracy = tf.cast(labels_test == predictions, dtype=tf.uint8)
accuracy = accuracy.numpy().sum() / len(accuracy)
print(f'test_categorical_accuracy: {accuracy:.4f}')

# classification report for each class
print(classification_report(labels_test, predictions.numpy()))
