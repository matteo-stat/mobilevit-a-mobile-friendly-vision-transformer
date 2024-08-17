import scipy
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight

NUM_CLASSES = 102
SEED = 1993

# load official data splits, wich are unusual
# around 1k samples for train and val sets, around 6k samples for test set!
data_splits = scipy.io.loadmat('data/setid.mat')
data_splits = {
    split_name.replace('trnid', 'train').replace('valid', 'val').replace('tstid', 'test'): indexes[0]
    for split_name, indexes in data_splits.items()
    if split_name in ('trnid', 'valid', 'tstid')
}

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
image_file_paths = np.array(glob.glob('data/jpg/*.jpg'))

train_test_split(image_file_paths, labels, test_size=0.125, random_state=SEED, shuffle=True, stratify=labels)


s = 0