import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image

np.random.seed(347)

data_folder = r"E:\wk\MGFNet\datasets\DATA\label"
save_folder = r"E:\wk\MGFNet\datasets\DATA"

train_ratio = 0.8
val_ratio = 0.2

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

all_images = []
all_labels = []

for label_file in os.listdir(data_folder):
    label_path = os.path.join(data_folder, label_file)
    if label_file.endswith(".png"):
        with Image.open(label_path) as img:
            label_data = np.array(img)

        unique_values = np.unique(label_data)

        if 0 in unique_values:
            unique_values = unique_values[1:]

        if len(unique_values) > 0:
            label_value = unique_values[0]
            all_images.append(label_file)
            all_labels.append(label_value)

X = np.array(all_images)
y = np.array(all_labels)

sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, test_size=val_ratio, random_state=347)

for train_index, val_index in sss.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

train_file_path = os.path.join(save_folder, 'train.txt')
val_file_path = os.path.join(save_folder, 'val.txt')

with open(train_file_path, 'w') as train_file:
    train_file.write("\n".join(X_train))

with open(val_file_path, 'w') as val_file:
    val_file.write("\n".join(X_val))
