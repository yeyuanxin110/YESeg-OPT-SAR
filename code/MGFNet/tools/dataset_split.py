import os
import random

dataset_folder = "MMSeg-YREB/train/label"
train_file = 'MMSeg-YREB/train/train.txt'
val_file = 'MMSeg-YREB/train/val.txt'

train_ratio = 0.7
val_ratio = 0.3

image_files = [f for f in os.listdir(dataset_folder) if f.endswith('.tif')]

random.shuffle(image_files)

total_images = len(image_files)
num_train = int(total_images * train_ratio)
num_val = total_images - num_train

train_images = image_files[:num_train]
val_images = image_files[num_train:]

with open(train_file, 'w') as f:
    for image in train_images:
        f.write(image + '\n')

with open(val_file, 'w') as f:
    for image in val_images:
        f.write(image + '\n')

