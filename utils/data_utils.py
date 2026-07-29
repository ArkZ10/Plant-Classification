"""Train/val splitting and Keras ImageDataGenerator setup."""

import os
import shutil

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def split_dataset(data_dir, classes, train_dir, val_dir, val_split=0.2, seed=42):
    """Copy each class's images into `train_dir`/`val_dir` subfolders."""
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        images = os.listdir(class_dir)
        train_images, val_images = train_test_split(
            images, test_size=val_split, random_state=seed
        )

        for image in train_images:
            shutil.copy(os.path.join(class_dir, image), os.path.join(train_class_dir, image))
        for image in val_images:
            shutil.copy(os.path.join(class_dir, image), os.path.join(val_class_dir, image))

    print(f"Dataset split into {train_dir} and {val_dir}")


def count_files(directory):
    """Total number of files under `directory`, recursively."""
    return sum(len(files) for _, _, files in os.walk(directory))


def build_generators(train_dir, val_dir, img_size=256, batch_size=20):
    """Augmented training generator + rescale-only validation generator."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
    )
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir, batch_size=batch_size,
        class_mode='categorical', target_size=(img_size, img_size),
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    validation_generator = val_datagen.flow_from_directory(
        directory=val_dir, batch_size=batch_size,
        class_mode='categorical', target_size=(img_size, img_size),
    )

    return train_generator, validation_generator
