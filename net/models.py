"""Six candidate architectures compared for flower classification."""

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, Dense, Dropout, Flatten,
    GlobalMaxPooling2D, LeakyReLU, MaxPooling2D,
)
from tensorflow.keras.regularizers import l2


def build_model1(input_shape=(256, 256, 3), num_classes=10):
    """Model 1: plain Conv2D + MaxPooling CNN."""
    return tf.keras.models.Sequential([
        Conv2D(128, (5, 5), padding='valid', activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        BatchNormalization(),

        Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_regularizer=l2(0.00005)),
        MaxPooling2D(2, 2),
        BatchNormalization(),

        Conv2D(32, (3, 3), padding='valid', activation='relu', kernel_regularizer=l2(0.00005)),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(32, (3, 3), padding='valid', activation='relu', kernel_regularizer=l2(0.00005)),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Flatten(),
        Dense(units=256, activation='relu'),
        Dropout(0.5),
        Dense(units=num_classes, activation='softmax'),
    ])


def build_model2(backbone, num_classes=10):
    """Model 2: InceptionV3 backbone + Conv2D head."""
    model = tf.keras.models.Sequential()
    model.add(backbone)
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D(3, 3))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D(1, 1))
    model.add(Dense(64, activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    return model


def build_model3(input_shape=(256, 256, 3), num_classes=10):
    """Model 3: Conv2D CNN using LeakyReLU activations."""
    model = tf.keras.models.Sequential()
    model.add(Conv2D(128, (3, 3), input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.02))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3)))
    model.add(LeakyReLU(alpha=0.02))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(GlobalMaxPooling2D())
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.02))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def build_model4(backbone, num_classes=10):
    """Model 4: InceptionV3 backbone + LeakyReLU head."""
    model = tf.keras.models.Sequential()
    model.add(backbone)

    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (2, 2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.25))

    model.add(GlobalMaxPooling2D())
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.02))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def build_model5(backbone, num_classes=10):
    """Model 5: VGG16 backbone + LeakyReLU head."""
    model = tf.keras.models.Sequential()
    model.add(backbone)

    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (2, 2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.25))

    model.add(GlobalMaxPooling2D())
    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.02))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def build_model6(backbone, num_classes=10):
    """Model 6: ResNet50 backbone + LeakyReLU head."""
    model = tf.keras.models.Sequential()
    model.add(backbone)

    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (2, 2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.25))

    model.add(GlobalMaxPooling2D())
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.02))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model
