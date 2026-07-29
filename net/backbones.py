"""Frozen pretrained feature extractors used by models 2, 4, 5, and 6."""

from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3


def load_inception_backbone(weights_path, input_shape=(256, 256, 3)):
    """InceptionV3 with local no-top weights, all layers frozen."""
    backbone = InceptionV3(input_shape=input_shape, include_top=False, weights=None)
    backbone.load_weights(weights_path)
    for layer in backbone.layers:
        layer.trainable = False
    return backbone


def load_vgg16_backbone(input_shape=(256, 256, 3)):
    """VGG16 with ImageNet weights, all layers frozen."""
    backbone = VGG16(input_shape=input_shape, weights='imagenet', include_top=False)
    for layer in backbone.layers:
        layer.trainable = False
    return backbone


def load_resnet50_backbone(input_shape=(256, 256, 3)):
    """ResNet50 with ImageNet weights, all layers frozen."""
    backbone = ResNet50(input_shape=input_shape, weights='imagenet', include_top=False)
    for layer in backbone.layers:
        layer.trainable = False
    return backbone
