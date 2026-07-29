import argparse

parser = argparse.ArgumentParser()

# Data
parser.add_argument('--data_dir', type=str, default='Dataset',
                    help='root folder of class subdirectories (superset of classes).')
parser.add_argument('--classes', nargs='+', default=[
    'black_eyed_susan', 'calendula', 'california_poppy', 'common_daisy',
    'coreopsis', 'dandelion', 'iris', 'rose', 'sunflower', 'tulip',
], help='which class subfolders of --data_dir to use for training.')
parser.add_argument('--train_dir', type=str, default='data_split/train',
                    help='where the training split is copied to.')
parser.add_argument('--val_dir', type=str, default='data_split/val',
                    help='where the validation split is copied to.')
parser.add_argument('--val_split', type=float, default=0.2,
                    help='fraction of each class held out for validation.')
parser.add_argument('--seed', type=int, default=42, help='random seed for the split.')

# Image / training
parser.add_argument('--img_size', type=int, default=256, help='square input resolution.')
parser.add_argument('--batch_size', type=int, default=20, help='batch size for both generators.')
parser.add_argument('--epochs', type=int, default=20, help='training epochs per model.')

# Pretrained backbones
parser.add_argument('--inception_weights', type=str,
                    default='Pre-Trained Model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    help='local InceptionV3 no-top weights file (models 2 and 4).')

# Paths
parser.add_argument('--ckpt_dir', type=str, default='checkpoints',
                    help='directory to save trained models.')
parser.add_argument('--history_dir', type=str, default='history',
                    help='directory to save per-model training history (json).')
parser.add_argument('--figures_dir', type=str, default='figures',
                    help='directory to save output plots.')

options = parser.parse_args()
