import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from net.backbones import load_inception_backbone
from net.models import build_model2
from options import options as opt
from train.common import train_and_save
from utils.data_utils import build_generators


def main():
    train_generator, validation_generator = build_generators(
        opt.train_dir, opt.val_dir, opt.img_size, opt.batch_size
    )

    backbone = load_inception_backbone(
        opt.inception_weights, input_shape=(opt.img_size, opt.img_size, 3)
    )
    model = build_model2(backbone, num_classes=len(opt.classes))
    model.build(input_shape=(None, opt.img_size, opt.img_size, 3))
    model.summary()

    train_and_save(
        model, "model2_inception", train_generator, validation_generator,
        opt.epochs, opt.ckpt_dir, opt.history_dir,
    )


if __name__ == "__main__":
    main()
