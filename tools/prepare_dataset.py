import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from options import options as opt
from utils.data_utils import count_files, split_dataset
from utils.eda_utils import check_uniform_resolution, count_images_per_class, plot_class_distribution


def main():
    os.makedirs(opt.figures_dir, exist_ok=True)

    resolutions = check_uniform_resolution(opt.data_dir)
    if len(resolutions) == 1:
        print(f"Uniform resolution: {resolutions.pop()}")
    else:
        print(f"Non-uniform resolutions: {resolutions}")

    class_counts = count_images_per_class(opt.data_dir, opt.classes)
    plot_class_distribution(
        class_counts, save_path=os.path.join(opt.figures_dir, "class_distribution.png")
    )

    split_dataset(
        opt.data_dir, opt.classes, opt.train_dir, opt.val_dir,
        val_split=opt.val_split, seed=opt.seed,
    )
    print(f"train: {count_files(opt.train_dir)}  val: {count_files(opt.val_dir)}")


if __name__ == "__main__":
    main()
