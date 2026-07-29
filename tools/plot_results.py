import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt

from options import options as opt

MODEL_NAMES = [
    ("model1_cnn", "Model 1"),
    ("model2_inception", "Model 2"),
    ("model3_leaky_cnn", "Model 3"),
    ("model4_inception_leaky", "Model 4"),
    ("model5_vgg16", "Model 5"),
    ("model6_resnet50", "Model 6"),
]


def main():
    histories = {}
    for filename, label in MODEL_NAMES:
        path = os.path.join(opt.history_dir, f"{filename}.json")
        with open(path) as f:
            histories[label] = json.load(f)

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    metrics = [
        ("accuracy", "Training Accuracy Comparison Over Epochs", "Accuracy", axs[0, 0]),
        ("val_accuracy", "Validation Accuracy Comparison Over Epochs", "Accuracy", axs[0, 1]),
        ("loss", "Training Loss Comparison Over Epochs", "Loss", axs[1, 0]),
        ("val_loss", "Validation Loss Comparison Over Epochs", "Loss", axs[1, 1]),
    ]
    for key, title, ylabel, ax in metrics:
        for label, history in histories.items():
            ax.plot(history[key], label=label)
        ax.set_title(title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel(ylabel)
        ax.legend()

    plt.tight_layout()
    os.makedirs(opt.figures_dir, exist_ok=True)
    save_path = os.path.join(opt.figures_dir, "model_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
