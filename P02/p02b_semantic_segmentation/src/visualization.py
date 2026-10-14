import numpy as np
import torch
from .dataset import MEAN, STD


def mask_rgb(mask):
    colors = np.zeros((*np.asarray(mask).shape, 3), dtype=np.uint8)
    colors[np.asarray(mask) == 1] = (74, 182, 117)
    colors[np.asarray(mask) == 255] = (220, 90, 200)
    return colors


def plot_history(history, output):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(10, 4))
    axes = fig.subplots(1, 2)
    epochs = [r["epoch"] for r in history]
    for split in ("train", "val"):
        axes[0].plot(epochs, [r[split]["loss"] for r in history], label=split)
        axes[1].plot(epochs, [r[split]["miou"] for r in history], label=split)
    for ax, title in zip(axes, ("Cross entropy / píxel válido", "mIoU")):
        ax.set(xlabel="Época", title=title)
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output, dpi=160)


@torch.inference_mode()
def comparison_figure(dataset, models, indices, device, output):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(12, 3 * len(indices)))
    axes = fig.subplots(len(indices), 4, squeeze=False)
    for row, index in enumerate(indices):
        x, target = dataset[index]
        original = (x * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]).permute(1, 2, 0).numpy().clip(0, 1)
        preds = [model.eval()(x[None].to(device)).argmax(1)[0].cpu().numpy() for model in models]
        for col, image in enumerate([original, mask_rgb(target), *[mask_rgb(p) for p in preds]]):
            axes[row, col].imshow(image)
            axes[row, col].axis("off")
        for col, title in enumerate(["Original", "GT (magenta: ignorado)", "FCN-lite", "Skip decoder"]):
            axes[row, col].set_title(title + ("\n" + dataset.ids[index] if col == 0 else ""), fontsize=9)
    fig.suptitle("Negro: fondo · verde: mascota · magenta: no evaluado", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output, dpi=160)
