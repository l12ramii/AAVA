import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.nn import functional as F
from .dataset import image_tensor
from .engine import load_checkpoint


@torch.inference_mode()
def predict_mask(model, image, size=128, device="cpu"):
    """Devuelve uint8 H×W (0/1) al tamaño original. Orientar EXIF antes de llamar."""
    # TODO T5_INFERENCE: Usa image_tensor, batch, dispositivo, eval, logits al tamaño original y argmax -> numpy uint8.
    raise NotImplementedError("T5_INFERENCE: Usa image_tensor, batch, dispositivo, eval, logits al tamaño original y argmax -> numpy uint8.")


@torch.inference_mode()
def predict_mask_labels(model, image, size=128, device="cpu"):
    """T5_LABELS: mismo preprocessing; argmax a resolución de red y resize NEAREST.
    Devuelve numpy uint8 HxW original con valores 0/1; no interpola logits.
    """
    # TODO T5_LABELS: Implementa inferencia argmax y resize nearest de etiquetas según el contrato.
    raise NotImplementedError("T5_LABELS: Implementa inferencia argmax y resize nearest de etiquetas según el contrato.")


def apply_effect(image, mask, effect, color="#86b6d8", blur=12):
    image = image.convert("RGB")
    mask = np.asarray(mask)
    if mask.shape != (image.height, image.width) or not np.isin(mask, [0, 1]).all():
        raise ValueError("La máscara debe ser binaria y tener el tamaño original.")
    alpha = Image.fromarray(mask.astype(np.uint8) * 255)
    if effect == "Sticker":
        result = image.convert("RGBA")
        result.putalpha(alpha)
        return result
    if effect == "Desenfocar fondo":
        background = image.filter(ImageFilter.GaussianBlur(float(blur)))
    elif effect == "Fondo de color":
        background = Image.new("RGB", image.size, color)
    elif effect == "Fondo gris":
        background = ImageOps.grayscale(image).convert("RGB")
    else:
        raise ValueError("Efecto desconocido.")
    return Image.composite(image, background, alpha)


class PetCut:
    def __init__(self, checkpoint, device="cpu"):
        self.model, record = load_checkpoint(checkpoint, device)
        self.size, self.device = record["config"]["size"], device

    def __call__(self, image, effect, color="#86b6d8", blur=12):
        if image is None:
            raise ValueError("Seleccione una fotografía.")
        image = ImageOps.exif_transpose(image).convert("RGB")
        mask = predict_mask(self.model, image, self.size, self.device)
        result = apply_effect(image, mask, effect, color, blur)
        overlay = Image.blend(image, apply_effect(image, mask, "Fondo de color", "#ff7f50"), 0.45)
        return Image.fromarray(mask * 255), overlay, result, (image, result)
