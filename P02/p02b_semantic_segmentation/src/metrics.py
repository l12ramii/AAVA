import torch
from .dataset import IGNORE_INDEX


class SegmentationMetrics:
    """Filas=real, columnas=predicho; se acumula el conjunto entero antes de dividir."""
    def __init__(self):
        self.confusion = torch.zeros((2, 2), dtype=torch.int64)

    def update(self, prediction, target):
        prediction, target = prediction.detach().cpu(), target.detach().cpu()
        if prediction.shape != target.shape:
            raise ValueError("Predicción y etiqueta deben tener la misma forma.")
        valid = target != IGNORE_INDEX
        p, t = prediction[valid].long(), target[valid].long()
        if ((p < 0) | (p > 1) | (t < 0) | (t > 1)).any():
            raise ValueError("Se esperan clases 0 y 1.")
        self.confusion += torch.bincount(2 * t + p, minlength=4).reshape(2, 2)

    def compute(self):
        """Una clase sin unión devuelve None y no participa en mIoU; sin válidos, todo None."""
        # TODO T4_METRICS: Calcula IoU fondo/mascota, miou y pixel_accuracy a partir de self.confusion; unión cero -> None.
        raise NotImplementedError("T4_METRICS: Calcula IoU fondo/mascota, miou y pixel_accuracy a partir de self.confusion; unión cero -> None.")
