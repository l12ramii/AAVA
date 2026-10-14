import torch
from torch import nn
from torch.nn import functional as F
from .encoder import FrozenEncoder


class FCNLite(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = FrozenEncoder(pretrained)
        # TODO T1_INIT: Crea self.classifier: convolución 1x1 de 512 a 2 canales.
        raise NotImplementedError("T1_INIT: Crea self.classifier: convolución 1x1 de 512 a 2 canales.")

    def forward(self, x):
        # TODO T1_FORWARD: Encoder -> última escala -> classifier -> bilinear al tamaño de x. Devuelve logits.
        raise NotImplementedError("T1_FORWARD: Encoder -> última escala -> classifier -> bilinear al tamaño de x. Devuelve logits.")


class DecoderBlock(nn.Module):
    def __init__(self, deep_channels, skip_channels, out_channels):
        super().__init__()
        # TODO T2_INIT: Define self.refine: dos Conv3x3+ReLU; entrada deep_channels+skip_channels, salida out_channels.
        raise NotImplementedError("T2_INIT: Define self.refine: dos Conv3x3+ReLU; entrada deep_channels+skip_channels, salida out_channels.")

    def forward(self, x, skip):
        # TODO T2_FORWARD: Interpola x al tamaño de skip, concatena por canales y aplica self.refine.
        raise NotImplementedError("T2_FORWARD: Interpola x al tamaño de skip, concatena por canales y aplica self.refine.")


class SkipSegmenter(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = FrozenEncoder(pretrained)
        self.decode3 = DecoderBlock(512, 256, 128)
        self.decode2 = DecoderBlock(128, 128, 64)
        self.decode1 = DecoderBlock(64, 64, 32)
        self.classifier = nn.Conv2d(32, 2, 1)

    def forward(self, x):
        # TODO T3_FORWARD: Extrae f1..f4, encadena decode3/2/1 con sus skips y clasifica/restaura tamaño.
        raise NotImplementedError("T3_FORWARD: Extrae f1..f4, encadena decode3/2/1 con sus skips y clasifica/restaura tamaño.")


def build_model(name, pretrained=True):
    if name not in ("fcn", "skip"):
        raise ValueError("Modelo desconocido: use fcn o skip.")
    return {"fcn": FCNLite, "skip": SkipSegmenter}[name](pretrained)
