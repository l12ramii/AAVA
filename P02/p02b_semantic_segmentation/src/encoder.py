"""ResNet18 proporcionada: congelación de gradientes Y estadísticas BatchNorm."""
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class FrozenEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layers = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # El padre puede llamar model.train(): este submódulo sigue en evaluación.
        return super().train(False)

    @torch.no_grad()
    def forward(self, x):
        x = self.stem(x)
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features  # 64@H/4, 128@H/8, 256@H/16, 512@H/32
