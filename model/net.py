import torch
from torch import nn
from typing import Tuple


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x)


class TwinNet(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv_block1 = self._create_conv_block(in_channels, 64, conv_kernel_size=(10, 10))
        self.conv_block2 = self._create_conv_block(64, 128, conv_kernel_size=(10, 10))
        self.conv_block3 = self._create_conv_block(128, 128, conv_kernel_size=(4, 4))
        self.final_block = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=256 * 256, out_features=266 * 256)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.conv_block1(x)
        result = self.conv_block2(result)
        result = self.conv_block3(result)
        result = self.final_block(result)
        return result

    def _create_conv_block(self,
                           in_channels: int,
                           out_channels: int,
                           conv_kernel_size: Tuple,
                           pool_kernel_size: Tuple = (2, 2),
                           ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel_size, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2),
        )
