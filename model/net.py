import torch
from torch import nn
from typing import Tuple


class TwinNet(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.block1 = self._create_block(in_channels, 64, conv_kernel_size=(10, 10))
        self.block2 = self._create_block(64, 128, conv_kernel_size=(10, 10))
        self.block3 = self._create_block(128, 128, conv_kernel_size=(4, 4))
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=1),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        result = self.block1(x)
        result = self.block2(result)
        result = self.block3(result)
        result = self.block4(result)
        return result

    def _create_block(self,
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
