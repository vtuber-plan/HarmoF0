import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import numpy as np
from torchaudio.transforms import Spectrogram


# Fixed Rate Dilated Casual Convolution
class FRDConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[1,3], dilation=[1, 1]) -> None:
        super().__init__()
        right = (kernel_size[1]-1) * dilation[1]
        bottom = (kernel_size[0]-1) * dilation[0]
        self.padding = nn.ZeroPad2d([0, right, 0 , bottom])
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self,x):
        x = self.padding(x)
        x = self.conv2d(x)
        return x