import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import numpy as np
from torchaudio.transforms import Spectrogram

# Multiple Rate Dilated Convolution
class MRDConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_list = [0, 12, 19, 24, 28, 31, 34, 36]):
        super().__init__()
        self.dilation_list = dilation_list
        self.conv_list = []
        for i in range(len(dilation_list)):
            self.conv_list += [nn.Conv2d(in_channels, out_channels, kernel_size = [1, 1])]
        self.conv_list = nn.ModuleList(self.conv_list)
        
    def forward(self, specgram):
        # input [b x C x T x n_freq]
        # output: [b x C x T x n_freq] 
        specgram
        dilation = self.dilation_list[0]
        y = self.conv_list[0](specgram)
        y = F.pad(y, pad=[0, dilation])
        y = y[:, :, :, dilation:]
        for i in range(1, len(self.conv_list)):
            dilation = self.dilation_list[i]
            x = self.conv_list[i](specgram)
            # => [b x T x (n_freq + dilation)]
            # x = F.pad(x, pad=[0, dilation])
            x = x[:, :, :, dilation:]
            n_freq = x.size()[3]
            y[:, :, :, :n_freq] += x

        return y