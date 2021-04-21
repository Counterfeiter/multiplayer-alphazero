import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
sys.path.append("..")
from model import Model

class ConvNet(Model):
    def __init__(self, input_shape, p_shape, v_shape, block=None, num_blocks=None):
        super(ConvNet, self).__init__(input_shape, p_shape, v_shape)

        in_channels = input_shape[-1]
        feature_dim = 2048

        self.shared_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(7744, feature_dim))

        self.p_head = nn.Linear(feature_dim, np.prod(p_shape))
        self.v_head = nn.Linear(feature_dim, np.prod(v_shape))

    def forward(self, x):
        batch_size = len(x)
        this_p_shape = tuple([batch_size] + list(self.p_shape))
        this_v_shape = tuple([batch_size] + list(self.v_shape))
        x = x.permute(0,3,1,2) # NHWC -> NCHW

        out = self.shared_layers(x)
        flat = out.view(out.size(0), -1)
        
        p_logits = self.p_head(flat).view(this_p_shape)
        v = torch.tanh(self.v_head(flat).view(this_v_shape))
        
        return p_logits, v