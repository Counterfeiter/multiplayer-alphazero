import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
sys.path.append("..")
from model import Model

class FFNet(Model):
    def __init__(self, input, p_shape, v_shape):
        super(FFNet, self).__init__(input, p_shape, v_shape)

        outsize = (input["ff_input"].shape[0] + np.prod(p_shape)) // 2

        self.linear = nn.Sequential(
            nn.Linear(input["ff_input"].shape[0], outsize),
            nn.ReLU(),
            nn.Linear(outsize, outsize),
        )

        self.p_head = nn.Sequential(
            nn.Linear(outsize, np.prod(p_shape))
        )

        self.v_head = nn.Sequential(
            nn.Linear(outsize, np.prod(v_shape))
        )

    def forward(self, input_dict):
        batch_size = len(input_dict["ff_input"])
        this_p_shape = tuple([batch_size] + list(self.p_shape))
        this_v_shape = tuple([batch_size] + list(self.v_shape))

        ff_out = self.linear(input_dict["ff_input"])

        p_logits = self.p_head(ff_out).view(this_p_shape)
        v = torch.tanh(self.v_head(ff_out).view(this_v_shape))
        
        return p_logits, v