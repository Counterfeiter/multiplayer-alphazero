import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
sys.path.append("..")
from model import Model


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes: 
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out


class SENetMixed(Model):
    def __init__(self, input, p_shape, v_shape, block=PreActBlock, num_blocks=[3,4,6,3]):
        super(SENetMixed, self).__init__(input, p_shape, v_shape)

        self.in_planes = 64
        self.conv1 = nn.Conv2d(input["cnn_input"].shape[-1], 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.parallel_linear = nn.Sequential(
            nn.Linear(input["ff_input"].shape[0], input["ff_input"].shape[0] * 2),
            #nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Linear(512 + input["ff_input"].shape[0] * 2, 1024),
            #nn.ReLU()
        )

        self.p_head = nn.Sequential(
            nn.Linear(1024, np.prod(p_shape))
        )

        self.v_head = nn.Sequential(
            nn.Linear(1024, np.prod(v_shape))
        )

        #self.p_head = torch.nn.Linear(512, np.prod(p_shape))
        #self.v_head = torch.nn.Linear(512, np.prod(v_shape))



    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, input_dict):
        x = input_dict["cnn_input"]
        batch_size = len(x)
        this_p_shape = tuple([batch_size] + list(self.p_shape))
        this_v_shape = tuple([batch_size] + list(self.v_shape))
        x = x.permute(0,3,1,2) # NHWC -> NCHW

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        flat = out.view(out.size(0), -1)

        ff_out = self.parallel_linear(input_dict["ff_input"])

        flat = self.linear(torch.cat((flat, ff_out), -1))

        p_logits = self.p_head(flat).view(this_p_shape)
        v = torch.tanh(self.v_head(flat).view(this_v_shape))
        
        return p_logits, v