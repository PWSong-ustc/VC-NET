import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from src.resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from src.aspp import ASPP, ASPP_Bottleneck

class DeepLabV3(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3, self).__init__()
        self.num_classes = num_classes
        self.resnet = ResNet50_OS16()   # NOTE! specify the type of ResNet here
        # self.aspp = ASPP(num_classes=self.num_classes) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead
        self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)

        # ProjHead
        self.ProjHead = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # MLP
        self.fc1 = nn.Linear(128, 128)  # 256 256
        self.depthwise = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1,stride=1,padding=0,
                                   dilation=1,groups=128,bias=False)
        self.pointwise = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1,stride=1,padding=0,
                                   dilation=1,groups=1,bias=False)
        self.act = nn.GELU()
        # self.fc2 = nn.Linear(128, 128)
        self.drop = nn.Dropout(0)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))

        output = F.upsample(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))

        x_projhead = F.normalize(self.ProjHead(feature_map), p=2, dim=1)  # 128
        '''DW_MLP'''
        x_conv = self.pointwise(self.depthwise(x_projhead))
        x_fc = self.fc1(x_projhead.permute(0,2,3,1))
        x_fc = x_fc.permute(0,3,1,2)
        xx = self.drop(self.act(x_fc + x_conv))  # 256

        return xx, {'out': output}
