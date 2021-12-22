#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/12/22 下午5:32  
@Author: Rocsky
@Project: dllib
@File: conv2d_ws.py
@Version: 0.1
@Description:
"""
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t


class Conv2dWS(nn.Conv2d):
    """
    Conv2d with weight scale
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros'):
        super(Conv2dWS, self).__init__(in_channels,
                                       out_channels,
                                       kernel_size,
                                       stride,
                                       padding,
                                       dilation,
                                       groups,
                                       bias,
                                       padding_mode)

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight
        weight = weight.to(x.device)
        weight_mean = weight.mean(dim=[1, 2, 3], keepdim=True)
        weight = weight - weight_mean
        weight_std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12) + 1e-5
        weight_std = weight_std.view(-1, 1, 1, 1)
        weight = weight / weight_std
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
