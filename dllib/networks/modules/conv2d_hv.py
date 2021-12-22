#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/12/22 下午5:34  
@Author: Rocsky
@Project: dllib
@File: conv2d_hv.py
@Version: 0.1
@Description:
"""
from typing import Union
import torch.nn as nn
from torch.nn.common_types import _size_2_t


class Conv2dHV(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 norm_layer: Union[str, nn.Module] = nn.BatchNorm2d,
                 stride: _size_2_t = 1,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True):
        """
        Do conv2d horizontally and vertically to reduce MADs
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param norm_layer:
        :param stride:
        :param dilation:
        :param groups:
        :param bias: if use bias in the point-wise conv2d
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        if isinstance(norm_layer, str):
            norm_layer = eval(norm_layer)
        if groups == 0:
            groups = in_channels
        layers = [
            nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size[0], 1), stride=(stride[0], 1),
                      padding=((kernel_size[0] - 1) // 2 * dilation[0], 0), dilation=(dilation[0], 1),
                      groups=groups, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_size[1]), stride=(1, stride[1]),
                      padding=(0, (kernel_size[1] - 1) // 2 * dilation[1]), dilation=(1, dilation[1]),
                      groups=groups, bias=False)
        ]
        if norm_layer:
            layers.append(norm_layer(in_channels))
        layers.append(
            nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        )
        if norm_layer:
            layers.append(norm_layer(out_channels))
        layers.append(nn.ReLU(inplace=True))
        super(Conv2dHV, self).__init__(*layers)
