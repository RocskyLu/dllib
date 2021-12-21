#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/9/14 下午5:21  
@Author: Rocsky
@Project: ailib
@File: conv2d_filter.py
@Version: 0.1
@Description:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t


class Conv2dFilter(nn.Conv2d):
    def __init__(self,
                 kernel: torch.FloatTensor,
                 in_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 0,
                 dilation: _size_2_t = 1,
                 padding_mode: str = 'zeros'):
        """

        :param kernel:
        :param in_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param padding_mode:
        """
        super(Conv2dFilter, self).__init__(in_channels=in_channels,
                                           out_channels=in_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           dilation=dilation,
                                           groups=in_channels,
                                           bias=False,
                                           padding_mode=padding_mode)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        assert kernel.shape[0] == kernel_size[0] and kernel.shape[1] == kernel_size[1], \
            'kernel\'s shape should be equal to kernel_size'
        self.weight = nn.Parameter(kernel.expand_as(self.weight), requires_grad=False)


def conv2d_filter(x: torch.FloatTensor,
                  kernel: torch.FloatTensor,
                  stride: _size_2_t = 1,
                  padding: _size_2_t = 0,
                  dilation: _size_2_t = 1,
                  padding_mode: str = 'zeros'):
    """

    :param x:
    :param kernel:
    :param stride:
    :param padding:
    :param dilation:
    :param padding_mode:
    :return:
    """
    _, in_channels = x.shape[:2]
    kernel = kernel.unsqueeze(dim=0).unsqueeze(dim=0)
    kernel = torch.cat([kernel] * in_channels, dim=0)
    if padding_mode == 'zeros':
        x = F.conv2d(x, kernel, stride=stride, padding=padding, dilation=dilation, groups=in_channels)
    else:
        x = F.pad(x, (padding, padding, padding, padding), mode=padding_mode)
        x = F.conv2d(x, kernel, stride=stride, padding=0, dilation=dilation, groups=in_channels)
    return x


class GaussianBlur(Conv2dFilter):
    def __init__(self,
                 kernel_size,
                 in_channels):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        padding = ((kernel_size[0]) // 2, (kernel_size[1]) // 2)
        kernel = torch.rand(kernel_size).type(torch.FloatTensor)
        super(GaussianBlur, self).__init__(kernel=kernel,
                                           in_channels=in_channels,
                                           kernel_size=kernel_size,
                                           stride=1,
                                           padding=padding,
                                           dilation=1)


class BoxBlur(Conv2dFilter):
    def __init__(self,
                 kernel_size,
                 in_channels):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        padding = ((kernel_size[0]) // 2, (kernel_size[1]) // 2)
        kernel = torch.ones(kernel_size).type(torch.FloatTensor) / (kernel_size[0] * kernel_size[1])
        super(BoxBlur, self).__init__(kernel=kernel,
                                      in_channels=in_channels,
                                      kernel_size=kernel_size,
                                      stride=1,
                                      padding=padding,
                                      dilation=1)


if __name__ == '__main__':
    x = torch.rand(1, 3, 100, 100).type(torch.FloatTensor)
    blur = BoxBlur(3, 3)
    y = blur(x)
    print(y.shape)
