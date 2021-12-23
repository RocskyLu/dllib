#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/12/22 下午5:31  
@Author: Rocsky
@Project: dllib
@File: pixel_unshuffle.py
@Version: 0.1
@Description:
"""
from torch import Tensor
import torch.nn as nn


def pixel_unshuffle(x: Tensor, downscale_factor: int) -> Tensor:
    b, c, h, w = x.shape
    assert h % downscale_factor == 0, "h should be divided by upscale_factor"
    assert w % downscale_factor == 0, "w should be divided by upscale_factor"
    x = x.view(b, c, h // downscale_factor, downscale_factor, w // downscale_factor, downscale_factor)
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.contiguous().view(b, c * downscale_factor ** 2, h // downscale_factor, w // downscale_factor)
    return x


class PixelUnShuffle(nn.Module):
    def __init__(self, downscale_factor: int):
        super(PixelUnShuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x: Tensor) -> Tensor:
        return pixel_unshuffle(x, self.downscale_factor)
