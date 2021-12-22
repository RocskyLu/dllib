#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/12/22 下午5:27  
@Author: Rocsky
@Project: dllib
@File: pixel_shuffle.py
@Version: 0.1
@Description:
"""
from torch import Tensor
import torch.nn as nn


def pixel_shuffle(x: Tensor, upscale_factor: int) -> Tensor:
    b, c, h, w = x.shape
    assert c % upscale_factor ** 2 == 0, "channel should be divided by upscale_factor ** 2"
    x = x.view(b, c // upscale_factor ** 2, upscale_factor, upscale_factor, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3)
    x = x.contiguous().view(b, c // upscale_factor ** 2, h * upscale_factor, w * upscale_factor)
    return x


class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor: int):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x: Tensor) -> Tensor:
        return pixel_shuffle(x, self.upscale_factor)
