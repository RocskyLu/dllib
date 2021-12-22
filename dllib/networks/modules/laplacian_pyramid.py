#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/12/22 下午5:08  
@Author: Rocsky
@Project: dllib
@File: laplacian_pyramid.py
@Version: 0.1
@Description:
"""
from typing import List
from torch import Tensor
import torch.nn as nn
from kornia.geometry.transform.pyramid import PyrUp, PyrDown


class LaplacianPyramid(nn.Module):
    def __init__(self, depth: int = 1):
        """

        :param depth:
        """
        super(LaplacianPyramid, self).__init__()
        self.pyrup = PyrUp()
        self.pyrdown = PyrDown()
        self.depth = depth

    def forward(self, x: Tensor) -> List[Tensor]:
        x_pyramids = []
        for i in range(self.depth):
            x_down = self.pyrdown(x)
            x_pyramids.append(x - self.pyrup(x_down))
            x = x_down
        return x_pyramids


def laplacian_pyramid(x: Tensor, depth: int = 1) -> List[Tensor]:
    """

    :param x:
    :param depth:
    :return:
    """
    pyrup = PyrUp()
    pyrdown = PyrDown()
    x_pyramids = []
    for i in range(depth):
        x_down = pyrdown(x)
        x_pyramids.append(x - pyrup(x_down))
        x = x_down
    return x_pyramids
