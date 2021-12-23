#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/12/22 下午5:07  
@Author: Rocsky
@Project: dllib
@File: laplacian_pyramid_loss.py
@Version: 0.1
@Description:
"""
from typing import List, Optional
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss
from ..modules import LaplacianPyramid, laplacian_pyramid
from .basic_losses import l1_loss


class LaplacianPyramidLoss(Loss):
    def __init__(self, depth: int = 5):
        """
        :param depth:
        """
        super(LaplacianPyramidLoss, self).__init__()
        self.depth = depth
        self.laplacian_pyramid = LaplacianPyramid(depth)

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        ratios = [pow(2, i) for i in range(self.depth)]
        y_pred_pyramids = self.laplacian_pyramid(y_pred)
        y_true_pyramids = self.laplacian_pyramid(y_true)
        loss = [l1_loss(pred, true) * ratio for pred, true, ratio in zip(y_pred_pyramids, y_true_pyramids, ratios)]
        loss = sum(loss)
        return loss


def laplacian_pyramid_loss(y_pred: Tensor, y_true: Tensor, depth: int = 5, ratios: Optional[List[int]] = None) -> Tensor:
    """

    :param y_pred:
    :param y_true:
    :param depth:
    :param ratios:
    :return:
    """
    if ratios is None:
        ratios = [pow(2, i) for i in range(depth)]
    y_pred_pyramids = laplacian_pyramid(y_pred, depth)
    y_true_pyramids = laplacian_pyramid(y_true, depth)
    loss = [l1_loss(pred, true) * ratio for pred, true, ratio in zip(y_pred_pyramids, y_true_pyramids, ratios)]
    loss = sum(loss)
    return loss
