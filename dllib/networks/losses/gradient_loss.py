#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/12/22 下午5:04  
@Author: Rocsky
@Project: dllib
@File: gradient_loss.py
@Version: 0.1
@Description:
"""
from typing import Optional
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss as Loss
from ..modules import Conv2dFilter


class GradientLoss(Loss):
    def __init__(self, in_channels: int, reduction: str = 'mean'):
        """
        Compute gradient loss
        :param in_channels:
        :param reduction:
        """
        super(GradientLoss, self).__init__()
        self.reduction = reduction
        kernel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).type(torch.FloatTensor)
        self.gradient_x = Conv2dFilter(kernel_x, in_channels, 3, 1, 1)
        kernel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).type(torch.FloatTensor)
        self.gradient_x = Conv2dFilter(kernel_y, in_channels, 3, 1, 1)

    def forward(self, y_pred: Tensor, y_true: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """

        :param y_pred:
        :param y_true:
        :param mask:
        :return:
        """
        y_pred_x = self.gradient_x(y_pred)
        y_pred_y = self.gradient_y(y_pred)
        y_true_x = self.gradient_x(y_true)
        y_true_y = self.gradient_y(y_true)
        if mask:
            eps = 1e-12
            dims = y_pred.dim()
            if self.reduction == 'mean':
                loss = (torch.abs((y_pred_x - y_true_x) * mask).sum(dim=list(range(1, dims)), keepdim=True) +
                        torch.abs((y_pred_y - y_true_y) * mask).sum(dim=list(range(1, dims)), keepdim=True)) / \
                       (mask.sum(dim=list(range(1, dims)), keepdim=True) + eps)
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = torch.abs((y_pred_x - y_true_x) * mask).sum() + torch.abs((y_pred_y - y_true_y) * mask).sum()
            else:
                loss = torch.abs((y_pred_x - y_true_x) * mask) + torch.abs((y_pred_y - y_true_y) * mask)
        else:
            if self.reduction == 'mean':
                loss = torch.abs((y_pred_x - y_true_x)).mean() + torch.abs(y_pred - y_true).mean()
            elif self.reduction == 'sum':
                loss = torch.abs(y_pred - y_true).sum() + torch.abs(y_pred_y - y_true_x).sum()
            else:
                loss = torch.abs(y_pred_x - y_true_x) + torch.abs(y_pred_y - y_pred_y)
        return loss
