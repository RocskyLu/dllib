#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/12/22 下午4:56  
@Author: Rocsky
@Project: dllib
@File: basic_losses.py
@Version: 0.1
@Description:
"""
from typing import Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss as Loss


class L1Loss(Loss):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor, mask: Optional[Tensor] = None, reduction: str = 'mean') -> Tensor:
        return l1_loss(y_pred, y_true, mask, reduction)


def l1_loss(y_pred: Tensor, y_true: Tensor, mask: Optional[Tensor] = None, reduction='mean') -> Tensor:
    if mask is not None:
        eps = 1e-12
        dims = y_pred.dim()
        if reduction == 'mean':
            loss = torch.abs((y_pred - y_true) * mask).sum(dim=list(range(1, dims)), keepdim=True) / \
                   (mask.sum(dim=list(range(1, dims)), keepdim=True) + eps)
            loss = loss.mean()
        elif reduction == 'sum':
            loss = torch.abs((y_pred - y_true) * mask).sum()
        else:
            loss = torch.abs((y_pred - y_true) * mask)
    else:
        if reduction == 'mean':
            loss = torch.abs((y_pred - y_true)).mean()
        elif reduction == 'sum':
            loss = torch.abs((y_pred - y_true)).sum()
        else:
            loss = torch.abs((y_pred - y_true))
    return loss


class L2Loss(Loss):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor, mask=None, reduction='mean') -> Tensor:
        return l2_loss(y_pred, y_true, mask, reduction)


def l2_loss(y_pred: Tensor, y_true: Tensor, mask: Optional[Tensor] = None, reduction: str = 'mean') -> Tensor:
    if mask is not None:
        eps = 1e-12
        dims = y_pred.dim()
        if reduction == 'mean':
            loss = torch.pow((y_pred - y_true) * mask, 2).sum(dim=list(range(1, dims)), keepdim=True) / \
                   (mask.sum(dim=list(range(1, dims)), keepdim=True) + eps)
            loss = loss.mean()
        elif reduction == 'sum':
            loss = torch.pow((y_pred - y_true) * mask, 2).sum()
        else:
            loss = torch.pow((y_pred - y_true) * mask, 2)
    else:
        if reduction == 'mean':
            loss = torch.pow((y_pred - y_true), 2).mean()
        elif reduction == 'sum':
            loss = torch.pow((y_pred - y_true), 2).sum()
        else:
            loss = torch.pow((y_pred - y_true), 2)
    return loss


class BCELoss(Loss):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor, mask: Optional[Tensor] = None, reduction: str = 'mean') -> Tensor:
        return bce_loss(y_pred, y_true, mask, reduction)


def bce_loss(y_pred: Tensor, y_true: Tensor, mask: Optional[Tensor] = None, reduction: str = 'mean') -> Tensor:
    if mask is not None:
        eps = 1e-6
        dims = y_pred.dim()
        if reduction == 'mean':
            loss = F.binary_cross_entropy(y_pred * mask, y_true * mask,
                                          reduction='none').sum(dim=list(range(1, dims)), keepdim=True) / \
                   (mask.sum(dim=list(range(1, dims)), keepdim=True) + eps)
            loss = loss.mean()
        elif reduction == 'sum':
            loss = F.binary_cross_entropy(y_pred * mask, y_true * mask, reduction='none').sum()
        else:
            loss = F.binary_cross_entropy(y_pred * mask, y_true * mask, reduction='none')
    else:
        if reduction == 'mean':
            loss = F.binary_cross_entropy(y_pred, y_true, reduction='none').mean()
        elif reduction == 'sum':
            loss = F.binary_cross_entropy(y_pred, y_true, reduction='none').sum()
        else:
            loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    return loss
