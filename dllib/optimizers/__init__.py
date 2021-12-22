#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/12/20 下午3:34  
@Author: Rocsky
@Project: dllib
@File: __init__.py
@Version: 0.1
@Description:
"""
import torch
from .scheduler import PolyLR

OPTIMIZER = {
    'Adadelta': torch.optim.Adadelta,
    'Adagrad': torch.optim.Adagrad,
    'Adam': torch.optim.Adam,
    'AdamW': torch.optim.AdamW,
    'SparseAdam': torch.optim.SparseAdam,
    'Adamax': torch.optim.Adamax,
    'ASGD': torch.optim.ASGD,
    'LBFGS': torch.optim.LBFGS,
    'RMSprop': torch.optim.RMSprop,
    'Rprop': torch.optim.Rprop,
    'SGD': torch.optim.SGD
}

SCHEDULER = {
    'LambdaLR': torch.optim.lr_scheduler.LambdaLR,
    # 'MultiplicativeLR': torch.optim.lr_scheduler.MultiplicativeLR,
    'StepLR': torch.optim.lr_scheduler.StepLR,
    'MultiStepLR': torch.optim.lr_scheduler.MultiStepLR,
    'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR,
    'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR,
    'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'CyclicLR': torch.optim.lr_scheduler.CyclicLR,
    # 'OneCycleLR': torch.optim.lr_scheduler.OneCycleLR,
    'CosineAnnealingWarmRestarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    # 'PolyLR': PolyLR could be replaced by LambdaLR
}
