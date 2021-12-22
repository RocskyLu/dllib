#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/12/21 上午11:48  
@Author: Rocsky
@Project: dllib
@File: hard_sigmoid.py
@Version: 0.1
@Description:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HardSigmoid(nn.Module):
    def __init__(self):
        super(HardSigmoid, self).__init__()

    def forward(self, x):
        return F.relu6(x + 3) / 6


def hard_sigmoid(x):
    return F.relu6(x + 3) / 6
