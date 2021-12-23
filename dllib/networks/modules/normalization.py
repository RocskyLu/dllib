#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/12/22 下午5:27  
@Author: Rocsky
@Project: dllib
@File: normalization.py
@Version: 0.1
@Description:
"""
import torch.nn as nn


class GroupNorm2d(nn.GroupNorm):
    def __init__(self, num_features: int, num_groups: int = 32):
        super(GroupNorm2d, self).__init__(num_channels=num_features, num_groups=num_groups)
