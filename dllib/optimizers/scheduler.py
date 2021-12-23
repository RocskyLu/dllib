#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/6/16 下午5:25
@Author: Rocsky
@Project: dllib
@File: scheduler.py
@Version: 0.1
@Description:
"""
from torch.optim.lr_scheduler import _LRScheduler


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, epochs, last_epoch=-1, power=0.9, min_lr=1e-6):
        """

        :param optimizer:
        :param epochs:
        :param last_epoch:
        :param power:
        :param min_lr:
        """
        self.epochs = epochs
        self.power = power
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.epochs) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]


