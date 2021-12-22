#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/12/22 下午5:24  
@Author: Rocsky
@Project: dllib
@File: basic_metrics.py
@Version: 0.1
@Description:
"""
import numpy as np
from numpy import ndarray


def sad_metric(y_pred: ndarray, y_true: ndarray) -> ndarray:
    return np.abs(y_pred - y_true).sum() / 1e3


class SADMetric(object):
    def __call__(self, y_pred: ndarray, y_true: ndarray) -> ndarray:
        return sad_metric(y_pred, y_true)


def mad_metric(y_pred: ndarray, y_true: ndarray) -> ndarray:
    return np.abs(y_pred - y_true).mean() * 1e3


class MADMetric(object):
    def __call__(self, y_pred: ndarray, y_true: ndarray) -> ndarray:
        return mad_metric(y_pred, y_true)


def mse_metric(y_pred: ndarray, y_true: ndarray) -> ndarray:
    return ((y_pred - y_true) ** 2).mean() * 1e3


class MSEMetric(object):
    def __call__(self, y_pred: ndarray, y_true: ndarray) -> ndarray:
        return mse_metric(y_pred, y_true)
