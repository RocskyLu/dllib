#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/12/22 下午5:21  
@Author: Rocsky
@Project: dllib
@File: gradient_metric.py
@Version: 0.1
@Description:
"""
from typing import Tuple
import cv2
import numpy as np
from numpy import ndarray


def gradient_metric(y_pred: ndarray, y_true: ndarray) -> ndarray:
    metric = GradientMetric()
    return metric(y_pred, y_true)


class GradientMetric(object):
    def __init__(self, sigma: float = 1.4):
        self.filter_x, self.filter_y = self.gauss_filter(sigma)

    def __call__(self, y_pred: ndarray, y_true: ndarray) -> ndarray:
        pred_normed = np.zeros_like(y_pred)
        true_normed = np.zeros_like(y_true)
        cv2.normalize(y_pred, pred_normed, 1., 0., cv2.NORM_MINMAX)
        cv2.normalize(y_true, true_normed, 1., 0., cv2.NORM_MINMAX)

        true_grad = self.gauss_gradient(true_normed).astype(np.float32)
        pred_grad = self.gauss_gradient(pred_normed).astype(np.float32)

        grad_loss = ((true_grad - pred_grad) ** 2).sum()
        return grad_loss / 1000

    def gauss_gradient(self, img: ndarray) -> ndarray:
        img_filtered_x = cv2.filter2D(img, -1, self.filter_x, borderType=cv2.BORDER_REPLICATE)
        img_filtered_y = cv2.filter2D(img, -1, self.filter_y, borderType=cv2.BORDER_REPLICATE)
        return np.sqrt(img_filtered_x ** 2 + img_filtered_y ** 2)

    @staticmethod
    def gauss_filter(sigma: float, epsilon: float = 1e-2) -> Tuple[ndarray, ndarray]:
        half_size = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
        size = np.int(2 * half_size + 1)

        # create filter in x axis
        filter_x = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                filter_x[i, j] = GradientMetric.gaussian(i - half_size, sigma) * GradientMetric.dgaussian(
                    j - half_size, sigma)

        # normalize filter
        norm = np.sqrt((filter_x ** 2).sum())
        filter_x = filter_x / norm
        filter_y = np.transpose(filter_x)

        return filter_x, filter_y

    @staticmethod
    def gaussian(x: ndarray, sigma: float) -> ndarray:
        return np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    @staticmethod
    def dgaussian(x: ndarray, sigma: float) -> ndarray:
        return -x * GradientMetric.gaussian(x, sigma) / sigma ** 2
