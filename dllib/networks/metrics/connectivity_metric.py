#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/12/22 下午5:19  
@Author: Rocsky
@Project: dllib
@File: connectivity_metric.py
@Version: 0.1
@Description:
"""
import cv2
import numpy as np
from numpy import ndarray


def connectivity_metric(y_pred: ndarray, y_true: ndarray) -> ndarray:
    step = 0.1
    thresh_steps = np.arange(0, 1 + step, step)
    round_down_map = -np.ones_like(y_true)
    for i in range(1, len(thresh_steps)):
        true_thresh = y_true >= thresh_steps[i]
        pred_thresh = y_pred >= thresh_steps[i]
        intersection = (true_thresh & pred_thresh).astype(np.uint8)

        # connected components
        _, output, stats, _ = cv2.connectedComponentsWithStats(
            intersection, connectivity=4)
        # start from 1 in dim 0 to exclude background
        size = stats[1:, -1]

        # largest connected component of the intersection
        omega = np.zeros_like(y_true)
        if len(size) != 0:
            max_id = np.argmax(size)
            # plus one to include background
            omega[output == max_id + 1] = 1

        mask = (round_down_map == -1) & (omega == 0)
        round_down_map[mask] = thresh_steps[i - 1]
    round_down_map[round_down_map == -1] = 1

    true_diff = y_true - round_down_map
    pred_diff = y_pred - round_down_map
    # only calculate difference larger than or equal to 0.15
    true_phi = 1 - true_diff * (true_diff >= 0.15)
    pred_phi = 1 - pred_diff * (pred_diff >= 0.15)

    connectivity_error = np.sum(np.abs(true_phi - pred_phi))
    return connectivity_error / 1000


class ConnectivityMetric(object):
    def __call__(self, y_pred: ndarray, y_true: ndarray) -> ndarray:
        connectivity_metric(y_pred, y_true)
