#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/12/20 下午4:09  
@Author: Rocsky
@Project: dllib
@File: acc_utils.py
@Version: 0.1
@Description:
"""
from typing import Union, Dict
import numpy as np
import torch


class AccRecord(object):
    """
    accumulate record and get the average value
    """

    def __init__(self):
        self.records = []
        self.length = 0
        self.acc_sum = 0.0

    def update_loss(self, record: Union[np.ndarray, torch.Tensor]):
        if isinstance(record, np.ndarray):
            record = record.tolist()
        if isinstance(record, torch.Tensor):
            record = record.detach().cpu().numpy().tolist()
        self.records.append(record)
        self.acc_sum += record
        self.length += 1

    def result(self) -> float:
        return self.acc_sum / self.length if self.length > 0 else 0.0

    def reset_state(self):
        self.records = []
        self.length = 0
        self.acc_sum = 0.0


class AccRecords(object):
    """
    manipulate records according to their names
    """

    def __init__(self, names):
        self.acc_records = {}
        for name in names:
            self.acc_records[name] = AccRecord()

    def update_loss(self, records: Dict):
        for k, v in records.items():
            self.acc_records[k].update_loss(v)

    def results(self) -> Dict:
        results = {}
        for k in self.acc_records.keys():
            results[k] = self.acc_records[k].result()
        return results

    def __str__(self) -> str:
        s = ['%s: %.5f' % (k, v) for k, v in self.results().items()]
        return ', '.join(s)

    def reset_state(self):
        for record in self.acc_records.values():
            record.reset_state()
