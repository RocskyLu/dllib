#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/7/16 下午1:48  
@Author: Rocsky
@Project: dllib
@File: calibrate_data.py
@Version: 0.1
@Description:
"""
from typing import Tuple, List, Dict, Optional
import numpy as np
import torch
import torch.utils.data as data
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader


class DummyDatasetTorch(data.Dataset):
    def __init__(self, input_shapes: List, target_shapes: Optional[List], length: int = 100):
        self.input_shapes = input_shapes
        self.target_shapes = target_shapes
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[List, Optional[List]]:
        inputs = [torch.rand(size).type(torch.FloatTensor) for size in self.input_shapes]
        if self.target_shapes:
            targets = [torch.rand(size).type(torch.FloatTensor) for size in self.target_shapes]
        else:
            targets = None
        return inputs, targets

    @staticmethod
    def data_collate(batch) -> Tuple[List, List]:
        inputs = []
        targets = []
        for item in batch:
            if inputs:
                for x in item[0]:
                    inputs.append([x])
            else:
                for i, x in enumerate(item[0]):
                    inputs[i].append(x)
            if item[1] is None:
                targets = None
            else:
                if targets:
                    for x in item[1]:
                        targets.append([x])
                else:
                    for i, x in enumerate(item[1]):
                        targets[i].append(x)
        return [torch.cat(item, 0) for item in inputs], \
               [torch.cat(item, 0) for item in targets] if targets is not None else targets


class DummyDatasetONNX(CalibrationDataReader):
    def __init__(self, model_path: str, length: int = 100):
        session = onnxruntime.InferenceSession(model_path)
        input_nodes = session.get_inputs()
        self.input_names = [node.name for node in input_nodes]
        self.input_shapes = [node.shape for node in input_nodes]
        for shape in self.input_shapes:
            if type(shape[0]) == str:
                shape[0] = 1
        self.length = length

    def get_next(self) -> Optional[Dict]:
        if self.length > 0:
            inputs = [np.random.random(size).astype(np.float32) for size in self.input_shapes]
            inputs = dict(zip(self.input_names, inputs))
            inputs = iter([inputs])
            self.length = self.length - 1
        else:
            inputs = iter([])
        return next(inputs, None)
