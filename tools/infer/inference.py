#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/7/16 上午10:57  
@Author: Rocsky
@Project: depoly
@File: inference.py
@Version: 0.1
@Description:
"""
from typing import List
import torch
import onnxruntime


def inference_torch(model: torch.nn.Module, inputs: List):
    model.eval()
    model = model.cpu()
    inputs = [torch.from_numpy(input) for input in inputs]
    with torch.no_grad():
        outputs = model(*inputs)
    if isinstance(outputs, list):
        return [output.numpy() for output in outputs]
    else:
        return [outputs.numpy()]


def inference_script(model_path: str, inputs: List):
    model = torch.jit.load(model_path)
    inputs = [torch.from_numpy(input) for input in inputs]
    with torch.no_grad():
        outputs = model(*inputs)
    if isinstance(outputs, list):
        return [output.numpy() for output in outputs]
    else:
        return [outputs.numpy()]


def inference_onnx(model_path: str, inputs: List):
    session = onnxruntime.InferenceSession(model_path)
    input_nodes = session.get_inputs()
    input_names = [node.name for node in input_nodes]
    inputs = dict(zip(input_names, inputs))
    return session.run([], inputs)