#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/7/16 下午1:42  
@Author: Rocsky
@Project: depoly
@File: benchmark.py
@Version: 0.1
@Description:
"""
from typing import List
import time
import numpy as np
import torch
import onnxruntime


def benchmark_torch(model: torch.nn.Module, input_shapes: List, show_detail: bool = False):
    model.eval()
    model = model.cpu()
    total = 0.0
    warm_ups = 10
    runs = 100
    inputs = [torch.rand(tuple(shape)).type(torch.FloatTensor) for shape in input_shapes]
    with torch.no_grad():
        for i in range(warm_ups):
            _ = model(*inputs)
        for i in range(runs):
            start = time.perf_counter()
            _ = model(*inputs)
            end = (time.perf_counter() - start) * 1000
            total += end
            if show_detail:
                print(f"{end:.2f}ms")
    avg_latency = total / runs
    return avg_latency


def benchmark_script(model_path: str, input_shapes: List, show_detail: bool = False):
    model = torch.jit.load(model_path)
    total = 0.0
    warm_ups = 10
    runs = 100
    inputs = [torch.rand(tuple(shape)).type(torch.FloatTensor) for shape in input_shapes]
    with torch.no_grad():
        for i in range(warm_ups):
            _ = model(*inputs)
        for i in range(runs):
            start = time.perf_counter()
            _ = model(*inputs)
            end = (time.perf_counter() - start) * 1000
            total += end
            if show_detail:
                print(f"{end:.2f}ms")
    avg_latency = total / runs
    return avg_latency


def benchmark_onnx(model_path: str, show_detail: bool = False):
    session = onnxruntime.InferenceSession(model_path)
    input_nodes = session.get_inputs()
    input_names = [node.name for node in input_nodes]
    input_shapes = [node.shape for node in input_nodes]
    for shape in input_shapes:
        if type(shape[0]) == str:
            shape[0] = 1
    total = 0.0
    warm_ups = 10
    runs = 100
    inputs = [np.random.random(tuple(shape)).astype(np.float32) for shape in input_shapes]
    inputs = dict(zip(input_names, inputs))
    for i in range(warm_ups):
        _ = session.run([], inputs)
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], inputs)
        end = (time.perf_counter() - start) * 1000
        total += end
        if show_detail:
            print(f"{end:.2f}ms")
    avg_latency = total / runs
    return avg_latency
