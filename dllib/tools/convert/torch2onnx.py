#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/7/16 下午2:11  
@Author: Rocsky
@Project: dllib
@File: torch2onnx.py
@Version: 0.1
@Description:
"""
from typing import List
import numpy as np
import torch
from ..infer import inference_torch, inference_onnx, benchmark_torch, benchmark_onnx


def convert(model: torch.nn.Module,
            save_path: str,
            input_names: List,
            input_shapes: List,
            output_names: List):
    model.eval()
    model = model.cpu()
    inputs = [torch.rand(tuple(shape)).type(torch.FloatTensor) for shape in input_shapes]
    torch.onnx.export(model,
                      tuple(inputs),
                      save_path,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=input_names,
                      output_names=output_names)
    print('[1/3]: check model...')
    try:
        import onnx

        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print('check model successfully!')
    except Exception as e:
        print('check model failed!')
        print(e)
        return
    print('[2/3]: check accuracy...')
    inputs = [np.random.uniform(0, 1, tuple(shape)).astype(np.float32) for shape in input_shapes]
    outputs_torch = inference_torch(model, inputs)
    outputs_onnx = inference_onnx(save_path, inputs)
    for ot, oo in zip(outputs_torch, outputs_onnx):
        try:
            np.testing.assert_allclose(ot, oo, rtol=1e-3, atol=1e-5)
            print('check accuracy successfully!')
        except Exception as e:
            print('check accuracy failed!')
            print(e)
            return
    print('[3/3]: check latency')
    try:
        latency_torch = benchmark_torch(model, input_shapes)
        print('latency of torch: %.2f ms' % latency_torch)
        latency_onnx = benchmark_onnx(save_path)
        print('latency of onnx: %.2f ms' % latency_onnx)
        print('check latency successfully!')
    except Exception as e:
        print('check latency failed!')
        print(e)
        return
