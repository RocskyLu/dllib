#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/7/16 下午3:59  
@Author: Rocsky
@Project: depoly
@File: quant.py
@Version: 0.1
@Description:
"""
from typing import List, Optional
import numpy as np
import torch
from onnxruntime import quantization
from onnxruntime.quantization import CalibrationDataReader
from ..infer import *


def quant_torch(model: torch.nn.Module,
                save_path: str,
                mode: str = 'static',
                input_shapes: Optional[List] = None,
                fuse: Optional[List] = None):
    model.eval()
    model = model.cpu()
    print('start quantization')
    quant_model = model
    quant_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.fuse_modules(quant_model, fuse, inplace=True)
    if mode == 'static':
        import torch.utils.data as data
        from .calibrate_data import DummyDatasetTorch

        assert input_shapes is not None, 'input_shapes should not be none when in static mode'
        dataset = DummyDatasetTorch(input_shapes)
        data_loader = data.DataLoader(dataset,
                                      batch_size=1,
                                      num_workers=1,
                                      collate_fn=dataset.data_collate)
        torch.quantization.prepare(quant_model, inplace=True)
        for inputs in data_loader:
            quant_model(*inputs)
    elif mode == 'qat':
        print("TODO: qat")
    torch.quantization.convert(quant_model, inplace=True)
    print('quantization successfully')
    print('[1/3]: check model...')
    try:
        torch.jit.save(torch.jit.script(quant_model), save_path)
        torch.jit.load(save_path)
        print('check model successfully!')
    except Exception as e:
        print('check model failed!')
        print(e)
        return

    print('[2/3]: check accuracy...')
    inputs = [np.random.uniform(0, 1, tuple(shape)).astype(np.float32) for shape in input_shapes]
    outputs_fp = inference_torch(model, inputs)
    outputs_quant = inference_torch(quant_model, inputs)
    for of, oq in zip(outputs_fp, outputs_quant):
        try:
            np.testing.assert_allclose(of, oq, rtol=1e-3, atol=1e-5)
            print('check accuracy successfully!')
        except Exception as e:
            print('check accuracy failed!')
            print(e)
            return
    print('[3/3]: check latency')
    try:
        latency_fp = benchmark_torch(model, input_shapes)
        print('latency of fp: %.2f ms' % latency_fp)
        latency_quant = benchmark_torch(quant_model, input_shapes)
        print('latency of quant: %.2f ms' % latency_quant)
        print('check latency successfully!')
    except Exception as e:
        print('check latency failed!')
        print(e)
        return


def quant_onnx(model_path: str,
               save_path: str,
               mode: str = 'static',
               input_shapes: Optional[List] = None,
               data_reader: Optional[CalibrationDataReader] = None):
    if mode == 'dynamic':
        quantization.quantize_dynamic(model_path,
                                      save_path)
    elif mode == 'static:':
        quantization.quantize_static(model_path,
                                     save_path,
                                     calibration_data_reader=data_reader,
                                     quant_format=quantization.QuantFormat.QDQ,
                                     per_channel=True,
                                     weight_type=quantization.QuantType.QInt8)
    elif mode == 'qat':
        pass
    print('[1/3]: check model...')
    try:
        import onnx

        quant_model = onnx.load(save_path)
        onnx.checker.check_model(quant_model)
        print('check model successfully!')
    except Exception as e:
        print('check model failed!')
        print(e)
        return
    print('[2/3]: check accuracy...')
    inputs = [np.random.uniform(0, 1, tuple(shape)).astype(np.float32) for shape in input_shapes]
    outputs_fp = inference_onnx(model_path, inputs)
    outputs_quant = inference_onnx(save_path, inputs)
    for of, oq in zip(outputs_fp, outputs_quant):
        try:
            np.testing.assert_allclose(of, oq, rtol=1e-3, atol=1e-5)
            print('check accuracy successfully!')
        except Exception as e:
            print('check accuracy failed!')
            print(e)
            return
    print('[3/3]: check latency')
    try:
        latency_fp = benchmark_onnx(model_path)
        print('latency of fp: %.2f ms' % latency_fp)
        latency_quant = benchmark_onnx(save_path)
        print('latency of quant: %.2f ms' % latency_quant)
        print('check latency successfully!')
    except Exception as e:
        print('check latency failed!')
        print(e)
        return
