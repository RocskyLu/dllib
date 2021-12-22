#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/12/21 下午1:47  
@Author: Rocsky
@Project: dllib
@File: __init__.py
@Version: 0.1
@Description:
"""
from .benchmark import benchmark_torch, benchmark_script, benchmark_onnx
from .inference import inference_torch, inference_script, inference_onnx
