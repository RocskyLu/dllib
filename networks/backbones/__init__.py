#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/12/20 下午3:22  
@Author: Rocsky
@Project: dllib
@File: __init__.py
@Version: 0.1
@Description:
"""
from .resnet import ResNetBackbone
from .mobilenet_v1 import MobileNetV1Backbone
from .mobilenet_v2 import MobileNetV2Backbone
from .mobilenet_v3 import MobileNetV3Backbone
from .efficientnet import EfficientNetBackbone
from .genefficientnet import GenEfficientBackbone

BACKBONES = {
    "resnet": ResNetBackbone,
    "mobilenet_v1": MobileNetV1Backbone,
    "mobilenet_v2": MobileNetV2Backbone,
    "mobilenet_v3": MobileNetV3Backbone,
    "efficientnet": EfficientNetBackbone,
    "genefficientnet": GenEfficientBackbone
}