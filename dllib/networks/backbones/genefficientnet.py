#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/6/16 下午5:25
@Author: Rocsky
@Project: dllib
@File: genefficientnet.py
@Version: 0.1
@Description:
"""
from functools import partial
import torch
import torch.nn as nn
import geffnet
from geffnet.gen_efficientnet import DepthwiseSeparableConv, InvertedResidual


class GenEfficientBackbone(nn.Module):
    def __init__(self,
                 in_channels=3,
                 layer_stride_list=[2, 2, 2, 2],
                 layer_return_list=[True, True, True, True],
                 return_layer0=True,
                 arch='efficientnet_lite0',
                 pretrained=True):
        """
        This is the backbone of geffnet models
        :param in_channels:
        :param layer_stride_list:
        :param layer_return_list:
        :param return_layer0:
        :param arch:
        :param pretrained:
        """
        super(GenEfficientBackbone, self).__init__()
        self.layer_stride_list = layer_stride_list
        self.layer_return_list = layer_return_list
        self.return_layer0 = return_layer0
        assert len(self.layer_stride_list) == len(self.layer_return_list), \
            'layer_stride_list and layer_return_list should have same length'

        model = geffnet.create_model(arch, pretrained=pretrained, no_jit=True)
        if in_channels > 3:
            weight = model.state_dict()['conv_stem.weight']
            channels, _, kh, kw = weight.size()
            weight_expansion = torch.zeros(channels, in_channels, kh, kw)
            nn.init.kaiming_normal_(weight_expansion, mode='fan_out', nonlinearity='relu')
            weight_expansion[:, :3] = weight
            model.conv_stem.in_channels = in_channels
            model.conv_stem.weight = torch.nn.Parameter(weight_expansion)
        if arch == 'efficientnet_lite0':
            self.enc_channels = [16, 24, 40, 160, 960]
        children_names = [name for name, _ in model.named_children()]
        if 'classifier' in children_names:
            del model.classifier
        if 'global_pool' in children_names:
            del model.global_pool
        self.stages = ['layer1', 'layer2', 'layer3', 'layer4']
        assert len(self.stages) == len(self.layer_return_list)
        self.return_features = dict(zip(self.stages, self.layer_return_list))
        stem = [model.conv_stem, model.bn1, model.act1, model.blocks[0]]
        self.stem = nn.Sequential(*stem)
        self.layer1 = model.blocks[1]
        self.layer2 = model.blocks[2]
        self.layer3 = model.blocks[3:5]
        self.layer4 = model.blocks[5:7]
        for stage_name, stride in zip(self.stages, self.layer_stride_list):
            if stride == 1:
                getattr(self, stage_name).apply(partial(self.replace_stride_dilate, dilation=2))
        self.enc_channels = []
        for layer in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            layer_last = layer[-1]
            if isinstance(layer_last, nn.Sequential):
                layer_last = layer_last[-1]
            if isinstance(layer_last, DepthwiseSeparableConv):
                self.enc_channels.append(layer_last.bn2.num_features)
            elif isinstance(layer_last, InvertedResidual):
                self.enc_channels.append(layer_last.bn3.num_features)

    @staticmethod
    def replace_stride_dilate(m: nn.Module, dilation):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and classname.find('ConvB') == -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilation // 2, dilation // 2)
                    m.padding = (dilation // 2, dilation // 2)
                elif m.kernel_size == (5, 5):
                    m.dilation = (dilation // 2, dilation // 2)
                    m.padding = (dilation, dilation)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilation, dilation)
                    m.padding = (dilation, dilation)
                elif m.kernel_size == (5, 5):
                    m.dilation = (dilation, dilation)
                    m.padding = (dilation * 2, dilation * 2)

    def forward(self, x):
        x = self.stem(x)
        feature_maps = []
        if self.return_layer0:
            feature_maps.append(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if self.return_features[stage_name]:
                feature_maps.append(x)
        return feature_maps
