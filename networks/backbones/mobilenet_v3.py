#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/6/16 下午5:25
@Author: Rocsky
@Project: ailib
@File: mobilenet_v3.py
@Version: 0.1
@Description:
"""
'''MobileNetV3 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
from collections import OrderedDict
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

__all__ = ['MobileNetV3Backbone']


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.se = nn.Sequential(
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV3Large(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )

        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out


class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        )

        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(576, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out


def mobilenet_v3(pretrained=False, mode='small', **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        mode (str): Large or Small
    """
    if mode == 'large':
        model = MobileNetV3Large(**kwargs)
    else:
        model = MobileNetV3Small(**kwargs)
    if pretrained:
        if mode == 'large':
            state_dict = torch.load('./pretrained/mobilenet_v3_large.pth', map_location='cpu')
        else:
            state_dict = torch.load('./pretrained/mobilenet_v3_small.pth', map_location='cpu')
        state_dict = state_dict['state_dict']
        state_dict_clean = OrderedDict()
        for k, v in state_dict.items():
            state_dict_clean[k[7:]] = v
        model.load_state_dict(state_dict_clean)
        del state_dict
    return model


class MobileNetV3Backbone(nn.Module):
    def __init__(self,
                 in_channels=3,
                 layer_stride_list=[2, 2, 2, 2],
                 layer_return_list=[True, True, True, True],
                 return_layer0=True,
                 mode='small',
                 pretrained=True):
        """
        This is the backbone of mobilenet v3
        :param in_channels:
        :param layer_stride_list:
        :param layer_return_list:
        :param return_layer0:
        :param mode:
        :param pretrained:
        """
        super(MobileNetV3Backbone, self).__init__()
        self.layer_stride_list = layer_stride_list
        self.layer_return_list = layer_return_list
        self.return_layer0 = return_layer0
        assert len(self.layer_stride_list) == len(self.layer_return_list), \
            'layer_stride_list and layer_return_list should have same length'

        model = mobilenet_v3(pretrained, mode)
        if in_channels > 3:
            weight = model.state_dict()['features.0.0.weight']
            channels, _, kh, kw = weight.size()
            weight_expansion = torch.zeros(channels, in_channels, kh, kw)
            nn.init.kaiming_normal_(weight_expansion, mode='fan_out', nonlinearity='relu')
            weight_expansion[:, :3] = weight
            model.features[0][0].in_channels = in_channels
            model.features[0][0].weight = torch.nn.Parameter(weight_expansion)
        if mode == 'large':
            self.enc_channels = [16, 24, 40, 160, 960]
        else:
            self.enc_channels = [16, 16, 24, 48, 576]
        del model.linear3
        del model.bn3
        del model.hs3
        del model.linear4
        self.stages = ['layer1', 'layer2', 'layer3', 'layer4']
        assert len(self.stages) == len(self.layer_return_list)
        self.return_features = dict(zip(self.stages, self.layer_return_list))

        if mode == 'large':
            self.stem = nn.Sequential(model.conv1, model.bn1, model.hs1, model.bneck[0])
            self.layer1 = model.bneck[1:3]
            self.layer2 = model.bneck[3:6]
            self.layer3 = model.bneck[6:13]
            self.layer4 = nn.Sequential(model.bneck[13], model.bneck[14],
                                        model.conv2, model.bn2, model.hs2)
        else:
            self.stem = nn.Sequential(model.conv1, model.bn1, model.hs1)
            self.layer1 = model.bneck[:1]
            self.layer2 = model.bneck[1:3]
            self.layer3 = model.bneck[3:8]
            self.layer4 = nn.Sequential(model.bneck[8], model.bneck[9], model.bneck[10],
                                        model.conv2, model.bn2, model.hs2)
        for stage_name, stride in zip(self.stages, self.layer_stride_list):
            if stride == 1:
                getattr(self, stage_name).apply(partial(self.replace_stride_dilate, dilation=2))

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

