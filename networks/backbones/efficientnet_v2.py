#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/6/16 下午5:25
@Author: Rocsky
@Project: ailib
@File: efficientnet_v2.py
@Version: 0.1
@Description:
"""
import math
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn

__all__ = ['efficientnet_v2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, _make_divisible(inp // reduction, 8)),
            # nn.SiLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(inp // reduction, 8), oup),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        # nn.SiLU(inplace=True)
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        # nn.SiLU(inplace=True)
        nn.ReLU(inplace=True)
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # nn.SiLU(inplace=True),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # nn.SiLU(inplace=True),
                nn.ReLU(inplace=True),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # nn.SiLU(inplace=True),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(EfficientNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s, SE
            [1, 24, 2, 1, 0],
            [4, 48, 4, 2, 0],
            [4, 64, 4, 2, 0],
            [4, 128, 6, 2, 1],
            [6, 160, 9, 1, 1],
            [6, 272, 15, 2, 1],
        ]

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def efficientnet_v2(pretrained=True, **kwargs):
    """
    Constructs a EfficientNet V2 model
    """
    model = EfficientNetV2(**kwargs)
    if pretrained:
        state_dict = torch.load('./pretrained/efficientnet_v2.pth', map_location='cpu')
        state_dict = state_dict['state_dict']
        state_dict_clean = OrderedDict()
        for k, v in state_dict.items():
            state_dict_clean[k[7:]] = v
        model.load_state_dict(state_dict_clean)
        del state_dict
    return model


class EfficientNetV2Backbone(nn.Module):
    def __init__(self,
                 in_channels=3,
                 layer_stride_list=[2, 2, 2, 2],
                 layer_return_list=[True, True, True, True],
                 return_layer0=True,
                 width_mult=1.0,
                 pretrained=True):
        """
        This is the backbone of efficientnetv2
        :param in_channels:
        :param layer_stride_list:
        :param layer_return_list:
        :param return_layer0:
        :param width_mult:
        :param pretrained:
        """
        super(EfficientNetV2Backbone, self).__init__()
        self.layer_stride_list = layer_stride_list
        self.layer_return_list = layer_return_list
        self.return_layer0 = return_layer0
        assert len(self.layer_stride_list) == len(self.layer_return_list), \
            'layer_stride_list and layer_return_list should have same length'

        model = efficientnet_v2(pretrained, **{'width_mult': width_mult})
        if in_channels > 3:
            weight = model.state_dict()['features.0.0.weight']
            channels, _, kh, kw = weight.size()
            weight_expansion = torch.zeros(channels, in_channels, kh, kw)
            nn.init.kaiming_normal_(weight_expansion, mode='fan_out', nonlinearity='relu')
            weight_expansion[:, :3] = weight
            model.features[0][0].in_channels = in_channels
            model.features[0][0].weight = torch.nn.Parameter(weight_expansion)
        if width_mult == 1.0:
            self.enc_channels = [24, 48, 64, 160, 1792]
        elif width_mult == 0.5:
            self.enc_channels = [16, 24, 32, 80, 1792]
        del model.avgpool
        del model.classifier
        self.stages = ['layer1', 'layer2', 'layer3', 'layer4']
        assert len(self.stages) == len(self.layer_return_list)
        self.return_features = dict(zip(self.stages, self.layer_return_list))
        self.stem = model.features[0:3]
        self.layer1 = model.features[3:7]
        self.layer2 = model.features[7:11]
        self.layer3 = model.features[11:26]
        self.layer4 = nn.Sequential(*(model.features[26:]), model.conv)
        for stage_name, stride in zip(self.stages, self.layer_stride_list):
            if stride == 1:
                getattr(self, stage_name).apply(partial(self.replace_stride_dilate, dilation=2))

    @staticmethod
    def replace_stride_dilate(m: nn.Module, dilation):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and classname.find('ConvB') == -1 and classname.find('MBConv') == -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilation // 2, dilation // 2)
                    m.padding = (dilation // 2, dilation // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilation, dilation)
                    m.padding = (dilation, dilation)

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
