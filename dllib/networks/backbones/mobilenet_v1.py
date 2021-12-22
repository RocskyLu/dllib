#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/6/16 下午5:25
@Author: Rocsky
@Project: dllib
@File: mobilenet_v1.py
@Version: 0.1
@Description:
"""
from functools import partial
from torch import nn
from torchvision.models.utils import load_state_dict_from_url

__all__ = ['MobileNetV1Backbone']


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


class ConvBlock(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBlock, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class SepConvBlock(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1):
        padding = (kernel_size - 1) // 2 * dilation
        super(SepConvBlock, self).__init__(
            nn.Conv2d(in_planes, in_planes, kernel_size, stride, padding=padding, groups=in_planes, bias=False,
                      dilation=dilation),
            nn.BatchNorm2d(in_planes),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class MobileNetV1Backbone(nn.Module):
    def __init__(self,
                 in_channels=3,
                 layer_stride_list=[2, 2, 2, 2],
                 layer_return_list=[True, True, True, True],
                 return_layer0=True,
                 width_mult=1.0,
                 pretrained=False):
        """
        This is backbone of mobilenet v1
        :param in_channels:
        :param layer_stride_list:
        :param layer_return_list:
        :param return_layer0:
        :param width_mult:
        :param pretrained:
        """
        super(MobileNetV1Backbone, self).__init__()
        input_channel = 32
        round_nearest = 8
        self.layer_stride_list = layer_stride_list
        self.layer_return_list = layer_return_list
        self.return_layer0 = return_layer0
        assert len(self.layer_stride_list) == len(self.layer_return_list), \
            'layer_stride_list and layer_return_list should have same length'
        # if in_channels > 3:
        #     weight = model.state_dict()['stem.0.0.weight']
        #     channels, _, kh, kw = weight.size()
        #     weight_expansion = torch.zeros(channels, in_channels, kh, kw)
        #     nn.init.kaiming_normal_(weight_expansion, mode='fan_out', nonlinearity='relu')
        #     weight_expansion[:, :3] = weight
        #     model.features[0][0].in_channels = in_channels
        #     model.features[0][0].weight = torch.nn.Parameter(weight_expansion)
        if width_mult == 1.0:
            self.enc_channels = [64, 128, 256, 512, 1024]
        elif width_mult == 0.5:
            self.enc_channels = [32, 64, 128, 256, 512]
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.stem = nn.Sequential(
            ConvBlock(in_channels, input_channel, kernel_size=3, stride=2),
            SepConvBlock(input_channel, input_channel * 2, kernel_size=3, stride=1)
        )

        self.layer1 = nn.Sequential(
            SepConvBlock(input_channel * 2, input_channel * 4, kernel_size=3, stride=self.layer_stride_list[0]),
            SepConvBlock(input_channel * 4, input_channel * 4, kernel_size=3, stride=1)
        )

        self.layer2 = nn.Sequential(
            SepConvBlock(input_channel * 4, input_channel * 8, kernel_size=3, stride=self.layer_stride_list[1]),
            SepConvBlock(input_channel * 8, input_channel * 8, kernel_size=3, stride=1)
        )

        self.layer3 = nn.Sequential(
            SepConvBlock(input_channel * 8, input_channel * 16, kernel_size=3, stride=self.layer_stride_list[2]),
            SepConvBlock(input_channel * 16, input_channel * 16, kernel_size=3, stride=1, dilation=1),
            SepConvBlock(input_channel * 16, input_channel * 16, kernel_size=3, stride=1, dilation=2),
            SepConvBlock(input_channel * 16, input_channel * 16, kernel_size=3, stride=1, dilation=3),
            SepConvBlock(input_channel * 16, input_channel * 16, kernel_size=3, stride=1, dilation=5),
            SepConvBlock(input_channel * 16, input_channel * 16, kernel_size=3, stride=1, dilation=7)
        )
        self.layer4 = nn.Sequential(
            SepConvBlock(input_channel * 16, input_channel * 32, kernel_size=3, stride=self.layer_stride_list[3]),
            SepConvBlock(input_channel * 32, input_channel * 32, kernel_size=3, stride=1, dilation=2)
        )

        self.stages = ['layer1', 'layer2', 'layer3', 'layer4']
        assert len(self.stages) == len(self.layer_return_list)
        self.return_features = dict(zip(self.stages, self.layer_return_list))

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

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
