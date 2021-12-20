#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/6/16 下午5:25
@Author: Rocsky
@Project: ailib
@File: mobilenet_v2.py
@Version: 0.1
@Description:
"""
import torch
from functools import partial
from torch import nn
from torchvision.models.utils import load_state_dict_from_url

__all__ = ['MobileNetV2Backbone']

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


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


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

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

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


class MobileNetV2Backbone(nn.Module):
    def __init__(self,
                 in_channels=3,
                 layer_stride_list=[2, 2, 2, 2],
                 layer_return_list=[True, True, True, True],
                 return_layer0=True,
                 width_mult=1.0,
                 pretrained=True):
        """
        This is the backbone of mobilenet v2
        :param in_channels:
        :param layer_stride_list:
        :param layer_return_list:
        :param return_layer0:
        :param width_mult:
        :param pretrained:
        """
        super(MobileNetV2Backbone, self).__init__()
        self.layer_stride_list = layer_stride_list
        self.layer_return_list = layer_return_list
        self.return_layer0 = return_layer0
        assert len(self.layer_stride_list) == len(self.layer_return_list), \
            'layer_stride_list and layer_return_list should have same length'

        model = mobilenet_v2(pretrained, True, **{'width_mult': width_mult})
        if in_channels > 3:
            weight = model.state_dict()['features.0.0.weight']
            channels, _, kh, kw = weight.size()
            weight_expansion = torch.zeros(channels, in_channels, kh, kw)
            nn.init.kaiming_normal_(weight_expansion, mode='fan_out', nonlinearity='relu')
            weight_expansion[:, :3] = weight
            model.features[0][0].in_channels = in_channels
            model.features[0][0].weight = torch.nn.Parameter(weight_expansion)
        if width_mult == 1.0:
            self.enc_channels = [16, 24, 32, 96, 1280]
        elif width_mult == 0.5:
            self.enc_channels = [8, 16, 16, 48, 1280]
        del model.classifier
        self.stages = ['layer1', 'layer2', 'layer3', 'layer4']
        assert len(self.stages) == len(self.layer_return_list)
        self.return_features = dict(zip(self.stages, self.layer_return_list))
        self.stem = model.features[0:2]
        self.layer1 = model.features[2:4]
        self.layer2 = model.features[4:7]
        self.layer3 = model.features[7:14]
        self.layer4 = model.features[14:]
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
