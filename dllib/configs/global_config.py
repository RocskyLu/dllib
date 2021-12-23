#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/6/16 下午5:25
@Author: Rocsky
@Project: dllib
@File: global_config.py
@Version: 0.1
@Description:
"""
from .config import Config

cfg = Config()

# ddp params
cfg.MASTER_ADDR = 'localhost'
cfg.MASTER_PORT = '12345'
cfg.MASTER_BACKEND = 'nccl'

# network params
cfg.bn_momentum = 0.1
cfg.bn_epsilon = 1e-5
cfg.align_corners = False
