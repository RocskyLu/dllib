#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/6/16 下午5:25
@Author: Rocsky
@Project: dllib
@File: config.py
@Version: 0.1
@Description:
"""


class Config(object):
    def __setattr__(self, key, value):
        self.__dict__[key] = value


cfg = Config()
cfg.seed = 666
