#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/12/20 下午3:53  
@Author: Rocsky
@Project: dllib
@File: log_utils.py
@Version: 0.1
@Description:
"""
import os
import time
import logging

LEVEL = {
    'info': logging.INFO,
    'debug': logging.DEBUG,
    'error': logging.ERROR
}

FORMATTER = {
    'info': logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'),
    'debug': logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'),
    'error': logging.Formatter(''),
}


class LogUtils(object):
    def __init__(self, save_root: str = './', prefix: str = 'log', level: str = 'info'):
        """
        Init a custom logger
        :param save_root: the root of the saved log file
        :param prefix: the log file prefix
        :param level: the log level
        """
        if not os.path.exists(save_root):
            os.makedirs(save_root, exist_ok=True)
        self.save_root = save_root
        self.logger = logging.getLogger('_'.join([prefix, level]))
        self.logger.setLevel(LEVEL[level])
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        formatter = FORMATTER[level]
        if level == 'info':
            save_path = os.path.join(self.save_root, 'log_%s_%s.log' % (prefix, timestamp))
            file_handler = logging.FileHandler(save_path, mode='a', encoding='utf-8')
            file_handler.setLevel(LEVEL[level])
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            file_handler.close()
        else:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(LEVEL[level])
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)
            stream_handler.close()

    def get_logger(self) -> logging.Logger:
        return self.logger
