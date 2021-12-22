#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/12/20 下午4:19  
@Author: Rocsky
@Project: dllib
@File: checkpoint_utils.py
@Version: 0.1
@Description:
"""
import os
import datetime
from typing import Tuple, Dict
import torch


class CheckpointUtils(object):
    def __init__(self, save_root: str, keep: int = 5):
        if not os.path.exists(save_root):
            os.makedirs(save_root, exist_ok=True)
        self.save_root = save_root
        self.keep = keep
        self.cache = [None] * keep
        self.count = 0
        self.best_path = os.path.join(self.save_root, 'best.pth')

    def save(self, weights: Dict, **kwargs):
        save_path = os.path.join(self.save_root, 'epoch_%03d.pth' % kwargs['epoch'])
        torch.save(weights, save_path)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        with open(os.path.join(self.save_root, 'checkpoint.txt'), 'a') as f:
            f.writelines(['time:%s,epoch:%d,step:%d,loss:%f,path:%s\n' %
                          (timestamp, kwargs['epoch'], kwargs['step'], kwargs['loss'], save_path)])
        with open(os.path.join(self.save_root, 'latest.txt'), 'w') as f:
            f.writelines(['time:%s,epoch:%d,step:%d,loss:%f,path:%s\n' %
                          (timestamp, kwargs['epoch'], kwargs['step'], kwargs['loss'], save_path)])
        if os.path.isfile(os.path.join(self.save_root, 'best.txt')):
            with open(os.path.join(self.save_root, 'best.txt'), 'r') as f:
                record = f.readlines()[-1].strip()
            loss_opt = float(record.split(',')[3].split(':')[-1])
            if kwargs['loss'] < loss_opt:
                os.system('cp %s %s' % (save_path, self.best_path))
                with open(os.path.join(self.save_root, 'best.txt'), 'w') as f:
                    f.writelines(['time:%s,epoch:%d,step:%d,loss:%f,path:%s\n' %
                                  (timestamp, kwargs['epoch'], kwargs['step'], kwargs['loss'], self.best_path)])
        else:
            os.system('cp %s %s' % (save_path, self.best_path))
            with open(os.path.join(self.save_root, 'best.txt'), 'w') as f:
                f.writelines(['time:%s,epoch:%d,step:%d,loss:%f,path:%s\n' %
                              (timestamp, kwargs['epoch'], kwargs['step'], kwargs['loss'], self.best_path)])

        if self.cache[self.count % self.keep]:
            os.system('rm %s' % self.cache[self.count % self.keep])
        self.cache[self.count % self.keep] = save_path
        self.count += 1

    def load(self, resume: str) -> Tuple[Dict, Dict]:
        if resume == 'latest':
            with open(os.path.join(self.save_root, 'latest.txt'), 'r') as f:
                record = f.readlines()[-1].strip()
            load_path = record.split(',')[4].split(':')[-1]
            epoch = int(record.split(',')[1].split(':')[-1])
            step = int(record.split(',')[2].split(':')[-1])
        elif resume == 'best':
            with open(os.path.join(self.save_root, 'latest.txt'), 'r') as f:
                record = f.readlines()[-1].strip()
                load_path = record.split(',')[4].split(':')[-1]
                epoch = int(record.split(',')[1].split(':')[-1])
                step = int(record.split(',')[2].split(':')[-1])
        else:
            load_path = resume
            epoch = None
            step = None
        weights = torch.load(load_path)
        return weights, {'epoch': epoch, 'step': step}
