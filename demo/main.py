#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/12/21 下午3:47  
@Author: Rocsky
@Project: dllib
@File: main.py
@Version: 0.1
@Description:
"""
import os
import yaml
import argparse
from dllib.configs.config import cfg
from dllib.configs.global_config import cfg as gfg


def get_args():
    parser = argparse.ArgumentParser('Demo')
    parser.add_argument('--param_path', type=str, required=True,
                        help='Parameter file path')
    parser.add_argument('--load_weights', type=str, default='False',
                        help='Parameter file path')
    args = parser.parse_args()
    return args


def merge_config(param_file):
    with open(param_file, 'r') as f:
        config = yaml.safe_load(f)
    version = param_file.split('/')[-1][:-5]
    cfg.train_params = config.get('train_params')
    cfg.train_params['version'] = version
    cfg.model_params = config.get('model_params')
    cfg.optimizer_params = config.get('optimizer_params')
    cfg.dataset_params = config.get('train_dataset_params')
    if 'global_params' in cfg.train_params.keys():
        global_params = cfg.train_params['global_params']
        gfg.bn_momentum = global_params.get('bn_momentum', 0.1)
        gfg.bn_epsilon = global_params.get('bn_epsilon', 1e-5)
        gfg.align_corners = global_params.get('align_corners', False)


def train(options):
    from demo.trainer import DemoTrainer

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(options.train_params['gpu_list'])
    trainer = DemoTrainer(options.train_params,
                          options.model_params,
                          options.optimizer_params,
                          options.dataset_params)
    # frozen_layers = [trainer.model.backbone, trainer.model.x32_block]
    # # frozen_layers = [trainer.model.backbone]
    # for layer in frozen_layers:
    #     for name, value in layer.named_parameters():
    #         value.requires_grad = False
    # params = filter(lambda p: p.requires_grad, trainer.model.parameters())
    # try:
    #     trainer.train()
    # except KeyboardInterrupt:
    #     trainer.checkpoint(-1, **{'loss': float('inf'), 'step': 0})
    trainer.train()


if __name__ == '__main__':
    args = get_args()
    merge_config(args.param_path)
    if args.load_weights != 'False':
        cfg.train_params['resume'] = args.load_weights
    train(cfg)
