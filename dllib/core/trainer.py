#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/12/20 下午3:51  
@Author: Rocsky
@Project: dllib
@File: trainer.py
@Version: 0.1
@Description:
"""
import os
import time
from copy import deepcopy
from typing import List, Dict, Union, Optional
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from dllib.configs.global_config import cfg as gfg
from dllib.utils.acc_utils import AccRecords
from dllib.utils.checkpoint_utils import CheckpointUtils


class Trainer(object):
    def __init__(self,
                 train_params: Dict,
                 model_params: Dict,
                 optimizer_params: Dict,
                 dataset_params: Dict):
        """
        Init a basic trainer
        :param train_params: dict
        :param model_params: dict
        :param optimizer_params: dict
        :param dataset_params: dict
        """
        self.train_params = train_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.dataset_params = dataset_params

        self.use_cuda = self.train_params['use_cuda']
        self.gpu_list = self.train_params['gpu_list']
        if not self.gpu_list or torch.cuda.is_available() is False:
            self.use_cuda = False
        self.parallel = self.train_params['parallel']
        if not self.use_cuda:
            self.parallel = False
        self.distributed = self.train_params['distributed']
        if not self.parallel:
            self.distributed = False
        self.use_sync_batch = self.train_params['use_sync_batch']
        if not self.distributed:
            self.use_sync_batch = False
        self.use_amp = self.train_params['use_amp']

        self.resume = self.train_params['resume']
        self.loss_ratios = self.train_params['loss_ratios']
        self.loss_names = ['total'] + list(self.loss_ratios.keys())
        self.loss_acc = self.train_params['loss_acc']
        self.cache_dir = self.train_params['cache_dir']
        self.prefix = self.train_params['prefix']
        self.version = self.train_params['version']

        self.log_dir = os.path.join(self.cache_dir, self.prefix, self.version, 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        self.log = None

        self.checkpoint_dir = os.path.join(self.cache_dir, self.prefix, self.version, 'checkpoints')
        self.checkpoint = CheckpointUtils(self.checkpoint_dir)

        self.debug_dir = os.path.join(self.cache_dir, self.prefix, self.version, 'debugs')
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir, exist_ok=True)

        self.writer_dir = os.path.join(self.cache_dir, self.prefix, self.version, 'writers')
        if not os.path.exists(self.writer_dir):
            os.makedirs(self.writer_dir, exist_ok=True)
        self.writer = None

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_sampler = None
        self.val_sampler = None
        self.train_loader = None
        self.val_loader = None
        self.scaler = None
        self.start_epoch = 1
        self.start_step = 0
        self.val_losses = []

    def load_model(self):
        raise NotImplementedError

    def load_optimizer(self):
        raise NotImplementedError

    def load_scheduler(self):
        raise NotImplementedError

    def load_data_loader(self, rank: Union[int, str], world_size: Optional[int]):
        raise NotImplementedError

    def load_criterion(self):
        raise NotImplementedError

    def train_on_step(self, inputs: Dict, targets: Dict, rank: Union[int, str]):
        raise NotImplementedError

    def val_on_step(self, inputs: Dict, targets: Dict, rank: Union[int, str]):
        raise NotImplementedError

    def debug(self, epoch: int, step: int, inputs: Dict, targets: Dict, preds: List, losses: Dict, mode: str):
        raise NotImplementedError

    def train_on_epoch(self, epoch: int, rank: Union[int, str]):
        losses_acc = AccRecords(self.loss_names)
        if rank in ['cpu', 'cuda', 0]:
            self.log.info('Epoch: %d, Rank: %s, Lr: %.8f' % (epoch, rank, self.get_lr()))
        if self.distributed:
            self.train_sampler.set_epoch(epoch)
        train_len = len(self.train_loader)
        for step, (inputs, targets) in enumerate(self.train_loader, start=1):
            losses, preds = self.train_on_step(inputs, targets, rank)
            losses_acc.update_loss(losses)
            if rank in ['cpu', 'cuda', 0]:
                if step % self.train_params['log_interval'] == 0:
                    self.log.info('[%d/%d]%s' % (step, train_len, str(losses_acc)))
                    # print('[%d/%d]%s' % (step, train_len, str(losses_acc)))
                if self.train_params['debug_interval'] > 0 and step % self.train_params['debug_interval'] == 0:
                    self.debug(epoch, step, inputs, targets, preds, losses, 'train')

    def val_on_epoch(self, epoch: int, rank: Union[int, str]):
        self.model.eval()
        losses_acc = AccRecords(self.loss_names)
        with torch.no_grad():
            for step, (inputs, targets) in enumerate(self.val_loader, start=1):
                losses, preds = self.val_on_step(inputs, targets, rank)
                losses_acc.update_loss(losses)
                if rank in ['cpu', 'cuda', 0]:
                    if self.train_params['debug_interval'] > 0 and step % self.train_params['debug_interval'] == 0:
                        self.debug(epoch, step, inputs, targets, preds, losses, 'val')
        if rank in ['cpu', 'cuda', 0]:
            self.log.info('[Validation]%s' % str(losses_acc))
            val_loss = losses_acc.results()['total']
            self.val_losses.append(val_loss)
        self.model.train()

    def eval_on_epoch(self, epoch: int, rank: Union[int, str]):
        raise NotImplementedError

    def train(self):
        if self.use_cuda:
            if self.parallel and self.distributed:
                mp.spawn(self._train, args=(len(self.gpu_list),), nprocs=len(self.gpu_list), join=True)
            else:
                self._train('cuda')
        else:
            self._train('cpu')

    def _train(self, rank, world_size=Optional[int]):
        if isinstance(rank, int):
            self.setup(rank, world_size)
        print(torch.cuda.current_device())
        if rank in ['cpu', 'cuda', 0]:
            # log and writer can not be pickled across processes
            import logging

            timestamp = time.strftime('%Y%m%d-%H%M%S')
            logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                                level=logging.INFO,
                                filename=os.path.join(self.log_dir, 'log_%s_%s.log' % (self.prefix, timestamp)),
                                filemode='a')
            self.log = logging
            self.log.info(self.train_params)
            self.log.info(self.model_params)
            self.log.info(self.optimizer_params)
            self.log.info(self.dataset_params)
            self.writer = SummaryWriter(self.writer_dir)
        self.load_model()
        self.load_criterion()
        self.load_optimizer()
        self.load_scheduler()
        self.load_data_loader(rank, world_size)
        self.scaler = GradScaler() if self.use_amp else None
        if self.resume:
            weights, kwargs = self.checkpoint.load(self.resume)
            if kwargs['epoch'] and kwargs['step']:
                if kwargs['step'] == 0:
                    self.start_epoch = kwargs['epoch'] + 1
                else:
                    self.start_epoch = kwargs['epoch']
                self.start_step = kwargs['step']
            self.model.load_state_dict(weights['model_weights'], strict=True)
            self.optimizer.load_state_dict(weights['optim_weights'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(rank)
        self.model = self.model.to(rank)
        if rank == 'cuda' and len(self.gpu_list) > 1:
            self.model = nn.DataParallel(self.model, device_ids=list(range(len(self.gpu_list))))
        elif isinstance(rank, int):
            if self.use_sync_batch:
                self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DistributedDataParallel(self.model, device_ids=[rank])
        self.model.train()

        for epoch in range(self.start_epoch, self.train_params['end_epoch'] + 1):
            self.train_on_epoch(epoch, rank)

            if epoch % self.train_params['val_interval'] == 0:
                self.val_on_epoch(epoch, rank)
            if epoch % self.train_params['eval_interval'] == 0:
                self.eval_on_epoch(epoch, rank)

            if rank in ['cpu', 'cuda', 0]:
                if epoch % self.train_params['checkpoint_interval'] == 0:
                    if rank in ['cuda', 0] and len(self.gpu_list) > 1:
                        model_weights = deepcopy(self.model.module.state_dict())
                    else:
                        model_weights = deepcopy(self.model.state_dict())
                    optim_weights = deepcopy(self.optimizer.state_dict())
                    for k, v in model_weights.items():
                        if isinstance(v, torch.Tensor):
                            model_weights[k] = v.to('cpu')
                    for state in optim_weights['state'].values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to('cpu')
                    weights = {'model_weights': model_weights, 'optim_weights': optim_weights}
                    self.checkpoint.save(weights, **{'loss': self.val_losses[-1], 'epoch': epoch, 'step': 0})
            self.scheduler.step()
        if isinstance(rank, int):
            self.cleanup()

    @staticmethod
    def weight_init(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    @staticmethod
    def setup(rank: int, world_size: int):
        # os.environ['MASTER_ADDR'] = gfg.MASTER_ADDR
        # os.environ['MASTER_PORT'] = gfg.MASTER_PORT

        dist.init_process_group(gfg.MASTER_BACKEND, init_method='tcp://192.168.22.229:12345', rank=rank,
                                world_size=world_size)
        torch.cuda.set_device(rank)
        print(rank, torch.distributed.get_rank())
        print(torch.distributed.get_world_size())

    @staticmethod
    def cleanup():
        dist.destroy_process_group()

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self, lr: float):
        self.optimizer.param_groups[0]['lr'] = lr
