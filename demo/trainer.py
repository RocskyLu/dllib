#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/12/21 下午3:46  
@Author: Rocsky
@Project: dllib
@File: trainer.py
@Version: 0.1
@Description:
"""
from typing import Dict, Union, Optional
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from dllib.core.trainer import Trainer
from dllib.optimizers import OPTIMIZER, SCHEDULER


class DemoModel(nn.Module):
    def __init__(self):
        super(DemoModel, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 1, 3, 1, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.block(x)


class DemoDataset(data.Dataset):
    def __init__(self, length: int):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = torch.rand(1, 3, 224, 224).type(torch.FloatTensor)
        pha = torch.ones(1, 1, 224, 224).type(torch.FloatTensor) * index
        return img, pha


class DemoTrainer(Trainer):
    def __init__(self,
                 train_params: Dict,
                 model_params: Dict,
                 optimizer_params: Dict,
                 dataset_params: Dict):
        super(DemoTrainer, self).__init__(train_params,
                                          model_params,
                                          optimizer_params,
                                          dataset_params)

    def load_model(self):
        self.model = DemoModel()

    def load_optimizer(self):
        optimizer_params = self.optimizer_params['optimizer']
        self.optimizer = OPTIMIZER[optimizer_params['arch']](self.model.parameters(), **optimizer_params['params'])

    def load_scheduler(self):
        scheduler_params = self.optimizer_params['scheduler']
        self.scheduler = SCHEDULER[scheduler_params['arch']](self.optimizer, **scheduler_params['params'])

    def load_data_loader(self, rank: Union[int, str], world_size: Optional[int]):
        self.train_dataset = DemoDataset(16)
        self.val_dataset = DemoDataset(16)
        self.train_sampler = DistributedSampler(self.train_dataset,
                                                num_replicas=world_size,
                                                rank=rank,
                                                shuffle=True,
                                                drop_last=True) if self.distributed else None
        self.val_sampler = DistributedSampler(self.val_dataset,
                                              num_replicas=world_size,
                                              rank=rank,
                                              shuffle=False,
                                              drop_last=False) if self.distributed else None
        train_batch = self.dataset_params['train_batch'] // len(self.gpu_list) if self.distributed else \
            self.dataset_params['train_batch']
        val_batch = self.dataset_params['val_batch'] // len(self.gpu_list) if self.distributed else \
            self.dataset_params['val_batch']
        self.train_loader = data.DataLoader(dataset=self.train_dataset,
                                            batch_size=train_batch,
                                            shuffle=self.train_sampler is None,
                                            sampler=self.train_sampler,
                                            collate_fn=self.data_collate,
                                            num_workers=self.dataset_params['num_workers'],
                                            pin_memory=True if self.use_cuda else False)

        self.val_loader = data.DataLoader(dataset=self.val_dataset,
                                          batch_size=val_batch,
                                          shuffle=False,
                                          sampler=self.val_sampler,
                                          collate_fn=self.data_collate,
                                          num_workers=self.dataset_params['num_workers'],
                                          pin_memory=True if self.use_cuda else False)

    def load_criterion(self):
        self.criterion = nn.L1Loss()

    def train_on_step(self, inputs: Dict, targets: Dict, rank: Union[int, str]):
        images = inputs['images'].type(torch.FloatTensor).to(rank)
        alphas = targets['alphas'].type(torch.FloatTensor).to(rank)
        losses = {}
        self.optimizer.zero_grad()
        with autocast(enabled=self.use_amp):
            preds = self.model(images)
        loss = self.criterion(preds, alphas)
        losses['l1'] = loss
        losses['total'] = loss
        if self.use_amp:
            self.scaler.scale(losses['total']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses['total'].backward()
            self.optimizer.step()
        return losses, preds

    def val_on_step(self, inputs: Dict, targets: Dict, rank: Union[int, str]):
        images = inputs['images'].type(torch.FloatTensor).to(rank)
        alphas = targets['alphas'].type(torch.FloatTensor).to(rank)
        losses = {}
        with autocast(enabled=self.use_amp):
            with torch.no_grad():
                preds = self.model(images)
        loss = self.criterion(preds, alphas)
        losses['l1'] = loss
        losses['total'] = loss
        return losses, preds

    def debug(self, epoch: int, step: int, inputs: Dict, targets: Dict, preds, losses, mode):
        print(mode, epoch, step)

    def eval_on_epoch(self, epoch: int, rank: Union[int, str]):
        pass

    @staticmethod
    def data_collate(batch):
        images = []
        alphas = []
        for item in batch:
            images.append(item[0])
            alphas.append(item[1])
        return {'images': torch.cat(images, dim=0)}, {'alphas': torch.cat(alphas, dim=0)}
