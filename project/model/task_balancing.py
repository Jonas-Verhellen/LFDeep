"""
Copyright (c) 2020-present, Royal Bank of Canada.
All rights reserved.
 This source code is licensed under the license found in the
 LICENSE file in the root directory of this source tree.
Task balacing approaches for multi-task learning
Two methods based on loss ratio called:
- DWA - Dynamic Weight Average
- LBTW - Loss Balanced Task Weighting
Written by Gabriel Oliveira in pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import randint, binomial
import numpy as np
import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig, OmegaConf

"""Comment: This is in numpy so I might have to change it to torch format"""

class TaskBalanceMTL:
    # Class for task balancing methods
    def __init__(self, config):
        # Hyper parameters
        self.balance_method = config.balance_method
        self.K = config.n_tasks
        self.T = config.n_tasks
        self.alpha_balance = config.alpha_balance
        self.n_tasks = config.n_tasks
        self.task_ratios = torch.zeros([self.n_tasks])
        self.task_weights = torch.zeros([self.n_tasks])
        self.initial_losses = torch.zeros([self.n_tasks])
        self.weight_history = []
        self.history_last = []
        for i in range(self.n_tasks):
            self.weight_history.append([])
            self.history_last.append([])

        # Setting weight method
        self.balance_mode = config.balance
        if self.balance_mode == "DWA":
            print("...DWA Weight balance")
        if self.balance_mode == "LBTW":
            print("...LBTW Weight balance")

    def add_loss_history(self, task_losses):
        for i in range(0, self.n_tasks):
            self.weight_history[i].append(task_losses[i])

    def last_elements_history(self):
        for i in range(0, self.n_tasks):
            self.history_last[i] = self.weight_history[i][-2:]

    def compute_ratios(self, task_losses, epoch):

        for i in range(0, self.n_tasks):
            if epoch <= 1:
                self.task_ratios[:] = 1
            else:
                before = "-"
                if self.history_last[i][-2] > -0.01 and self.history_last[i][-2] < 0.01:
                    before = self.history_last[i][-2]
                    self.history_last[i][-2] = 0.01

                self.task_ratios[i] = (
                    self.history_last[i][-1] / self.history_last[i][-2]
                )

    def sum_losses_tasks(self):
        ratios_sum = 0.0
        for i in range(0, self.n_tasks):
            ratios_sum += torch.exp(self.task_ratios[i] / self.T)
        return ratios_sum

    def DWA(self, task_losses, epoch):
        self.compute_ratios(task_losses, epoch)
        ratios_sum = self.sum_losses_tasks()

        for i in range(0, self.n_tasks):
            self.task_weights[i] = max(
                min((self.K * torch.exp(self.task_ratios[i] / self.T)) / ratios_sum, 1.5),
                0.5,
            )

    def get_weights(self):
        #task_weights = task_weights.to(device="cuda")
        return self.task_weights

    def get_initial_loss(self, losses, task):
        self.initial_losses[task] = losses

    def LBTW(self, batch_losses, task):
        self.task_weights[task] = torch.fmax(
            torch.fmin(pow(batch_losses / self.initial_losses[task], torch.Tensor([self.alpha_balance])), torch.Tensor([1.0])),
            torch.Tensor([0.01]),
        )
        
        #if (batch_losses / self.initial_losses[task]) == float('NaN'):
        #    self.task_weights[task] = 1.0
        #    print("Found it!")
        #else:
        #    self.task_weights[task] = max(
        #    min(pow(batch_losses / self.initial_losses[task], self.alpha_balance), 1.0),
        #    0.01,
        #)
        
