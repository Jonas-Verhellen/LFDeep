import sys
import hydra
import numpy as np
from numpy.random import randint

import torch
import torch.nn as nn
import torch.nn.functional as F
from temporal_convolution import TemporalConvNet
import pytorch_lightning as pl

  
sys.path.append("/src")

class MMoE(pl.LightningModule):
    def __init__(self, config):
        super(MMoE, self).__init__()
        self.save_hyperparameters()

        self.num_experts = config.num_experts
        self.num_tasks = config.num_tasks

        self.use_expert_bias = config.use_expert_bias
        self.use_gate_bias = config.use_gate_bias

        self.seqlen = config.seqlen
        self.num_features = config.num_features

        self.criterion = [nn.BCEWithLogitsLoss()] + [nn.MSELoss for _ in range(self.num_tasks -1)]
        self.expert_kernels = nn.ModuleList([TemporalConvNet() for i in range(self.num_experts)])

        gate_kernels = torch.rand((self.num_tasks, self.seqlen * self.num_features, self.num_experts)).float()
        self.expert_bias = nn.Parameter(torch.zeros(self.num_experts, self.seqlen), requires_grad=True)

        if self.use_expert_bias:
            self.expert_bias = nn.Parameter(torch.zeros(self.num_experts, 6400), requires_grad=True)
        if self.use_gate_bias:
            self.gate_bias = nn.Parameter(torch.zeros(self.num_tasks, 1, self.num_experts), requires_grad=True)

        self.gate_kernels = nn.Parameter(gate_kernels, requires_grad=True)
        self.task_bias = nn.Parameter(torch.zeros(self.num_tasks), requires_grad=True)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.towers_list = nn.ModuleList([nn.Linear(self.num_experts, self.num_units) for _ in range(self.num_tasks)])
        self.output_list = nn.ModuleList([nn.Linear(self.num_units, 1) for _ in range(self.num_tasks)])

    def forward(self, inputs, diversity=False):
        inputs = inputs[:, :, :self.seqlen]
        batch_size = inputs.shape[0]

        expert_outputs = self.calculating_experts(inputs) 
        gate_outputs = self.calculating_gates(inputs, batch_size)
        product_outputs = self.multiplying_gates_and_experts(expert_outputs, gate_outputs)

        final_outputs = []
        for task in range(self.num_tasks):
            aux = self.towers_list[task](product_outputs) 
            aux = self.output_list[task](aux)
            final_outputs.append(aux)

        if diversity:
            return final_outputs, expert_outputs  
        else:
            return final_outputs

    def calculating_experts(self, inputs):
        """ 
        Calculating the experts, i.e. f_{i}(x) = activation(W_{i} * x + b) where the activation function is ReLU according to the paper. 
        Shape: (E x n x U)
        """

        for i in range(self.num_experts):
            aux = self.expert_kernels[i](inputs)
            if i == 0:
                expert_outputs = aux.reshape(1, aux.shape[0], aux.shape[1])
            else:
                expert_outputs = torch.cat((expert_outputs, aux.reshape(1, aux.shape[0], aux.shape[1])), dim=0)

        if self.use_expert_bias:
            for expert in range(self.num_experts):
                    expert_bias = self.expert_bias[expert]
                    expert_outputs[expert] = expert_outputs[expert].add(expert_bias[None, :])

        expert_outputs = F.relu(expert_outputs)
        return expert_outputs

    def calculating_gates(self, inputs, batch_size):
        """ 
        Calculating the gates, i.e.  g^{k}(x) = activation(W_{gk} * x + b), where the activation function is softmax according to the paper. 
        Shape: (T x n x E)
        """
        inputs = inputs.reshape((inputs.shape[0], inputs.shape[1] * inputs.shape[2]))

        for index in range(self.num_tasks):
            if index == 0:
                gate_outputs = torch.mm(inputs, self.gate_kernels[index]).reshape(1, batch_size, self.num_experts)
            else:
                gate_outputs = torch.cat((gate_outputs, torch.mm(inputs, self.gate_kernels[index]).reshape(1, batch_size, self.num_experts)), dim=0)

        if self.use_gate_bias:
            gate_outputs = gate_outputs.add(self.gate_bias)

        gate_outputs = F.softmax(gate_outputs, dim=2)
        return gate_outputs

    def multiplying_gates_and_experts(self, expert_outputs, gate_outputs):
        """ 
        Multiplying gates and experts
        """

        for task in range(self.num_tasks):
            gate = gate_outputs[task]
            for expert in range(self.num_experts):
                gate_output = gate[:, expert]
                product = expert_outputs[expert] * gate_output[:, None]
                if expert == 0:
                    products = product
                else:
                    products = products.add(product)
            final_product = products.add(self.task_bias[task])

            if task == 0:
                final_products = final_product.reshape(1, final_product.shape[0], final_product.shape[1])
            else:
                final_products = torch.cat((final_products, final_product.reshape(1, final_product.shape[0], final_product.shape[1])), dim=0)
        return final_products
        
    def training_step(self, batch, batch_idx):
        loss = 0.0
        input, output = batch
        predicted_output = self(input)
        for task in range(self.num_tasks):
            loss_temp = (self.criterion[task](predicted_output[task],output[task]))
            loss += loss_temp
        self.log("loss/train", loss) # don't use this when in parrallel
        return loss

    def validation_step(self, batch, batch_idx):
        loss = 0.0
        input, output = batch
        predicted_output = self(input)
        for task in range(self.num_tasks):
            loss_temp = (self.criterion[task](predicted_output[task],output[task]))
            loss += loss_temp
        self.log("loss/val", loss)
        return loss 

    def test_step(self, batch, batch_idx):
        loss = 0.0
        input, output = batch
        predicted_output = self(input)
        for task in range(self.num_tasks):
            loss_temp = (self.criterion[task](predicted_output[task],output[task]))
            loss += loss_temp
        self.log("loss/test", loss)
        return loss 

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.hparams.optim, params=self.parameters())

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"loss/val": 0, "loss/val": 0, "loss/test": 0})

