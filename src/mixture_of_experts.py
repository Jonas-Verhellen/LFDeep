import sys
import numpy as np
from numpy.random import randint

import torch
import torch.nn as nn
import torch.nn.functional as F

from temporal_convolution import TemporalConvNet

sys.path.append("/src")

class MixExperts(nn.Module):
    def __init__(self, data, tasks_name, num_tasks, num_experts, num_units, num_features, modelname, task_info=None, task_number=None,  n_layers=1, prob_exclusivity=0.5, type="exclusivity"):
        super(MixExperts, self).__init__()
        self.data = data
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.modelname = modelname
        self.tasks_name = tasks_name
        self.num_units = num_units
        self.type = type
        self.task_number = task_number
        self.task_info = task_info

        if modelname == "MMoE":
            self.MMoE = MMoE(data=data, units=num_units, num_experts=num_experts, num_tasks=num_tasks, num_features=num_features, use_expert_bias=True, use_gate_bias=True, n_layers=n_layers)
        elif modelname == "MMoEEx" or modelname == "Md":
            self.MMoEEx = MMoEEx(data=data, units=num_units, num_experts=num_experts, num_tasks=num_tasks, num_features=num_features, use_expert_bias=True, use_gate_bias=True, n_layers=n_layers, prob_exclusivity=prob_exclusivity, type=type)

        out = [1 for _ in range(num_tasks)]
        inp = [self.num_units for _ in range(num_tasks)]

        tower = self.num_experts

        self.towers_list = nn.ModuleList([nn.Linear(tower, self.num_units) for i in range(self.num_tasks)])
        self.output_list = nn.ModuleList([nn.Linear(inp[i], out[i]) for i in range(self.num_tasks)])

    def forward(self, input, params=None, diversity=False):
        input = input.float()

        if params is not None:
            for (_1, p), (_2, p_) in zip(params.items(), self.named_parameters()):
                p_.data = p.data

        if self.modelname == "MMoE" or self.modelname == "Mm":
            if diversity:
                x, div = self.MMoE(input, diversity=True)
            else:
                x = self.MMoE(input, diversity=False)  
        elif self.modelname == "MMoEEx" or self.modelname == "Md":
            if diversity:
                x, div = self.MMoEEx(input, diversity=True)
            else:
                x = self.MMoEEx(input, diversity=False)

        output = []
        for task in range(self.num_tasks):
            aux = self.towers_list[task](x) 
            aux = self.output_list[task](aux)
            output.append(aux)

        if diversity:
            return output, diversity  
        else:
            return output

class MMoE(nn.Module):
    def __init__(self, num_experts, num_tasks, num_features, use_expert_bias=True, use_gate_bias=True, n_layers=1, seqlen=400):
        super(MMoE, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.num_features = num_features
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.n_layers = n_layers
        self.seqlen = seqlen
        print("... Model - MMOE")

        self.expert_kernels = nn.ModuleList([TemporalConvNet() for i in range(self.num_experts)])

        """ Initialize gate weights (number of input features * number of experts * number of tasks)"""
        gate_kernels = torch.rand((self.num_tasks, self.seqlen * self.num_features, self.num_experts)).float()
        self.expert_bias = nn.Parameter(torch.zeros(self.num_experts, self.seqlen), requires_grad=True)

        """Initialize expert bias (number of units per expert * number of experts)"""
        if use_expert_bias:
            self.expert_bias = nn.Parameter(torch.zeros(self.num_experts, 6400), requires_grad=True)

        """ Initialize gate bias (number of experts * number of tasks)"""
        if use_gate_bias:
            self.gate_bias = nn.Parameter(torch.zeros(self.num_tasks, 1, self.num_experts), requires_grad=True)

        self.gate_kernels = nn.Parameter(gate_kernels, requires_grad=True)
        self.task_bias = nn.Parameter(torch.zeros(self.num_tasks), requires_grad=True)
        self.dropout = nn.Dropout(0.25)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, diversity=False):
        batch_size = inputs.shape[0]
        inputs = inputs[:, :, :self.seqlen]

        """ Calculating the experts """
        # f_{i}(x) = activation(W_{i} * x + b), where activation is ReLU according to the paper (E x n x U)

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

        """ Calculating the gates"""
        # g^{k}(x) = activation(W_{gk} * x + b), where activation is softmax according to the paper T x n x E

        inputs = inputs.reshape((inputs.shape[0], inputs.shape[1] * inputs.shape[2]))

        for index in range(self.num_tasks):
            if index == 0:
                gate_outputs = torch.mm(inputs, self.gate_kernels[index]).reshape(1, batch_size, self.num_experts)
            else:
                gate_outputs = torch.cat((gate_outputs, torch.mm(inputs, self.gate_kernels[index]).reshape(1, batch_size, self.num_experts)), dim=0)

        if self.use_gate_bias:
            gate_outputs = gate_outputs.add(self.gate_bias)

        gate_outputs = F.softmax(gate_outputs, dim=2)

        """ Multiplying gates and experts"""

        for task in range(self.num_tasks):
            gate = gate_outputs[task]
            for expert in range(self.num_experts):
                gate_output = gate[:, expert]
                product = expert_outputs[expert] * gate_output[:, None]
                if expert == 0:
                    products = product
                else:
                    products = products.add(product)
            final_output = products.add(self.task_bias[task])

            if task == 0:
                final_outputs = final_output.reshape(1, final_output.shape[0], final_output.shape[1])
            else:
                final_outputs = torch.cat((final_outputs, final_output.reshape(1, final_output.shape[0], final_output.shape[1])), dim=0)

        # T x n x E
        if diversity:
            return final_outputs, expert_outputs
        else:
            return final_outputs

class MMoEEx(nn.Module):
    def __init__(self, data, units, num_experts, num_tasks, num_features, use_expert_bias=True, use_gate_bias=True, seqlen=None, n_layers=1, prob_exclusivity=0.5, type="exclusivity"):
        super(MMoEEx, self).__init__()
        self.data = data
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.num_features = num_features
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.seqlen = seqlen
        self.n_layers = n_layers
        self.prob_exclusivity = prob_exclusivity
        self.type = type  # exclusivity or exclusion

        """Creating exclusivity or exclusion array"""
        exclusivity = np.repeat(self.num_tasks + 1, self.num_experts)
        to_add = int(self.num_experts * self.prob_exclusivity)
        for e in range(to_add):
            exclusivity[e] = randint(0, self.num_tasks)
        self.exclusivity = exclusivity

        print("... Model - MMOEEx")
        print("... ", self.type, ":", exclusivity)

        self.expert_kernels = nn.ModuleList([TemporalConvNet() for i in range(self.num_experts)])

        """ Initialize gate weights (number of input features * number of experts * number of tasks)"""
        gate_kernels = torch.rand((self.num_tasks, self.seqlen * self.num_features, self.num_experts)).float()
        self.expert_bias = nn.Parameter(torch.zeros(self.num_experts, self.seqlen), requires_grad=True)

        """Initialize expert bias (number of units per expert * number of experts)"""
        if use_expert_bias:
            self.expert_bias = nn.Parameter(torch.zeros(self.num_experts, 6400), requires_grad=True)

        """ Initialize gate bias (number of experts * number of tasks)"""
        if use_gate_bias:
            self.gate_bias = nn.Parameter(torch.zeros(self.num_tasks, 1, self.num_experts), requires_grad=True)

        """
        Setting the exclusivity
         task_number: which task has the expert (all other should be set to 0)
         expert_number: which expert is exclusive
        """
        for expert_number, task_number in enumerate(self.exclusivity):
            if task_number < self.num_tasks + 1:
                if self.type == "exclusivity":
                    for task in range(self.num_tasks):
                        if task != task_number:
                            gate_kernels[task][:, expert_number] = 0.0
                else:
                    gate_kernels[task_number][:, expert_number] = 0.0


        self.gate_kernels = nn.Parameter(gate_kernels, requires_grad=True)
        self.task_bias = nn.Parameter(torch.zeros(self.num_tasks), requires_grad=True)
        self.dropout = nn.Dropout(0.25)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, diversity=False):
        batch_size = inputs.shape[0]
        inputs = inputs[:, :, :self.seqlen]

        """ Calculating the experts """
        # f_{i}(x) = activation(W_{i} * x + b), where activation is ReLU according to the paper (E x n x U)

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

        """ Calculating the gates"""
        # g^{k}(x) = activation(W_{gk} * x + b), where activation is softmax according to the paper T x n x E

        inputs = inputs.reshape((inputs.shape[0], inputs.shape[1] * inputs.shape[2]))

        for index in range(self.num_tasks):
            if index == 0:
                gate_outputs = torch.mm(inputs, self.gate_kernels[index]).reshape(1, batch_size, self.num_experts)
            else:
                gate_outputs = torch.cat((gate_outputs, torch.mm(inputs, self.gate_kernels[index]).reshape(1, batch_size, self.num_experts)), dim=0)

        if self.use_gate_bias:
            gate_outputs = gate_outputs.add(self.gate_bias)

        gate_outputs = F.softmax(gate_outputs, dim=2)

        """ Multiplying gates and experts"""

        for task in range(self.num_tasks):
            gate = gate_outputs[task]
            for expert in range(self.num_experts):
                gate_output = gate[:, expert]
                product = expert_outputs[expert] * gate_output[:, None]
                if expert == 0:
                    products = product
                else:
                    products = products.add(product)
            final_output = products.add(self.task_bias[task])

            if task == 0:
                final_outputs = final_output.reshape(1, final_output.shape[0], final_output.shape[1])
            else:
                final_outputs = torch.cat((final_outputs, final_output.reshape(1, final_output.shape[0], final_output.shape[1])), dim=0)

        # T x n x E
        if diversity:
            return final_outputs, expert_outputs
        else:
            return final_outputs
