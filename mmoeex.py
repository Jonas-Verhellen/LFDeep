import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import randint, binomial
import numpy as np
import sys

sys.path.append("/src")

class MMoETowers(nn.Module):
    def __init__(self, data, tasks_name, num_tasks, num_experts, num_units, num_features, modelname, task_info=None, task_number=None, runits=None, expert=None, expert_blocks=None, seqlen=None, n_layers=1, prob_exclusivity=0.5, type="exclusivity"):
        super(MMoETowers, self).__init__()
        self.data = data
        self.type = type
        self.modelname = modelname
        self.tasks_name = tasks_name
        self.num_tasks = num_tasks
        self.num_units = num_units
        self.num_experts = num_experts
        self.task_number = task_number
        self.task_info = task_info

        if modelname == "MMoE":
            self.MMoE = MMoE(data=data, units=num_units, num_experts=num_experts, num_tasks=num_tasks, num_features=num_features, use_expert_bias=True, use_gate_bias=True, expert_blocks=expert_blocks, n_layers=n_layers)
        elif modelname == "MMoEEx" or modelname == "Md":
            self.MMoEEx = MMoEEx(data=data, units=num_units, num_experts=num_experts, num_tasks=num_tasks, num_features=num_features, use_expert_bias=True, use_gate_bias=True, expert_blocks=expert_blocks, n_layers=n_layers, prob_exclusivity=prob_exclusivity, type=type)

        out = [1 for t in range(num_tasks)]
        inp = [self.num_units for t in range(num_tasks)]
        tower = self.num_experts

    def forward(self, input, params=None, diversity=False):
        input = input.float()
        if params is not None:
            for (_1, p), (_2, p_) in zip(params.items(), self.named_parameters()):
                p_.data = p.data

        if self.modelname == "MMoE" or self.modelname == "Mm":
            if diversity:
                x, div = self.MMoE(input, diversity=True)
            else:
                x = self.MMoE(input, diversity=False)  # T x N x E
        elif self.modelname == "MMoEEx" or self.modelname == "Md":
            if diversity:
                x, div = self.MMoEEx(input, diversity=True)
            else:
                x = self.MMoEEx(input, diversity=False)

        output = []
        for task in range(self.num_tasks):
            aux = self.mimic_fix_task_time(x[task], task)
            aux = self.towers_list[task](aux)  # n x seq x new_units
            if self.tasks_name[task] == "ihm":
                s0, s1, s2 = aux.shape
                aux = aux.reshape(s0, s1 * s2)
            elif self.tasks_name[task] == "pheno":
                s0, s1, s2 = aux.shape
                aux = aux.reshape(s0, s1 * s2)
            aux = self.output_list[task](aux)
            output.append(aux)

        if diversity:
            return div
        else:
            return output

class MMoEEx(nn.Module):
    def __init__(self, data, units, num_experts, num_tasks, num_features, use_expert_bias=True, use_gate_bias=True, expert=None, expert_blocks=None, n_layers=1, prob_exclusivity=0.5, type="exclusivity"):
        super(MMoEEx, self).__init__()
        self.data = data
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.num_features = num_features
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_blocks = expert_blocks
        self.n_layers = n_layers
        self.prob_exclusivity = prob_exclusivity
        self.type = type  # exclusivity or exclusion

        """Creating exclusivity or exclusion array"""
        exclusivity = np.repeat(self.num_tasks + 1, self.num_experts)
        to_add = int(self.num_experts * self.prob_exclusivity)
        for e in range(to_add):
            exclusivity[e] = randint(0, self.num_tasks)
        self.exclusivity = exclusivity

        print("Model - MMOEEx")
        print(self.type, ":", exclusivity)

        self.expert_kernels = nn.ModuleList([Expert_CNN() for i in range(self.num_experts)])
        self.expert_output = nn.ModuleList([nn.Linear().float() for i in range(self.num_experts)])

        """ Initialize gate weights (number of input features * number of experts * number of tasks)"""
        gate_kernels = torch.rand((self.num_tasks, self.num_features, self.num_experts)).float()

        """ Initialize expert bias (number of units per expert * number of experts)# Bias parameter"""
        if use_expert_bias:
            self.expert_bias = nn.Parameter(torch.zeros(self.num_experts), requires_grad=True)

        """ Initialize gate bias (number of experts * number of tasks)"""
        if use_gate_bias:
            self.gate_bias = nn.Parameter(torch.zeros(self.num_tasks, 1, self.num_experts), requires_grad=True)

        """
        Setting the exclusivity
         t: which task has the expert (all other should be set to 0)
         e: which expert is exclusive
        """

        for e, t in enumerate(self.exclusivity):
            if t < self.num_tasks + 1:
                if self.type == "exclusivity":
                    for tasks in range(self.num_tasks):
                        if tasks != t:
                            gate_kernels[tasks][:, e] = 0.0
                else:
                    gate_kernels[t][:, e] = 0.0

        self.gate_kernels = nn.Parameter(gate_kernels, requires_grad=True)
        self.task_bias = nn.Parameter(torch.zeros(self.num_tasks), requires_grad=True)
        self.dropout = nn.Dropout(0.25)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, diversity=False):
        n = inputs.shape[0]
        """ Calculating the experts """
        # f_{i}(x) = activation(W_{i} * x + b), where activation is ReLU according to the paper (E x n x U)
        print("...warning: new dataset might need to be updated here")
        for i in range(self.num_experts):
            aux = torch.mm(inputs, self.expert_kernels[i])
            aux = torch.reshape(aux, (n, self.expert_kernels[i].shape[1]))
            if i == 0:
                expert_outputs = self.expert_output[i](aux)
            else:
                expert_outputs = torch.cat((expert_outputs, self.expert_output[i](aux)), dim=1)

        if self.use_expert_bias:
            print("...warning: new dataset might need to be updated here")
            for i in range(self.num_experts):
                expert_outputs[i] = expert_outputs[i].add(self.expert_bias[i])
            expert_outputs = F.relu(expert_outputs)

        """ Calculating the gates"""
        # g^{k}(x) = activation(W_{gk} * x + b), where activation is softmax according to the paper T x n x E
        for index in range(self.num_tasks):
            if index == 0:
                gate_outputs = torch.mm(inputs, self.gate_kernels[index]).reshape(1, n, self.num_experts)
            else:
                gate_outputs = torch.cat((gate_outputs, torch.mm(inputs, self.gate_kernels[index]).reshape(1, n, self.num_experts),), dim=0)
        if self.use_gate_bias:
            gate_outputs = gate_outputs.add(self.gate_bias)
        gate_outputs = F.softmax(gate_outputs, dim=2)

        """ Multiplying gates and experts"""
        for task in range(self.num_tasks):
            gate = gate_outputs[task]
            final_outputs_t = torch.mul(gate, expert_outputs).reshape(1, gate.shape[0], gate.shape[1])
            final_outputs_t = final_outputs_t.add(self.task_bias[task])
            if task == 0:
                final_outputs = final_outputs_t
            else:
                final_outputs = torch.cat((final_outputs, final_outputs_t), dim=0)

        # T x n x E
        if diversity:
            return final_outputs, expert_outputs
        else:
            return final_outputs


class MMoE(nn.Module):
    """
    Multi-gate Mixture-of-Experts demo with census income data.
    Ma, Jiaqi, et al. "Modeling task relationships in multi-task learning with multi-gate mixture-of-experts."
    Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018.

    The code is based on the TensorFlow implementation:
    Reference: https://github.com/drawbridge/keras-mmoe
    """
  
    def __init__(self, data, units, num_experts, num_tasks, num_features, use_expert_bias=True, use_gate_bias=True, expert=None, expert_blocks=None, n_layers=1):
        super(MMoE, self).__init__()
        self.data = data
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.num_features = num_features
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert = expert
        self.n_layers = n_layers

        print("Model - MMOE")
        """Defining the experts: right now, all experts have the same architecture"""

        self.expert_kernels = nn.ModuleList([Expert_CNN() for i in range(self.num_experts)])
        self.expert_output = nn.ModuleList([nn.Linear().float() for i in range(self.num_experts)])

        """ Initialize gate weights (number of input features * number of experts * number of tasks)"""
        gate_kernels = torch.rand((self.num_tasks, self.num_features, self.num_experts)).float()

        """Initialize expert bias (number of units per expert * number of experts)# Bias parameter"""
        if use_expert_bias:
            if self.seqlen is None:
                self.expert_bias = nn.Parameter(torch.zeros(self.num_experts), requires_grad=True)
            else:
                self.expert_bias = nn.Parameter(torch.zeros(self.num_experts, self.seqlen), requires_grad=True)

        """ Initialize gate bias (number of experts * number of tasks)"""
        if use_gate_bias:
            self.gate_bias = nn.Parameter(torch.zeros(self.num_tasks, 1, self.num_experts), requires_grad=True)

        self.gate_kernels = nn.Parameter(gate_kernels, requires_grad=True)
        self.task_bias = nn.Parameter(torch.zeros(self.num_tasks), requires_grad=True)
        self.dropout = nn.Dropout(0.25)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, diversity=False):
        n = inputs.shape[0]
        """ Calculating the experts """
        # f_{i}(x) = activation(W_{i} * x + b), where activation is ReLU according to the paper (E x n x U)

        print("...warning: new dataset might need to be updated here")
        for i in range(self.num_experts):
            aux = torch.mm(inputs, self.expert_kernels[i])
            aux = torch.reshape(aux, (n, self.expert_kernels[i].shape[1]))
            if i == 0:
                expert_outputs = self.expert_output[i](aux)
            else:
                expert_outputs = torch.cat((expert_outputs, self.expert_output[i](aux)), dim=1)

        if self.use_expert_bias:
            for i in range(self.num_experts):
                expert_outputs[i] = expert_outputs[i].add(self.expert_bias[i])
            expert_outputs = F.relu(expert_outputs)

        """ Calculating the gates"""
        # g^{k}(x) = activation(W_{gk} * x + b), where activation is softmax according to the paper T x n x E
        for index in range(self.num_tasks):
            if index == 0:
                gate_outputs = torch.mm(inputs, self.gate_kernels[index]).reshape(1, n, self.num_experts)
            else:
                gate_outputs = torch.cat((gate_outputs, torch.mm(inputs, self.gate_kernels[index]).reshape(1, n, self.num_experts),),dim=0,)
        if self.use_gate_bias:
            gate_outputs = gate_outputs.add(self.gate_bias)
        gate_outputs = F.softmax(gate_outputs, dim=2)

        """ Multiplying gates and experts"""
        for task in range(self.num_tasks):
            gate = gate_outputs[task]
            final_outputs_t = torch.mul(gate, expert_outputs).reshape(
                1, gate.shape[0], gate.shape[1]
            )
            final_outputs_t = final_outputs_t.add(self.task_bias[task])
            if task == 0:
                final_outputs = final_outputs_t
            else:
                final_outputs = torch.cat((final_outputs, final_outputs_t), dim=0)

        # T x n x E
        if diversity:
            return final_outputs, expert_outputs
        else:
            return final_outputs


class Expert_CNN(nn.Module): 
    def __init__(self, input_window_size=400, num_segments=1278, num_syn_types=1, filter_sizes=[64,32,16], kernel_size=54, dilation=1, stride=1,activation_function=nn.ReLU()):
        super(Expert_CNN, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.input_channels = num_segments * num_syn_types
        self.activation_function = activation_function
        n_layers = len(filter_sizes)
        filter_sizes.insert(0,self.input_channels)
        layer_list = []
        for i in range(n_layers):
            layer_list.append(nn.Conv1d(filter_sizes[i],filter_sizes[i+1],kernel_size,padding=self.padding, dilation=dilation,
                                      stride=stride))
            layer_list.append(nn.ReLU())
            layer_list.append(nn.BatchNorm1d(filter_sizes[i+1]))            
        self.cnn = nn.Sequential(*layer_list)
        
    def forward(self,x):
        out = self.cnn(x) 
        return out
    
    
    