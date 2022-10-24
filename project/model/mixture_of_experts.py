import hydra

import torch
import torch.nn as nn
import torch.nn.functional as F

from temporal_convolution import TemporalConvNet
import pytorch_lightning as pl

class MMoE(pl.LightningModule):
    def __init__(self, config):
        super(MMoE, self).__init__()
        self.num_tasks = config.num_tasks
        self.num_experts = config.num_experts
        self.num_layers = config.num_layers
        self.num_units = config.num_units

        self.sequence_len = config.sequence_len
        self.num_features =  config.num_features
        self.output_features = config.expert.num_channels[-1]

        self.use_expert_bias = config.use_expert_bias
        self.use_gate_bias = config.use_gate_bias

        self.towers_list = nn.ModuleList([nn.Linear(self.sequence_len*self.num_features, self.num_units) for _ in range(self.num_tasks)])
        self.output_list = nn.ModuleList([nn.Linear(self.num_units, 1) for _ in range(self.num_tasks)])

        self.expert_kernels = nn.ModuleList([hydra.utils.instantiate(config.expert) for _ in range(self.num_experts)])
        gate_kernels = torch.rand((self.num_tasks, self.sequence_len * self.num_features, self.num_experts)).float()
        self.expert_bias = nn.Parameter(torch.zeros(self.num_experts, self.sequence_len), requires_grad=True)

        if self.use_expert_bias:
            self.expert_bias = nn.Parameter(torch.zeros(self.num_experts, self.output_features*self.sequence_len), requires_grad=True)

        if self.use_gate_bias:
            self.gate_bias = nn.Parameter(torch.zeros(self.num_tasks, 1, self.num_experts), requires_grad=True)

        self.gate_kernels = nn.Parameter(gate_kernels, requires_grad=True)
        self.task_bias = nn.Parameter(torch.zeros(self.num_tasks), requires_grad=True)

        self.train_metric = [pl.metrics.Accuracy(), pl.metrics.MeanSquaredError()]
        self.val_metric = [pl.metrics.Accuracy(), pl.metrics.MeanSquaredError()]
        self.test_metric = [pl.metrics.Accuracy(), pl.metrics.MeanSquaredError()]

    def forward(self, inputs, diversity=False):
        batch_size = inputs.shape[0]
        inputs = inputs[:, :, :self.sequence_len] #TODO: to be removed!!
        expert_outputs = self.calculating_experts(inputs)
        gate_outputs = self.calculating_gates(inputs, batch_size)
        product_outputs = self.multiplying_gates_and_experts(expert_outputs, gate_outputs)

        output = []
        for task in range(self.num_tasks):
            aux = self.towers_list[task](product_outputs) 
            aux = self.output_list[task](aux)
            output.append(aux)

        if diversity:
            return output, expert_outputs  
        else:
            return output

    def calculating_experts(self, inputs):
        """ 
        Calculating the experts, f_{i}(x) = activation(W_{i} * x + b), where activation is ReLU according to the paper (E x n x U) 
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
        Calculating the gates, g^{k}(x) = activation(W_{gk} * x + b), where activation is softmax according to the paper T x n x E
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
        data, targets = batch
        predictions = self(data)
        loss = nn.BCELoss(predictions[0], targets[0]) + [nn.MSELoss(prediction, target) for prediction, target in zip(predictions[1,:], targets[1,:])]
        return {'loss': loss, 'predictions': predictions, 'targets': targets}

    def training_step_end(self, outputs):
        self.training_metric[0](outputs['predictions'][0], outputs['target'][0])
        self.training_metric[1](outputs['predictions'][1], outputs['target'][1])
        self.log('loss/train', outputs['loss'])
        self.log('metric/train', self.training_metric)

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        predictions = self(data)
        loss = nn.BCELoss(predictions[0], targets[0]) + [nn.MSELoss(prediction, target) for prediction, target in zip(predictions[1,:], targets[1,:])]
        return {'loss': loss, 'predictions': predictions, 'targets': targets}

    def validation_epoch_end(self, outputs):
        self.validation_metric[0](outputs['predictions'][0], outputs['target'][0])
        self.validation_metric[1](outputs['predictions'][1], outputs['target'][1])
        self.log('loss/val', outputs['loss'])
        self.log('metric/val', self.training_metric)

    def test_step(self, batch, batch_idx):
        data, targets = batch
        predictions = self(data)
        loss = nn.BCELoss(predictions[0], targets[0]) + [nn.MSELoss(prediction, target) for prediction, target in zip(predictions[1,:], targets[1,:])]
        return {'loss': loss, 'predictions': predictions, 'targets': targets}

    def test_epoch_end(self, outputs):
        self.test_metric[0](outputs['predictions'][0], outputs['target'][0])
        self.test_metric[1](outputs['predictions'][1], outputs['target'][1])
        self.log('loss/test', outputs['loss'])
        self.log('metric/test', self.test_metric)

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.hparams.optim, params=self.parameters())

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"metric/training": [0], "metric/test": [0], "metric/val": [0]})


""" 
class MMoEEx(MMoE):
    def __init__(self, config):
        super(MMoEEx, self).__init__()
        self.prob_exclusivity = config.prob_exclusivity
        self.type = config.type 

        exclusivity = np.repeat(self.num_tasks + 1, self.num_experts)
        to_add = int(self.num_experts * self.prob_exclusivity)
        for e in range(to_add):
            exclusivity[e] = randint(0, self.num_tasks)

        self.exclusivity = exclusivity
        gate_kernels = torch.rand((self.num_tasks, self.seqlen * self.num_features, self.num_experts)).float()

        for expert_number, task_number in enumerate(self.exclusivity):
            if task_number < self.num_tasks + 1:
                if self.type == "exclusivity":
                    for task in range(self.num_tasks):
                        if task != task_number:
                            gate_kernels[task][:, expert_number] = 0.0
                else:
                    gate_kernels[task_number][:, expert_number] = 0.0

        self.gate_kernels = nn.Parameter(gate_kernels, requires_grad=True) 
"""