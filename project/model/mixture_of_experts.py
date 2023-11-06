import hydra

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import torchmetrics
from omegaconf import OmegaConf


import numpy as np
from random import randint


class MH(pl.LightningModule):
    """
    
    Multi-task Hard-Parameter Sharing (MH) model.

    This class defines the Multi-task Hard-Parameter Sharing model, which uses a convolutional expert as the shared bottom for all tasks before data is sent into the task-specific towers.

    Args:
        config: Configuration parameters for the model.

    Attributes:
        num_tasks (int): Number of feedforward neural networks to process the data for different tasks.
        num_units (int): Number of neurons in the hidden layer of the towers.
        sequence_len (int): Length of the input sequence.
        num_features (int): Number of input features.
        optimizer (dict): Configuration for the optimizer used in training.
        shared_bottom (nn.Module): The shared convolutional expert.
        tcn_output_size (int): Output size of the shared bottom.
        input_list (nn.ModuleList): List of input layers for each task.
        towers_list (nn.ModuleList): List of tower layers for each task.
        output_list (nn.ModuleList): List of output layers for each task.
        loss_fn (nn.Module): Mean Squared Error loss function.
        loss_fn_spikes (nn.Module): Binary Cross-Entropy loss function for spikes.
        training_metric (torchmetrics.MeanSquaredError): Training metric for mean squared error.
        training_metric_spike (torchmetrics.Accuracy): Training metric for spike predictions.
        validation_metric (torchmetrics.MeanSquaredError): Validation metric for mean squared error.
        validation_metric_spike (torchmetrics.Accuracy): Validation metric for spike predictions.
        test_metric (torchmetrics.MeanSquaredError): Test metric for mean squared error.
        test_metric_spike (torchmetrics.Accuracy): Test metric for spike predictions.

    Methods:
        forward(inputs, diversity=False): Forward pass of the model to generate outputs for different tasks.
        calculating_shared_bottom(inputs): Calculate the shared bottom with ReLU activation.
        compute_element_errors(pred_out, true_out): Compute element-wise squared errors.
        training_step(batch, batch_idx): Training step for the model.
        training_step_end(outputs): Process and log training step outputs.
        validation_step(batch, batch_idx): Validation step for the model.
        validation_step_end(outputs): Process and log validation step outputs.
        test_step(batch, batch_idx): Test step for the model.
        test_step_end(outputs): Process and log test step outputs.
        configure_optimizers(): Configure the optimizer for training.
        on_train_start(): Log hyperparameters for training
    
    """
        
    def __init__(self, config):
        super(MH, self).__init__()
        self.save_hyperparameters()
        self.num_tasks = config.num_tasks
        self.num_units = config.num_units 
        self.sequence_len = config.sequence_len 
        self.num_features = config.num_features

        self.optimizer = OmegaConf.load(hydra.utils.to_absolute_path(config.optimizer))
        self.shared_bottom = hydra.utils.instantiate(OmegaConf.load(hydra.utils.to_absolute_path(config.expert)))
        self.tcn_output_size = self.shared_bottom.num_channels[-1]*self.sequence_len

        self.input_list = nn.ModuleList([nn.Linear(self.tcn_output_size, self.num_units) for _ in range(self.num_tasks)])
        self.towers_list = nn.ModuleList(nn.Linear(self.num_units, self.num_units) for _ in range(self.num_tasks))
        self.output_list = nn.ModuleList([nn.Linear(self.num_units, 1) for _ in range(self.num_tasks)])


        self.loss_fn = nn.MSELoss()
        self.loss_fn_spikes = nn.BCEWithLogitsLoss()

        self.training_metric =  torchmetrics.MeanSquaredError()
        self.training_metric_spike = torchmetrics.Accuracy()
        self.validation_metric = torchmetrics.MeanSquaredError()
        self.validation_metric_spike = torchmetrics.Accuracy()
        self.test_metric =  torchmetrics.MeanSquaredError()
        self.test_metric_spike = torchmetrics.Accuracy()

    def forward(self, inputs, diversity = False):
        """
        Forward pass of the Multi-task Hard-Parameter Sharing (MH) model. This method generates the model's output for different tasks based on the input data.

        Args:
            inputs (Tensor): Input data for the model.
            diversity (bool, optional): Flag indicating whether diversity is considered. Default is False.

        Returns:
            Tensor: Model output, combining the outputs for different tasks.
        """
        shared_bottom_outputs = self.calculating_shared_bottom(inputs)
        output = []
        for task in range(self.num_tasks):
            aux = self.input_list[task](shared_bottom_outputs)
            aux = F.elu(aux)
            aux = self.towers_list[task](aux)
            aux = F.elu(aux)
            aux = self.output_list[task](aux)
            output.append(aux) 
        output = torch.cat([x.float() for x in output], dim=1) 
        return output


    def calculating_shared_bottom(self, inputs):
        """
        Calculate the shared bottom with ReLU activation. This method processes the input data through the shared convolutional expert, applying ReLU activation to the output.

        Args:
            inputs (Tensor): Input data for the shared bottom.

        Returns:
            Tensor: Output of the shared bottom with ReLU activation applied.
        """
        aux = self.shared_bottom(inputs) 
        shared_bottom_outputs = F.relu(aux,inplace=False)
        return shared_bottom_outputs

    def compute_element_errors(self, pred_out, true_out):
        """
        Compute element-wise squared errors between predicted and true outputs. This method calculates the element-wise squared errors between the predicted model outputs and 
        the true target values.

        Args:
            pred_out (Tensor): Predicted model outputs.
            true_out (Tensor): True target values.

        Returns:
            Tensor: Element-wise squared errors.
        """
        return (pred_out-true_out)**2

    def training_step(self, batch, batch_idx):
        """
        Training step for the Multi-task Hard-Parameter Sharing (MH) model. This method processes a batch of training data and calculates the loss for the current batch.

        Args:
            batch (dict): A dictionary containing 'data' and 'target' tensors for the current batch.
            batch_idx (int): Index of the current batch.

        Returns:
            dict: A dictionary containing the loss, predicted model outputs, true target values, predicted spike values, and true spike values.
        """
        data, targets = batch['data'], batch['target']
        predictions = self(data)
        predictions_spike, targets_spike = predictions[:,639], targets[:,639,-1]
        predictions, targets = torch.cat([predictions[:,:639],predictions[:,640,None]],1), torch.cat([targets[:,:639,-1],targets[:,640,-1,None]],1)
        loss = self.loss_fn(predictions, targets) + self.loss_fn_spikes(predictions_spike,targets_spike)
        return {'loss': loss, 'predictions': predictions, 'targets': targets, 'predictions_spike': predictions_spike, 'targets_spike': targets_spike}

    def training_step_end(self, outputs):
        """
        Process and log the training step outputs. This method processes the outputs of the training step, updates training metrics, and logs relevant metrics.

        Args:
            outputs (dict): Outputs from the training step, including loss, predicted model outputs, true target values, predicted spike values, and true spike values.
        """
        self.training_metric(outputs['predictions'], outputs['targets'])
        self.training_metric_spike(F.softmax(outputs['predictions_spike']).int(),outputs['targets_spike'].int())
        self.log('loss/train', outputs['loss'])
        self.log('metric/train', self.training_metric)
        self.log('metric/train/spike', self.training_metric_spike)

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the Multi-task Hard-Parameter Sharing (MH) model. This method processes a batch of validation data and calculates the loss for the current batch.

        Args:
            batch (dict): A dictionary containing 'data' and 'target' tensors for the current batch.
            batch_idx (int): Index of the current batch.

        Returns:
            dict: A dictionary containing the loss, predicted model outputs, true target values, predicted spike values, and true spike values.
        """
        data, targets = batch['data'], batch['target']
        with torch.no_grad():
            predictions = self(data)
            predictions_spike, targets_spike = predictions[:,639], targets[:,639,-1]
            predictions, targets = torch.cat([predictions[:,:639],predictions[:,640,None]],1), torch.cat([targets[:,:639,-1],targets[:,640,-1,None]],1)
            loss = self.loss_fn(predictions, targets) + self.loss_fn_spikes(predictions_spike,targets_spike)
        return {'loss': loss, 'predictions': predictions, 'targets': targets, 'predictions_spike': predictions_spike, 'targets_spike': targets_spike}

    def validation_step_end(self, outputs):
        """
        Process and log the validation step outputs. This method processes the outputs of the validation step, including element-wise errors, updates validation metrics, and logs relevant information.

        Args:
            outputs (dict): Outputs from the validation step, including loss, predicted model outputs, true target values, predicted spike values, and true spike values.
        """
        element_errors = self.compute_element_errors(outputs['predictions'],outputs['targets'])
        self.validation_metric(outputs['predictions'], outputs['targets'])
        self.validation_metric_spike(F.softmax(outputs['predictions_spike']).int(),outputs['targets_spike'].int())
        self.log('loss/val', outputs['loss'])
        self.log('metric/val', self.validation_metric)
        self.log('metric/val/spike',self.validation_metric_spike)
        self.log('metric/val/element_errors',element_errors)

    def test_step(self, batch, batch_idx):
        """
        Test step for the Multi-task Hard-Parameter Sharing (MH) model. This method processes a batch of test data and calculates the loss for the current batch.

        Args:
            batch (dict): A dictionary containing 'data' and 'target' tensors for the current batch.
            batch_idx (int): Index of the current batch.

        Returns:
            dict: A dictionary containing the loss, predicted model outputs, true target values, predicted spike values, and true spike values.
        """
        data, targets = batch['data'], batch['target']
        with torch.no_grad():
            predictions = self(data)
            predictions_spike, targets_spike = predictions[:,639], targets[:,639,-1]
            predictions, targets = torch.cat([predictions[:,:639],predictions[:,640,None]],1), torch.cat([targets[:,:639,-1],targets[:,640,-1,None]],1)
            loss = self.loss_fn(predictions, targets) + self.loss_fn_spikes(predictions_spike,targets_spike)
        return {'loss': loss, 'predictions': predictions, 'targets': targets, 'predictions_spike': predictions_spike, 'targets_spike': targets_spike}


    def test_step_end(self, outputs):
        """
        Process and log the test step outputs. This method processes the outputs of the test step, updates test metrics, and logs relevant information.

        Args:
            outputs (dict): Outputs from the test step, including loss, predicted model outputs, true target values, predicted spike values, and true spike values.
        """
        self.test_metric(outputs['predictions'], outputs['targets'])
        self.validation_metric_spike(F.softmax(outputs['predictions_spike']).int(),outputs['targets_spike'].int())
        self.log('loss/test', outputs['loss'])
        self.log('metric/test', self.test_metric)
        self.log('metric/test/spike',self.test_metric_spike)

    def configure_optimizers(self):
        """
        Configure and instantiate the optimizer for training. This method is responsible for configuring and instantiating the optimizer to be used during training. 
        It uses the optimizer settings specified in the model's configuration.

        Returns:
            torch.optim.Optimizer: The instantiated optimizer.
        """
        return hydra.utils.instantiate(self.optimizer, self.parameters())

    def on_train_start(self):
        """
        Log hyperparameters and initial metric values at the start of training. This method is called at the beginning of the training process and is responsible for logging hyperparameters and initializing metric values for training, validation, and testing.
        """
        self.logger.log_hyperparams(self.hparams, {"metric/training": 0, "metric/test": 0, "metric/val": 0})

class MMoE(pl.LightningModule):
    """Multi-gate Mixture of Experts (MMoE) model.

    This class defines the Multi-gate Mixture of Experts model, which uses a convolutional networks as the experts which are used to generate data representations before these representations are sent into the task-specific towers, based on a data dependent, representation sharing mechanism.

    Args:
        config: A configuration object containing model hyperparameters.

    Attributes:
        num_tasks (int): The number of tasks the model is designed to perform.
        num_experts (int): The number of experts in the mixture.
        num_units (int): The number of hidden units in the model.
        sequence_len (int): The length of input sequences.
        num_features (int): The number of input features.
        use_expert_bias (bool): Flag indicating whether expert bias is used.
        use_gate_bias (bool): Flag indicating whether gate bias is used.
        optimizer: The optimizer used for model training.
        expert_kernels_tcn (nn.ModuleList): List of expert kernels for temporal convolution.
        tcn_output_size (int): The output size of the temporal convolution.
        compressor: Compressor module for input data.
        input_list (nn.ModuleList): List of input layers for each task.
        towers_list (nn.ModuleList): List of tower layers for each task.
        output_list (nn.ModuleList): List of output layers for each task.
        expert_bias (nn.Parameter): Expert bias parameters.
        gate_kernels (nn.Parameter): Gate kernels for task-specific gates.
        gate_bias (nn.Parameter): Gate bias parameters.
        task_bias (nn.Parameter): Task-specific bias parameters.
        loss_fn (nn.MSELoss): Mean Squared Error loss function.
        loss_fn_spikes (nn.BCEWithLogitsLoss): Binary Cross-Entropy loss function for spikes.
        training_metric (torchmetrics.MeanSquaredError): Metric for training.
        training_metric_spike (torchmetrics.Accuracy): Spike-related metric for training.
        validation_metric (torchmetrics.MeanSquaredError): Metric for validation.
        validation_metric_spike (torchmetrics.Accuracy): Spike-related metric for validation.
        test_metric (torchmetrics.MeanSquaredError): Metric for testing.
        test_metric_spike (torchmetrics.Accuracy): Spike-related metric for testing.

    Methods:
        forward(inputs, diversity=False): Forward pass of the MMoE model.
        calculating_experts(inputs): Calculate expert outputs.
        calculating_gates(inputs, batch_size): Calculate gate outputs.
        multiplying_gates_and_experts(expert_outputs, gate_outputs): Multiply gates and experts.
        compute_diversity(batch): Compute diversity metrics for a batch.
        compute_element_errors(pred_out, true_out): Compute element-wise errors.
        training_step(batch, batch_idx): Training step of the LightningModule.
        training_step_end(outputs): Post-training step operations for training.
        validation_step(batch, batch_idx): Validation step of the LightningModule.
        validation_step_end(outputs): Post-validation step operations for validation.
        test_step(batch, batch_idx): Test step of the LightningModule.
        test_step_end(outputs): Post-test step operations for testing.
        configure_optimizers(): Configure model optimizers.
        on_train_start(): Callback function when training starts.
    """
    def __init__(self, config):
        super(MMoE, self).__init__()
        self.save_hyperparameters()
        self.num_tasks = config.num_tasks
        self.num_experts = config.num_experts
        self.num_units = config.num_units

        self.sequence_len = config.sequence_len
        self.num_features =  config.num_features

        self.use_expert_bias = config.use_expert_bias
        self.use_gate_bias = config.use_gate_bias
        self.optimizer = OmegaConf.load(hydra.utils.to_absolute_path(config.optimizer))

        self.expert_kernels_tcn = nn.ModuleList([hydra.utils.instantiate(OmegaConf.load(hydra.utils.to_absolute_path(config.expert))) for _ in range(self.num_experts)])
        self.tcn_output_size = self.expert_kernels_tcn[0].num_channels[-1]*self.sequence_len
        self.compressor = hydra.utils.instantiate(OmegaConf.load(hydra.utils.to_absolute_path(config.expert)))

        self.input_list = nn.ModuleList([nn.Linear(self.tcn_output_size, self.num_units) for _ in range(self.num_tasks)])
        self.towers_list = nn.ModuleList(nn.Linear(self.num_units, self.num_units) for _ in range(self.num_tasks))
        self.output_list = nn.ModuleList([nn.Linear(self.num_units, 1) for _ in range(self.num_tasks)])

        if self.use_expert_bias:
            self.expert_bias = nn.Parameter(torch.zeros(self.num_experts, self.tcn_output_size), requires_grad=True)

        gate_kernels = torch.rand((self.num_tasks, self.tcn_output_size, self.num_experts)).float()
        self.gate_kernels = nn.Parameter(gate_kernels, requires_grad=True)

        if self.use_gate_bias:
            self.gate_bias = nn.Parameter(torch.zeros(self.num_tasks, 1, self.num_experts), requires_grad=True)

        self.task_bias = nn.Parameter(torch.zeros(self.num_tasks), requires_grad=True)

        self.loss_fn = nn.MSELoss()
        self.loss_fn_spikes = nn.BCEWithLogitsLoss()

        self.training_metric =  torchmetrics.MeanSquaredError()
        self.training_metric_spike = torchmetrics.Accuracy()
        self.validation_metric = torchmetrics.MeanSquaredError()
        self.validation_metric_spike = torchmetrics.Accuracy()
        self.test_metric =  torchmetrics.MeanSquaredError()
        self.test_metric_spike = torchmetrics.Accuracy()

    def forward(self, inputs, diversity=False):
        """
        Forward pass of the MMoE model.

        Args:
            inputs (torch.Tensor): Input data for the forward pass.
            diversity (bool, optional): Flag to indicate whether to compute diversity metrics.

        Returns:
            torch.Tensor or tuple: The model's output tensor or a tuple containing the output and expert outputs.
        """
        batch_size = inputs.shape[0]
        expert_outputs = self.calculating_experts(inputs)
        gate_outputs = self.calculating_gates(inputs, batch_size)
        product_outputs = self.multiplying_gates_and_experts(expert_outputs, gate_outputs)

        output = []
        for task in range(self.num_tasks):
            aux = self.input_list[task](product_outputs[task,:,:])
            aux = F.elu(aux)
            aux = self.towers_list[task](aux)
            aux = F.elu(aux)
            aux = self.output_list[task](aux)
            output.append(aux)

        output = torch.cat([x.float() for x in output], dim=1)
        if diversity:
            return output, expert_outputs
        else:
            return output

    def calculating_experts(self, inputs):
        """
        Calculate expert outputs based on the input data.

        Args:
            inputs (torch.Tensor): Input data for expert calculation.

        Returns:
            torch.Tensor: Expert outputs after processing the input data.
        """
        for i in range(self.num_experts):
            aux = self.expert_kernels_tcn[i](inputs)
            if i == 0:
                expert_outputs = aux.reshape(1, aux.shape[0], aux.shape[1])
            else:
                expert_outputs = torch.cat((expert_outputs, aux.reshape(1, aux.shape[0], aux.shape[1])), dim=0)

        if self.use_expert_bias:
            for expert in range(self.num_experts):
                    expert_bias = self.expert_bias[expert]
                    expert_outputs[expert] = expert_outputs[expert].add(expert_bias[None, :])
        expert_outputs = F.relu(expert_outputs,inplace=False)
        return expert_outputs

    def calculating_gates(self, inputs, batch_size):
        """
        Calculate the gating values for the experts. Calculate the gates, g^{k}(x) = activation(W_{gk} * x + b), where activation is softmax according to the original paper T x n x E

        Args:
            inputs (torch.Tensor): Input data for gate calculation.
            batch_size (int): The size of the input batch.

        Returns:
            torch.Tensor: Gating values (g^{k}(x)) for each expert.
        """
        compressed_inputs = self.compressor(inputs)
        for index in range(self.num_tasks):
            if index == 0:
                gate_outputs = torch.mm(compressed_inputs, self.gate_kernels[index]).reshape(1, batch_size, self.num_experts)
            else:
                gate_outputs = torch.cat((gate_outputs, torch.mm(compressed_inputs, self.gate_kernels[index]).reshape(1, batch_size, self.num_experts)), dim=0)

        if self.use_gate_bias:
            gate_outputs = gate_outputs.add(self.gate_bias)
        gate_outputs = F.softmax(gate_outputs, dim=2)
        return gate_outputs

    def multiplying_gates_and_experts(self, expert_outputs, gate_outputs):
        """
        Multiply gating values and expert outputs to produce task-specific results.

        Args:
            expert_outputs (torch.Tensor): Expert outputs for each expert.
            gate_outputs (torch.Tensor): Gating values for each expert.

        Returns:
            torch.Tensor: Task-specific results after multiplying gating and expert outputs.
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

    def compute_diversity(self, batch):
        """
        Compute diversity metrics for the given batch of data.

        Args:
            batch (torch.Tensor): Input data batch for diversity metric calculation.

        Returns:
            tuple: A tuple containing diversity score, determinant, and permanent.
        """
        import project.utils.diversity_metrics as dm
        batch = torch.reshape(batch,[batch.shape[0],batch.shape[1]*batch.shape[2]])
        diversity_matrix = dm.diversity_matrix(batch.T)
        diversity_score = torch.mean(diversity_matrix)
        diversity_determinant = torch.linalg.det(diversity_matrix)
        diversity_permanent = dm.permanent(diversity_matrix)
        return diversity_score, diversity_determinant, diversity_permanent

    def compute_element_errors(self, pred_out, true_out):
        """
        Compute element-wise errors between predicted and true outputs.

        Args:
            pred_out (torch.Tensor): Predicted output tensor.
            true_out (torch.Tensor): True output tensor.

        Returns:
            torch.Tensor: Element-wise errors between predicted and true outputs.
        """
        return torch.mean((pred_out-true_out)**2,0)

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step on the input batch.

        Args:
            batch (dict): A dictionary containing 'data' and 'target' keys with input and target tensors.
            batch_idx (int): Index of the current batch.

        Returns:
            dict: A dictionary with loss and various prediction-related information.
        """
        data, targets = batch['data'], batch['target']
        predictions = self(data)
        predictions_spike, targets_spike = predictions[:,639], targets[:,639,-1]
        predictions, targets = torch.cat([predictions[:,:639],predictions[:,640,None]],1), torch.cat([targets[:,:639,-1],targets[:,640,-1,None]],1)
        loss = self.loss_fn(predictions, targets) + self.loss_fn_spikes(predictions_spike,targets_spike)
        return {'loss': loss, 'predictions': predictions, 'targets': targets, 'predictions_spike': predictions_spike, 'targets_spike': targets_spike}

    def training_step_end(self, outputs):
        """
        Process and log the training step outputs. This method processes the outputs of the training step, updates training metrics, and logs relevant metrics.

        Args:
            outputs (dict): Outputs from the training step, including loss, predicted model outputs, true target values, predicted spike values, and true spike values.
        """
        self.training_metric(outputs['predictions'], outputs['targets'])
        self.training_metric_spike(F.softmax(outputs['predictions_spike']).int(),outputs['targets_spike'].int())
        self.log('loss/train', outputs['loss'])
        self.log('metric/train', self.training_metric)
        self.log('metric/train/spike', self.training_metric_spike)

    def validation_step(self, batch, batch_idx):
        """
        Process and log the validation step outputs. This method processes the outputs of the validation step, including element-wise errors, updates validation metrics, and logs relevant information.

        Args:
            outputs (dict): Outputs from the validation step, including loss, predicted model outputs, true target values, predicted spike values, and true spike values.
        """
        data, targets = batch['data'], batch['target']
        with torch.no_grad():
            predictions = self(data)
            predictions_spike, targets_spike = predictions[:,639], targets[:,639,-1]
            predictions, targets = torch.cat([predictions[:,:639],predictions[:,640,None]],1), torch.cat([targets[:,:639,-1],targets[:,640,-1,None]],1)
            loss = self.loss_fn(predictions, targets) + self.loss_fn_spikes(predictions_spike,targets_spike)
        return {'loss': loss, 'predictions': predictions, 'targets': targets, 'predictions_spike': predictions_spike, 'targets_spike': targets_spike, 'current_batch': data}

    def validation_step_end(self, outputs):
        expert_output = self.calculating_experts(outputs['current_batch'])
        element_errors = self.compute_element_errors(outputs['predictions'],outputs['targets'])
        diversity_score, diversity_determinant, diversity_permanent = self.compute_diversity(expert_output)
        self.validation_metric(outputs['predictions'], outputs['targets'])
        self.validation_metric_spike(F.softmax(outputs['predictions_spike']).int(),outputs['targets_spike'].int())
        self.log('loss/val', outputs['loss'])
        self.log('metric/val', self.validation_metric)
        self.log('metric/val/spike',self.validation_metric_spike)
        for i in range(len(element_errors)):
            self.log('metric/val/element_errors_'+str(i),element_errors[i])
        self.log('diversity/val/score', diversity_score)
        self.log('diversity/val/determinant',diversity_determinant)
        self.log('diversity/val/permanent',diversity_permanent)


    def test_step(self, batch, batch_idx):
        """
        Process and log the test step outputs. This method processes the outputs of the test step, updates test metrics, and logs relevant information.

        Args:
            outputs (dict): Outputs from the test step, including loss, predicted model outputs, true target values, predicted spike values, and true spike values.
        """
        data, targets = batch['data'], batch['target']

        with torch.no_grad():
            predictions = self(data)
            predictions_spike, targets_spike = predictions[:,639], targets[:,639,-1]
            predictions, targets = torch.cat([predictions[:,:639],predictions[:,640,None]],1), torch.cat([targets[:,:639,-1],targets[:,640,-1,None]],1)
            loss = self.loss_fn(predictions, targets) + self.loss_fn_spikes(predictions_spike,targets_spike)
        return {'loss': loss, 'predictions': predictions, 'targets': targets, 'predictions_spike': predictions_spike, 'targets_spike': targets_spike}


    def test_step_end(self, outputs):
        """
        Process and log the validation step outputs. This method processes the outputs of the validation step, including element-wise errors, updates validation metrics, and logs relevant information.

        Args:
            outputs (dict): Outputs from the validation step, including loss, predicted model outputs, true target values, predicted spike values, and true spike values.
        """
        self.test_metric(outputs['predictions'], outputs['targets'])
        self.test_metric_spike(F.softmax(outputs['predictions_spike']).int(),outputs['targets_spike'].int())
        self.log('loss/test', outputs['loss'])
        self.log('metric/test', self.test_metric)
        self.log('metric/test/spike',self.test_metric_spike)

    def configure_optimizers(self):
        """
        Configure and instantiate the optimizer for training. This method is responsible for configuring and instantiating the optimizer to be used during training. 
        It uses the optimizer settings specified in the model's configuration.

        Returns:
            torch.optim.Optimizer: The instantiated optimizer.
        """
        return hydra.utils.instantiate(self.optimizer, self.parameters())

    def on_train_start(self):
        """
        Log hyperparameters and initial metric values at the start of training. This method is called at the beginning of the training process and is responsible for logging hyperparameters and initializing metric values for training, validation, and testing.
        """
        self.logger.log_hyperparams(self.hparams, {"metric/training": 0, "metric/test": 0, "metric/val": 0})


class MMoEEx(MMoE):
    """ Multi-gate Mixture of Experts with Exclusivity(MMoEEx) model with additional features.

    Args:
        config: A configuration object containing model hyperparameters.

    Attributes:
        prob_exclusivity (float): The probability of introducing task exclusivity.
        type (str): The type of exclusivity, either "exclusivity" or "non-exclusivity".
        exclusivity (numpy.ndarray): Array representing expert-task exclusivity.
        gate_kernels (nn.Parameter): Task-specific gate kernels with potential exclusivity.
    """
    def __init__(self, config):
        super(MMoEEx, self).__init__(config)
        self.prob_exclusivity = config.prob_exclusivity
        self.type = config.type

        exclusivity = np.repeat(self.num_tasks + 1, self.num_experts)
        to_add = int(self.num_experts * self.prob_exclusivity)
        for e in range(to_add):
            exclusivity[e] = randint(0, self.num_tasks)

        self.exclusivity = exclusivity
        gate_kernels = torch.rand((self.num_tasks, self.tcn_output_size, self.num_experts)).float()
        
        for expert_number, task_number in enumerate(self.exclusivity):
            if task_number < self.num_tasks + 1:
                if self.type == "exclusivity":
                    for task in range(self.num_tasks):
                        if task != task_number:
                            gate_kernels[task][:, expert_number] = 0.0
                else:
                    gate_kernels[task_number][:, expert_number] = 0.0

        self.gate_kernels = nn.Parameter(gate_kernels, requires_grad=True)