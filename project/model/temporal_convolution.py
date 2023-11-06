import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import pytorch_lightning as pl
torch.autograd.set_detect_anomaly(True)

class Chomp1d(pl.LightningModule):
    """
    1D Convolutional Chomp Layer.

    Pad by (k-1)*d on the two sides of the input for convolution, and then use Chomp1d to remove the (k-1)*d elements on the right.
    This ensures causality by effectively removing "future elements" from the output of ordinary conv1d, shifting it by (k-2)/2.

    Args:
        chomp_size (int): The number of elements to remove from the right side of the input.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """Apply the Chomp1d operation to the input tensor. This method removes the right-most elements from the input tensor to perform Chomp1d, ensuring causality.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, num_channels, sequence_length).

        Returns:
            torch.Tensor: The input tensor with right-most elements removed.
        """
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(pl.LightningModule):
    """
    Temporal Block for Temporal Convolutional Network (TCN).

    A Temporal Block consists of two convolutional layers and one downsample convolutional layer.
    Weight normalization is applied to allow stochastic gradient descent with respect to weight parameters.
    
    Args:
        n_inputs (int): Number of input channels.
        n_outputs (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride for the convolution.
        dilation (int): Dilation factor for the convolution.
        padding (int): Padding size for the convolution.
        dropout (float): Dropout probability.

    Attributes:
        conv1 (nn.Module): Weight-normalized convolutional layer.
        chomp1 (Chomp1d): Chomp1d layer for removing right-padded elements.
        relu1 (nn.Module): Sigmoid activation function.
        dropout1 (nn.Module): Dropout layer.
        conv2 (nn.Module): Weight-normalized convolutional layer.
        chomp2 (Chomp1d): Chomp1d layer for removing right-padded elements.
        relu2 (nn.Module): Sigmoid activation function.
        dropout2 (nn.Module): Dropout layer.
        net (nn.Sequential): Sequential network with convolutional layers, chomping, activations, and dropout.
        downsample (nn.Module): Downsample convolutional layer (if n_inputs != n_outputs).
        relu (nn.Module): Sigmoid activation function.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.Sigmoid()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.Sigmoid()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of convolutional layers according to a normal distribution. This method initializes the weights of the two convolutional layers and, if applicable, the downsample convolutional layer.
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        Forward pass through the TemporalBlock. This method processes the input tensor through the network and computes the output.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, num_channels, sequence_length).

        Returns:
            torch.Tensor: The output tensor after processing.
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x) #
        return self.relu(out + res) 
    
class TemporalConvNet(pl.LightningModule):
    """
    Temporal Convolutional Neural Network (TCN) model.

    This class creates a full temporal convolutional neural network. The configuration file should include all the initialization parameters.

    Args:
        config: A configuration object containing model hyperparameters.

    Attributes:
        num_inputs (int): Number of input channels.
        num_channels (list): List of channel sizes for each layer.
        kernel_size (int): Size of the convolutional kernel.
        dropout (float): Dropout probability.
        network (nn.Sequential): Sequential network of temporal blocks.
    """
    def __init__(self, config):
        super(TemporalConvNet, self).__init__()
        self.num_inputs = config.num_inputs
        self.num_channels = config.num_channels
        self.kernel_size=config.kernel_size
        self.dropout=config.dropout
        layers = []
        num_levels = len(self.num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = self.num_inputs if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, self.kernel_size, stride=1, dilation=dilation_size, padding=(self.kernel_size-1) * dilation_size, dropout=self.dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the Temporal Convolutional Network (TCN) to generate an output. This method passes the input tensor through the entire TCN and generates an output. It also flattens the output
        from the convolutional neural network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, num_inputs, sequence_length).

        Returns:
            torch.Tensor: The flattened output tensor.
        """
        out = self.network(x)
        out = torch.flatten(out, start_dim=1)  
        return out
