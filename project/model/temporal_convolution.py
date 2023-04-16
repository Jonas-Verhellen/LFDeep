import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import pytorch_lightning as pl
torch.autograd.set_detect_anomaly(True)

class Chomp1d(pl.LightningModule):
    """Pad by (k-1)*d on the two sides of the input for convolution, and then use Chomp1d to remove the (k-1)*d elements on the right.
    This would essentially be the same as removing the "future elements", which ensures causality. We are shifting the output of ordinary conv1d
    by (k-2)/2, where k is the kernel size."""

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(pl.LightningModule):
    """ Here we want to perform a weight normalization so that we are capable of performing stochastic gradient descent with respect to our weight parameters.
        Each of these temporal blocks consists of one net with two convolutional layers and
        one seperate convolutional layer which is defined as the downsample. """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.sigmoid1 = nn.Sigmoid()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.sigmoid2 = nn.Sigmoid()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.sigmoid1, self.dropout1, self.conv2, self.chomp2, self.sigmoid2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        """Initialize the weights according to a normal distribution.
        Here we initialize the two convolutional layers and possibly a downsample convolutinal layer. """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        '''Here we pass through our network and compute our output. '''

        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x) # If the input does not have the same number of elements as output --> downsample.
        return self.sigmoid(out + res) # We add the input to the output.

class TemporalConvNet(pl.LightningModule):
    '''Here we create our full temporal convolutional neural network. (Here the config file will include all the initilaization parameters).'''
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
            in_channels = self.num_inputs if i == 0 else self.num_channels[i-1] # For first block we use input length.
            out_channels = self.num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, self.kernel_size, stride=1, dilation=dilation_size, padding=(self.kernel_size-1) * dilation_size, dropout=self.dropout)]
            # [TemporalBlock(1918, 32, 10, stride = 1, dilation =  1, padding= (10-1)*1), 0.2 ]
            # [TemporalBlock(32, 16, 10, stride = 1, dilation =  1, padding= (10-1)*1), 0.2 ]
            # [TemporalBlock(16, 8, 10, stride = 1, dilation =  1, padding= (10-1)*1), 0.2 ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        '''Pass through the whole temporal convolutional net and generate an output. '''
        out = self.network(x)
        out = torch.flatten(out, start_dim=1) # Here we flatten the output from the convolutional neural network. 
        return out
