import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class SpikeDataset(Dataset):
    def __init__(self,in_path,target_path):
        self.spike_data = np.load(in_path,mmap_mode='r+')
        self.target_data = np.load(target_path, mmap_mode='r+')
        
    def __len__(self):
        return len(self.spike_data)
    
    def __getitem__(self, idx):
        data_sample = torch.from_numpy(self.spike_data[:,:,idx])
        target_sample = torch.from_numpy(self.target_data[:,:,idx])
        sample = {'data': data_sample, 'target': target_sample}
        
        return sample
    
    
class TestNet(nn.Module):
    def __init__(self,n_inputs,n_out):
        super(TestNet, self).__init__()
        self.layer = nn.Linear(n_inputs,n_out)
        
        
    def forward(self,x):
        out = self.layer(x.float().T)
        return out