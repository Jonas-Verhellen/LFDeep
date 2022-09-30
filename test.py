from torch.utils.data import DataLoader
from src.SpikeDataloader import SpikeDataset, TestNet
import mmoeex

dat = SpikeDataset('/Users/constb/Data/LFDeep_Data/input_spikes.npy','/Users/constb/Data/LFDeep_Data/output.npy')
loader = DataLoader(dat,batch_size=2,shuffle=True)#,num_workers=1,persistent_workers=True)

a = dat.__getitem__(3)
model = mmoeex.Expert_CNN()
    
dat_batch = next(iter(loader))

outs = model(dat_batch['data'])

