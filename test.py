from torch.utils.data import DataLoader
from src.SpikeDataloader import SpikeDataset
import mmoeex

dat = SpikeDataset('/dcip/LFDeepData/input_spikes.npy','/dcip/LFDeepData/output.npy')
loader = DataLoader(dat,batch_size=10,shuffle=True)#,num_workers=1,persistent_workers=True)

a = dat.__getitem__(3)

model = mmoeex.MMoE(3, 5, 1278)
    
dat_batch = next(iter(loader))

outs = model(dat_batch['data'])
print(outs.size())
