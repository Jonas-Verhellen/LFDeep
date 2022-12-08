import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
data = torch.randn([128, 2, 1, 100], dtype=torch.float, device='cuda', requires_grad=True)
net = torch.nn.Conv2d(2, 32, kernel_size=[1, 54], padding=[0, 53], stride=[1, 1], dilation=[1, 1], groups=1)
net = net.cuda().float()
out = net(data)
out.backward(torch.randn_like(out))
torch.cuda.synchronize()
