import torch
from torch.nn.functional import mse_loss

x = torch.rand(size=(1000, 8))
y = torch.rand(size=(8, ))
print(x.device, y.device)
print(mse_loss(y, x))           # this works fine

x = x.cuda()
y = y.cuda()
print(x.device, y.device)
print(mse_loss(y, x))           # error
