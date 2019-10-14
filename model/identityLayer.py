import torch.nn as nn


class IdentityNet(nn.Module):
    def __init__(self):
        super(IdentityNet, self).__init__()
    
    def forward(self, *args):
        return args
