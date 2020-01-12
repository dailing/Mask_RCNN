import torch.nn as nn
import torch

from torch.nn import Parameter
import torch.nn.init as init
import math


class ExperimentModel(nn.Module):

    def __init__(self, num_classes=1000):
        super(ExperimentModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.randmask = Parameter((torch.rand(100, 1250) > 0.5).to(torch.float32), requires_grad=False)
        self.classification = nn.Sequential(
            nn.Linear(1250, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        self.noise_detection = nn.Sequential(
            nn.Linear(1250, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classification(x)
        noise_score = self.noise_detection(x)
        rand_pred = [self.classification(x * self.randmask[i, :] * 2) for i in range(self.randmask.size(0))]
        rand_pred = torch.stack(rand_pred, dim=1)
        return dict(out=out, rand_pred=rand_pred, feature = x)


class Input

# class Noise