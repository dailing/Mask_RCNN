import torch.nn as nn
import torch

from torch.nn import Parameter
import torch.nn.init as init
import math
from util.logs import get_logger


logger = get_logger('experiment')


class ExperimentModel(nn.Module):

    def __init__(self, num_classes=10):
        super(ExperimentModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.f1 = nn.Linear(in_features=800, out_features=num_classes, bias=True)
        self.theta_t1 = nn.Softmax(dim=1)
        self.l1 = nn.Linear(num_classes, num_classes, bias=False)
        self.theta_t2 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        t1 = self.f1(x)
        y_hat = self.theta_t1(t1)
        t2 = self.l1(y_hat)
        y_hat_slash = self.theta_t2(t2)

        return dict(feature=x, y_=y_hat, y_label=y_hat_slash)


class ExperimentLoss(nn.Module):
    def __init__(self):
        super(ExperimentLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.beta =  0.5

    def forward(self, input, target):
        # pred = (input['out'].detach().argmax(dim=1) == target['noise_label']).to(torch.float32)
        l_nna = self.ce_loss(input['y_label'], target['label'])
        target_one_hot = torch.zeros_like(input['y_'])
        # logger.info(target['label'])
        # logger.info(target_one_hot.shape)
        target_one_hot.scatter_(1, target['label'].reshape(-1, 1), 1)
        # logger.info(target_one_hot)
        l_qc = -(1 / input['y_'].size(0)) * ((1-self.beta) * target_one_hot + self.beta * input['y_']) * torch.log(input['y_'])
        l_qc = l_qc.sum()
        loss =  l_qc + l_nna
        # logger.info(l_nna)
        # logger.info(l_qc)
        return loss

# class Noise