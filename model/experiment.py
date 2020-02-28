import torch.nn as nn
import torch

from torch.nn import Parameter
import torch.nn.init as init
import math
from util.logs import get_logger
from .inceptionv4 import InceptionV4
from .resnet import ResNet18, ResNet101
from torchvision.models.utils import load_state_dict_from_url
from . import MODEL_REGISTRY


logger = get_logger('experiment')


@MODEL_REGISTRY.register()
class ExperimentModel(nn.Module):

    def __init__(self, num_classes=10):
        super(ExperimentModel, self).__init__()
        # self.features = nn.Sequential(
        #     nn.Conv2d(1, 20, kernel_size=5, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(20, 50, kernel_size=5),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )

        ####################################
        # self.features = InceptionV4()
        # state_dict = load_state_dict_from_url(
        #     'http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth',
        #     model_dir='asserts',
        #     progress=True,
        # )
        # unused = self.features.load_state_dict(state_dict, strict=False)
        # logger.warn(unused)

        # self.f1 = nn.Linear(in_features=1536, out_features=num_classes, bias=True)


        ####################################
        self.features = ResNet18(pretrained=True)
        self.f1 = nn.Linear(in_features=512, out_features=num_classes, bias=True)

        self.quality = nn.Linear(512, 10, bias=True)
        self.clear = nn.Linear(512, 10, bias=True)
        self.artifact = nn.Linear(512, 10, bias=True)
        self.position = nn.Linear(512, 10, bias=True)


        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.features(x)['feature']
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        quality = self.quality(x)
        clear = self.clear(x)
        artifact = self.artifact(x)
        position = self.position(x)

        return dict(feature=x, quality=quality,clear=clear,artifact=artifact,position=position)


@MODEL_REGISTRY.register()
class ExperimentLoss(nn.Module):
    def __init__(self):
        super(ExperimentLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.beta =  0.5
        self.target_key = 'level'

    def forward(self, input, target):
        # pred = (input['out'].detach().argmax(dim=1) == target['noise_label']).to(torch.float32)
        # self.ce_loss(input['y_label'], target[self.target_key])
        l_quality = self.ce_loss(input['quality'], target['quality'])
        l_clear = self.ce_loss(input['clear'], target['clear'])
        l_artifact = self.ce_loss(input['artifact'], target['artifact'])
        l_position = self.ce_loss(input['position'], target['position'])
        # target_one_hot = torch.zeros_like(input['y_'])
        # # logger.info(target[self.target_key])
        # # logger.info(target_one_hot.shape)
        # target_one_hot.scatter_(1, target[self.target_key].reshape(-1, 1), 1)
        # # logger.info(target_one_hot)
        # l_qc = -(1 / input['y_'].size(0)) * ((1-self.beta) * target_one_hot + self.beta * input['y_']) * torch.log(input['y_'])
        # l_qc = l_qc.sum()

        loss = l_quality + l_clear + l_artifact + l_position
        # logger.info(l_nna)
        # logger.info(l_qc)
        return loss

# class Noise