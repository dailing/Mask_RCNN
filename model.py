import torch.nn as nn
import numpy as np
import math
from voc2012 import VOC
from torch.utils.data import Dataset, DataLoader
import torch
from itertools import product


class VGG(nn.Module):
    def __init__(self, init_weights=True):
        super(VGG, self).__init__()
        self.features = self.make_layers(
            [64, 64, 'M',  # 3, 5, 6
             128, 128, 'M',  # 10, 14 16
             256, 256, 256, 'M',  # 24, 32, 40, 44
             512, 512, 512, 'M',  # 60, 76, 92, 100
             512, 512, 512, 'M'],  # 132, 164, 196, 212
            batch_norm=True)

        self.rpn_sliding_window = nn.Conv2d(
            512, 256, 3, 1, 1
        )
        self.box_classification = nn.Conv2d(256, 2 * 9, 1)
        self.box_regression = nn.Conv2d(256, 4 * 9, 1)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        rpn_feature = self.rpn_sliding_window(x)
        box_predict = self.box_classification(rpn_feature)
        box_regression = self.box_regression(rpn_feature)
        return box_predict, box_regression

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    @staticmethod
    def make_layers(cfg, batch_norm=False):
        layers = []
        in_channels = 3
        dilation = 1
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'D':
                dilation = 2
            else:
                conv2d = nn.Conv2d(
                    in_channels, v,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


def get_intersection(c1, w1, c2, w2):
    if abs(c1 - c2) >= (w1 + w2) / 2:
        return 0
    elif abs(c1 - c2) <= (w1 - w2) / 2:
        return w2
    elif abs(c1 - c2) <= (w2 - w1) / 2:
        return w1
    else:
        return (w1 + w2) / 2 - abs(c1 - c2)


def mean_iou(box1, box2):
    intersection = get_intersection(box1[0], box1[2], box2[0], box2[2]) * \
                   get_intersection(box1[1], box1[3], box2[1], box2[3])
    union = box1[2] * box1[3] + box2[2] * box2[3] - intersection
    return intersection / union


class BBloader(Dataset):
    def __init__(self, split, name, archers=None):
        self.split = split
        self.name = name
        if archers is None:
            archers = [
                (64, 64),
                (64, 32),
                (32, 64),
                (128, 128),
                (128, 64),
                (64, 128),
                (256, 256),
                (256, 128),
                (128, 256),
            ]
        self.n_archer = len(archers)
        self.archers = archers
        if name == 'voc':
            self.data = VOC(split=split)
        self.ratio = 32

    def __getitem__(self, index):
        img, bbox = self.data[index]
        img = img.astype(np.float)
        img = img.transpose((2, 0, 1)) / 255
        img = img.astype(np.float32)
        nchannel, nrow, ncol = img.shape

        arow, acol = nrow // self.ratio, ncol // self.ratio
        archor_cls = np.zeros((self.n_archer, 2, arow, acol), np.float)
        archor_reg = np.zeros((self.n_archer, 4, arow, acol), np.float)
        train_mask = np.zeros((self.n_archer, 1, arow, acol), np.float)

        center_rows = [self.ratio // 2 + i * self.ratio for i in range(arow)]
        center_cols = [self.ratio // 2 + i * self.ratio for i in range(acol)]

        for irow, icol, iarc in product(range(arow), range(acol), range(self.n_archer)):
            # print(irow, icol, iarc)
            abox = (
                self.ratio // 2 + self.ratio * irow,
                self.ratio // 2 + self.ratio * icol,
                *self.archers[iarc]
            )
            train_mask[iarc, 0, irow, icol] = max((mean_iou(abox, label_box) for label_box in bbox))
        print('fff', np.sum(train_mask > 0.5))
        print('fff', img.shape)

        return img, (archor_cls, archor_reg, train_mask)

    def __len__(self):
        return self.data.__len__()


class mrcnn:
    def __init__(self):
        self.device = torch.device('cuda')
        self.net = VGG()
        self.net.to(self.device)
        self.epoach = 0
        self.train_loader = DataLoader(
            BBloader('train', 'voc'),
            1,
            True
        )

    # def train(self):
    #     for img, ()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n = VGG()
    loader = DataLoader(BBloader('train', 'voc'), batch_size=1, shuffle=True)

    for img, label in loader:
        print(img.dtype)
        rr = n(img)
        print(img.shape)
        print(rr[0].shape)
        print(rr[1].shape)
        print(label[0].shape)
        print(label[1].shape)
        break
    print(mean_iou(
        (1000, 1000, 100, 100),
        (1000, 1000, 100, 100)
    ) - 1)
    print(mean_iou(
        (1010, 1010, 100, 100),
        (1000, 1000, 100, 100)
    ) - 0.81 / 1.19)
    print(mean_iou(
        (1010, 1000, 100, 100),
        (1000, 1000, 100, 100)
    ) - 0.9 / 1.1)
    print(mean_iou(
        (1010, 1010, 10, 10),
        (1000, 1000, 100, 100)
    ) - 0.01)
    print(mean_iou(
        (2010, 1010, 10, 10),
        (1000, 1000, 100, 100)
    ) - 0)
