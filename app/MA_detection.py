import os
import pickle
from itertools import product
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from model import Mrcnn, mean_iou
import torch.nn as nn
import math
from os.path import join as pjoin, exists
import sys

from util.files import assert_exist, check_exist
from voc2012 import VOC
from augment import FundusAOICrop, Resize, ToFloat, RandFlip, RandRotate, \
    RandomNoise, ToTensor, CompostImageAndLabel, ToNumpyType

from util.logs import get_logger
from collections import namedtuple

logger = get_logger('ma detection')


class VGG(nn.Module):
    def __init__(self, init_weights=True):
        super(VGG, self).__init__()
        self.features = self.make_layers(
            [64, 64, 'M',  # 3, 5, 6
             128, 128, 'M',  # 10, 14 16
             256, 256, 512, 'M',  # 24, 32, 40, 44
             # 512, 512, 512, 'M',  # 60, 76, 92, 100
             # 512, 512, 512, 'M',
             ],  # 132, 164, 196, 212
            batch_norm=True)

        self.rpn_sliding_window = nn.Conv2d(
            512, 256, 3, 1, 15
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


class ChallengeDB:
    def __init__(self,
                 root='/home/d/data/challenge/A. Segmentation/',
                 split=None,
                 augment_transform=False,
                 repeat=1):
        self.split = split
        self.transform = [
            FundusAOICrop(),
            # Resize(1200),
            (ToFloat(), None),
        ]
        if augment_transform == True:
            self.transform += [
                (RandomNoise(), None),
                RandFlip(),
                RandRotate(),
            ]
        elif augment_transform == False:
            pass
        elif augment_transform is not None:
            self.transform += augment_transform
        self.transform += [
            (ToTensor(), ToNumpyType(np.int)),
        ]
        if split == 'train':
            self.dataFiles = tuple(
                (
                    assert_exist(pjoin(root, f'1. Original Images/a. Training Set/IDRiD_{i:02d}.jpg')),
                    check_exist(pjoin(root, f'2. All Segmentation Groundtruths/'
                    f'a. Training Set/1. Microaneurysms/IDRiD_{i:02d}_MA.tif')),
                    check_exist(pjoin(root, f'2. All Segmentation Groundtruths/'
                    f'a. Training Set/2. Haemorrhages/IDRiD_{i:02d}_HE.tif')),
                    check_exist(pjoin(root, f'2. All Segmentation Groundtruths/a. Training Set/'
                    f'3. Hard Exudates/IDRiD_{i:02d}_EX.tif')),
                    check_exist(pjoin(root, f'2. All Segmentation Groundtruths/a. Training Set/'
                    f'4. Soft Exudates/IDRiD_{i:02d}_SE.tif')),
                    check_exist(pjoin(root, f'2. All Segmentation Groundtruths/a. Training Set/'
                    f'5. Optic Disc/IDRiD_{i:02d}_OD.tif'))
                ) for i in range(1, 55)
            )
        elif split == 'test':
            self.dataFiles = tuple(
                (
                    assert_exist(pjoin(root, f'1. Original Images/b. Testing Set/IDRiD_{i:02d}.jpg')),
                    check_exist(pjoin(root, f'2. All Segmentation Groundtruths/'
                    f'b. Testing Set/1. Microaneurysms/IDRiD_{i:02d}_MA.tif')),
                    check_exist(pjoin(root, f'2. All Segmentation Groundtruths/'
                    f'b. Testing Set/2. Haemorrhages/IDRiD_{i:02d}_HE.tif')),
                    check_exist(pjoin(root, f'2. All Segmentation Groundtruths/b. Testing Set/'
                    f'3. Hard Exudates/IDRiD_{i:02d}_EX.tif')),
                    check_exist(pjoin(root, f'2. All Segmentation Groundtruths/b. Testing Set/'
                    f'4. Soft Exudates/IDRiD_{i:02d}_SE.tif')),
                    check_exist(pjoin(root, f'2. All Segmentation Groundtruths/b. Testing Set/'
                    f'5. Optic Disc/IDRiD_{i:02d}_OD.tif'))
                ) for i in range(55, 82)
            )
        else:
            raise Exception(f'split ({split}) not recognized!')
        cacheStep = 2
        self.cacheTransform = CompostImageAndLabel(self.transform[:cacheStep])
        self.transform = CompostImageAndLabel(self.transform[cacheStep:])
        self.repeat = repeat
        self.cacheDir = 'runs/cache/'

    def _getCacheItem(self, index):
        cacheName = pjoin(self.cacheDir, f'ma_detection.{self.split}.{index}.pkl')
        if False and exists(cacheName):
            result = pickle.load(open(cacheName, 'rb'))
            return result
        else:
            logger.debug(f'miss {cacheName}')
            files = self.dataFiles[index]
            images = []
            for i in files:
                if i is None:
                    images.append(np.zeros(images[-1].shape[:2], np.uint8))
                else:
                    images.append(np.array(Image.open(i)))
            record = self.cacheTransform(*images)
            try:
                pickle.dump(record, open(cacheName, 'wb'))
            except Exception as e:
                os.remove(cacheName)
                raise e
        return record

    def __getitem__(self, index):
        logger.debug(f'getting {index}')
        if index >= len(self.dataFiles) * self.repeat:
            raise IndexError()
        index = index % len(self.dataFiles)
        images = self._getCacheItem(index)
        images = self.transform(*images)
        image = images[0]
        xx = (np.zeros(images[1].shape, images[1].dtype), *images[1:])
        labels = np.array(xx)
        labels = labels.argmax(0)
        return image, labels

    def __len__(self):
        return len(self.dataFiles) * self.repeat


def slide_image(img, size, overlap=0.2):
    image_patch = []
    top_left_points = []
    stride = (1-overlap)*size
    def calculate_start_p(stride, size):
        num_step = math.ceil(size/stride)
        stride = size / num_step
        for i in range(num_step+1):
            yield math.floor(stride*i)

    for row, col in product(
            calculate_start_p(stride, img.shape[-2] - size),
            calculate_start_p(stride, img.shape[-1] - size)):
        row = min(img.shape[-2] - size, row)
        col = min(img.shape[-1] - size, col)
        image_patch.append(img[::, row:row + size, col:col + size])
        top_left_points.append((row, col))
    return image_patch, top_left_points


class BBloader(Dataset):
    def __init__(self, split, archers=None):
        self.split = split
        if archers is None:
            archers = [
                (5, 5),
            ]
        self.n_archer = len(archers)
        self.archers = archers
        self.ratio = 32

    def _make_slices(self):
        store_dir = f'runs/fundus_image_data/{self.split}'
        csv_file = pjoin(store_dir, f'list.csv')
        image_size = 512
        stride = 400
        if exists(csv_file):
            dd = pd.read_csv(csv_file)
            logger.info(dd.head())
            return dd
        if not exists(store_dir):
            os.makedirs(store_dir, exist_ok=True)
        data = ChallengeDB(split=self.split)
        records = []
        for img, gt in data:
            logger.info(img.shape)
            logger.info(gt.shape)
            image_patches, cornels = slide_image(img, image_size)
            for p, c in zip(image_patches, cornels):
                idx = len(records)
                record_name = pjoin(store_dir, f'data{idx}.pickle')
                gtp = gt[
                      c[0]:c[0] + image_size,
                      c[1]:c[1] + image_size]
                pickle.dump(
                    (p, gtp, c),
                    open(record_name, 'wb'))
                records.append(dict(
                    file=record_name
                ))
            break
        data_csv = pd.DataFrame.from_records(records)
        data_csv.to_csv(csv_file, index=False)
        logger.info(data_csv.head())


    def __getitem__(self, index):
        image, bbox = self.data[index]
        image = image.astype(np.float)
        image = image.transpose((2, 0, 1)) / 255
        image = image.astype(np.float32)
        nchannel, nrow, ncol = image.shape

        arow, acol = nrow // self.ratio, ncol // self.ratio
        archor_reg = np.zeros((self.n_archer, 4, arow, acol), np.float32)
        iou_map = np.zeros((self.n_archer, 1, arow, acol), np.float)
        arc_to_bbox_map = np.zeros((self.n_archer, 1, arow, acol), np.int) - 1

        center_rows = [self.ratio // 2 + i * self.ratio for i in range(arow)]
        center_cols = [self.ratio // 2 + i * self.ratio for i in range(acol)]

        for irow, icol, iarc in product(range(arow), range(acol), range(self.n_archer)):
            abox = (
                center_rows[irow],
                center_cols[icol],
                *self.archers[iarc]
            )
            iou_on_each_bbox = np.array([mean_iou(abox, label_box) for label_box in bbox])
            iou_map[iarc, 0, irow, icol] = np.max(iou_on_each_bbox)
            if iou_map[iarc, 0, irow, icol] > 0.5:
                arc_to_bbox_map[iarc, 0, irow, icol] = np.argmax(iou_on_each_bbox)
        positive = iou_map > 0.01
        negative = iou_map < 0.01
        n_postive = np.sum(positive)
        negative_points = np.where(negative)
        indices = np.random.choice(
            np.arange(negative_points[0].size),
            replace=False,
            size=max(0, negative_points[0].size - n_postive)
        )
        sss = tuple((i[indices] for i in negative_points))
        negative[sss] = 0
        train_mask = positive | negative
        archor_cls = np.concatenate((negative.astype(np.int), positive.astype(np.int)), axis=1)
        for irow, icol, iarc in product(range(arow), range(acol), range(self.n_archer)):
            if not train_mask[iarc, 0, irow, icol]:
                continue
            current_bbox = bbox[arc_to_bbox_map[iarc, 0, irow, icol]]
            t_row = (current_bbox[0] - center_rows[irow]) / self.archers[iarc][0]
            t_col = (current_bbox[1] - center_cols[icol]) / self.archers[iarc][1]
            t_row_len = math.log(current_bbox[2] / self.archers[iarc][0])
            t_col_len = math.log(current_bbox[3] / self.archers[iarc][1])
            archor_reg[iarc, :, irow, icol] = (t_row, t_col, t_row_len, t_col_len)
        archor_cls = archor_cls.astype(np.float32)
        train_mask = train_mask.astype(np.float32)
        return image, (archor_cls, archor_reg, train_mask)

    def __len__(self):
        return self.data.__len__()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dd = BBloader(split='train')
    dd._make_slices()
    sys.exit(0)

    dev = torch.device('cpu')
    ddata = ChallengeDB(split='train')
    net = VGG()
    net = net.to(dev)
    image, label = ddata[23]
    plt.figure()
    out = net(torch.Tensor(image[np.newaxis, ::]).to(dev))
    print(out)
    image = image.transpose(1, 2, 0)
    logger.info(image.shape)
    plt.imshow(image)
    plt.figure()
    plt.imshow(label == 5)
    plt.show()
