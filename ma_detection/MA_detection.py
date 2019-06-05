import math
import os
import pickle
import sys
from itertools import product
from os.path import join as pjoin, exists

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from tqdm import tqdm
from tensorboardX import SummaryWriter

from augment import FundusAOICrop, CompostImageAndLabel
from model import mean_iou, Mrcnn
from util.files import assert_exist, check_exist
from util.logs import get_logger
from util.npdraw import draw_bounding_box
from util.segmentation2bbox import segmentation2bbox
from model import restore_box_reg
import matplotlib.pyplot as plt
logger = get_logger('ma detection')


class VGG(nn.Module):
    def __init__(self, init_weights=True):
        super(VGG, self).__init__()
        self.features = self.make_layers(
            [64, 64, 'M',  # 3, 5, 6
             128, 128, 'M',  # 10, 14 16
             256, 256, 512, 'M',  # 24, 32, 40, 44
             512, 512, 512, 'M',  # 60, 76, 92, 100
             # 512, 512, 512, 'M',
             ],  # 132, 164, 196, 212
            batch_norm=True)

        self.rpn_sliding_window = nn.Conv2d(
            512, 256, 3, 1, 1
        )
        self.box_classification = nn.Conv2d(256, 2 * 1, 1)
        self.box_regression = nn.Conv2d(256, 2 * 1, 1)
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
                    layers += [
                        conv2d, nn.BatchNorm2d(v),
                        nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


class ChallengeDB:
    def __init__(self,
                 root='/home/d/data/challenge/A. Segmentation/',
                 split=None):
        self.split = split
        self.transform = [
            FundusAOICrop(),
        ]
        if split == 'train':
            self.dataFiles = tuple((
                assert_exist(pjoin(
                    root,
                    f'1. Original Images/a. Training Set/IDRiD_{i:02d}.jpg')),
                check_exist(pjoin(
                    root,
                    f'2. All Segmentation Groundtruths/a. Training Set/'
                    f'1. Microaneurysms/IDRiD_{i:02d}_MA.tif')),
                check_exist(pjoin(
                    root,
                    f'2. All Segmentation Groundtruths/a. Training Set/'
                    f'2. Haemorrhages/IDRiD_{i:02d}_HE.tif')),
                check_exist(pjoin(
                    root,
                    f'2. All Segmentation Groundtruths/a. Training Set/'
                    f'3. Hard Exudates/IDRiD_{i:02d}_EX.tif')),
                check_exist(pjoin(
                    root,
                    f'2. All Segmentation Groundtruths/a. Training Set/'
                    f'4. Soft Exudates/IDRiD_{i:02d}_SE.tif')),
                check_exist(pjoin(
                    root,
                    f'2. All Segmentation Groundtruths/a. Training Set/'
                    f'5. Optic Disc/IDRiD_{i:02d}_OD.tif'))
            ) for i in range(1, 55))
        elif split == 'test':
            self.dataFiles = tuple((
                assert_exist(pjoin(
                    root,
                    f'1. Original Images/b. Testing Set/IDRiD_{i:02d}.jpg')),
                check_exist(pjoin(
                    root,
                    f'2. All Segmentation Groundtruths/'
                    f'b. Testing Set/1. Microaneurysms/IDRiD_{i:02d}_MA.tif')),
                check_exist(pjoin(
                    root,
                    f'2. All Segmentation Groundtruths/'
                    f'b. Testing Set/2. Haemorrhages/IDRiD_{i:02d}_HE.tif')),
                check_exist(pjoin(
                    root,
                    f'2. All Segmentation Groundtruths/b. Testing Set/'
                    f'3. Hard Exudates/IDRiD_{i:02d}_EX.tif')),
                check_exist(pjoin(
                    root,
                    f'2. All Segmentation Groundtruths/b. Testing Set/'
                    f'4. Soft Exudates/IDRiD_{i:02d}_SE.tif')),
                check_exist(pjoin(
                    root,
                    f'2. All Segmentation Groundtruths/b. Testing Set/'
                    f'5. Optic Disc/IDRiD_{i:02d}_OD.tif'))
            ) for i in range(55, 82))
        else:
            raise Exception(f'split ({split}) not recognized!')
        self.cacheTransform = CompostImageAndLabel(self.transform)
        self.cacheDir = 'runs/cache/'

    def _getCacheItem(self, index):
        cacheName = pjoin(
            self.cacheDir,
            f'ma_detection.{self.split}.{index}.pkl')
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
        if index >= len(self.dataFiles):
            raise IndexError()
        index = index % len(self.dataFiles)
        images = self._getCacheItem(index)
        image = images[0]
        xx = (np.zeros(images[1].shape, images[1].dtype), *images[1:])
        labels = np.array(xx)
        labels = labels.argmax(0)
        return image, labels

    def __len__(self):
        return len(self.dataFiles)


def slide_image(img, size, overlap=0.2):
    image_patch = []
    top_left_points = []
    stride = (1 - overlap) * size

    def calculate_start_p(stride, size):
        num_step = math.ceil(size / stride)
        stride = size / num_step
        for i in range(num_step + 1):
            yield math.floor(stride * i)

    for row, col in product(
            calculate_start_p(stride, img.shape[0] - size),
            calculate_start_p(stride, img.shape[1] - size)):
        row = min(img.shape[0] - size, row)
        col = min(img.shape[1] - size, col)
        image_patch.append(img[row:row + size, col:col + size, ::])
        top_left_points.append((row, col))
    return image_patch, top_left_points


class BBloader(Dataset):
    def __init__(self, split, archers=None):
        self.split = split
        if archers is None:
            archers = [
                (30, 30),
            ]
        self.archers = archers
        self.n_archer = len(self.archers)
        self.ratio = 16
        self.file_list = self._make_slices()
        self.thresh = 0.99
        self.smooth_factor = 100000

    def _make_slices(self):
        store_dir = f'runs/fundus_image_data/{self.split}'
        csv_file = pjoin(store_dir, f'list.csv')
        image_size = 512
        if exists(csv_file):
            dd = pd.read_csv(csv_file)
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
                all_bbox = []
                for lesionType in range(4):
                    bbox = segmentation2bbox(gtp == lesionType + 1)
                    bbox = list(map(
                        lambda x: (*x, lesionType),
                        bbox
                    ))
                    logger.info(bbox)
                    all_bbox += bbox
                pickle.dump(
                    (p, all_bbox),
                    open(record_name, 'wb'))
                records.append(dict(
                    file=record_name
                ))
        data_csv = pd.DataFrame.from_records(records)
        data_csv.to_csv(csv_file, index=False)
        return data_csv

    def __getitem__(self, index):
        if index >= self.__len__():
            raise IndexError()
        image, bbox = pickle.load(open(self.file_list.file[index], 'rb'))
        image = image.astype(np.float)
        image = image.transpose((2, 0, 1)) / 255
        image = image.astype(np.float32)
        nchannel, nrow, ncol = image.shape

        arow, acol = nrow // self.ratio, ncol // self.ratio
        archor_reg = np.zeros((self.n_archer, 2, arow, acol), np.float32)
        positive = np.zeros((self.n_archer, 1, arow, acol), np.int)
        negative = np.zeros((self.n_archer, 1, arow, acol), np.int)
        arc_to_bbox_map = np.zeros((self.n_archer, 1, arow, acol), np.int) - 1

        center_rows = [self.ratio // 2 + i * self.ratio for i in range(arow)]
        center_cols = [self.ratio // 2 + i * self.ratio for i in range(acol)]
        for bbox_idx, label_box in enumerate(bbox):
            if label_box[-1] != 1:
                continue
            iou_map = np.zeros((self.n_archer, 1, arow, acol), np.float)
            for irow, icol, iarc in product(
                    range(arow),
                    range(acol),
                    range(self.n_archer)):
                abox = (
                    center_rows[irow],
                    center_cols[icol],
                    *self.archers[iarc]
                )
                iou_map[iarc, 0, irow, icol] = mean_iou(abox, label_box)
            h_thresh = 0
            l_thresh = 0
            positive += iou_map > h_thresh
            negative += iou_map <= l_thresh
            arc_to_bbox_map[iou_map >= h_thresh] = bbox_idx
        positive = positive > 0
        positive = positive.astype(np.float32)
        negative = negative > 0
        negative = negative.astype(np.float32)

        n_postive = np.sum(positive)
        mask = positive > 0
        random_sample = np.random.rand(*mask.shape)
        mask += random_sample > self.thresh
        mask = mask > 0
        mask = mask.astype(np.float32)
        n_psample = np.sum((mask * positive) > 0)
        n_nsample = np.sum((mask * negative) > 0)
        self.thresh += n_psample / self.smooth_factor
        self.thresh -= n_nsample / self.smooth_factor

        archor_cls = np.concatenate(
            (negative.astype(np.int), positive.astype(np.int)),
            axis=1)

        # logger.info(f'{arow}, {acol}, {self.n_archer}, {len(bbox)}')
        for irow, icol, iarc in product(
                range(arow),
                range(acol),
                range(self.n_archer)):
            if positive[iarc, 0, irow, icol] == 0:
                continue
            current_bbox = bbox[arc_to_bbox_map[iarc, 0, irow, icol]]
            t_row = (current_bbox[0] - center_rows[irow])\
                / self.archers[iarc][0]
            t_col = (current_bbox[1] - center_cols[icol])\
                / self.archers[iarc][1]
            archor_reg[iarc, :, irow, icol] = (
                t_row, t_col)
        archor_cls = archor_cls.astype(np.float32)
        return image, (archor_cls, archor_reg, mask)

    def __len__(self):
        return self.file_list.__len__()


class MaDetector:
    def __init__(
            self,
            device=None,
            model=None,
            train_data=None,
            test_data=None,
            ratio=None,
            net=None):
        self.device = torch.device(device) if device else torch.device('cpu')
        self.net = VGG() if net is None else net
        if model is not None:
            logger.info(f'loading form {model}')
            dd = torch.load(model)
            self.net.load_state_dict(dd)
        self.net.to(self.device)
        self.epoach = 0
        self.train_data = BBloader('train', [(30, 30)]) 
        self.test_data = BBloader('train', [(30, 30)])
        if ratio is None:
            self.ratio = self.train_data.ratio
        else:
            self.ratio = ratio
        self.train_loader = DataLoader(
            self.train_data,
            3,
            True,
            num_workers=5)
        self.optm = SGD(
            self.net.parameters(),
            lr=0.0003,
            momentum=0.9,
            nesterov=True,
            weight_decay=0.00005
        )
        self.logWriter = SummaryWriter(logdir=f'log/fuck')

    def train(self):
        self.net.train()
        tt = tqdm(self.train_loader, total=len(self.train_loader))
        for idx, (img, (cls, reg, mask)) in enumerate(tt):
            npmask = mask.numpy()
            if npmask.sum() == 0:
                continue
            mask = mask.to(self.device)

            img = img.to(self.device)
            cls = cls.to(self.device)
            reg = reg.to(self.device)
            mask = mask.to(self.device)
            pcls, preg = self.net(img)

            # Calculate Loss_cls
            pcls = pcls.reshape(cls.shape)
            pcls = torch.nn.functional.log_softmax(pcls, dim=2)
            pcls = pcls * cls
            pcls = pcls * mask
            L_cls = - pcls.sum() / mask.sum()

            # Calculate Loss_reg
            preg = preg.reshape(reg.shape)
            L_reg = torch.abs(preg - reg)
            L_reg = torch.where(L_reg < 1, 0.5 * L_reg ** 2, L_reg - 0.5)
            positive = cls[:, :, 1:, :, :]
            positive_sum = positive.sum()
            L_reg = L_reg * positive
            L_reg = L_reg.sum() / positive_sum

            self.optm.zero_grad()
            loss = L_cls + (0.1 * L_reg if positive_sum > 0 else 0)
            loss.backward()
            self.optm.step()

            loss = loss.detach().cpu()
            tt.set_postfix_str(str(loss))
            self.logWriter.add_scalar(
                'train/loss',
                float(loss),
                (self.epoach - 1) * len(self.train_loader) + idx)

    def step(self, n=300):
        for i in tqdm(range(n), total=n):
            self.epoach += 1
            self.train()

if __name__ == '__main__':

    detector = MaDetector(
        device='cuda',
        net=VGG(),
    )
    detector.step()
    sys.exit(0)

    loader = BBloader(split='train')

    for i in range(100):
        idx = random.randint(0, len(loader))
        iimg, (classification, regression, mask) = loader[idx]
        logger.info(iimg.shape)

        plt.figure()
        plt.imshow(classification[0, 1, :, :])
        plt.colorbar()

        plt.figure()
        plt.imshow(iimg.transpose(1, 2, 0))

        plt.figure()
        plt.imshow(mask[0, 0, ::])
        plt.show()
    sys.exit(0)

    logger.info(iimg.shape)

    image = Image.open('/data/home/d/data/challenge/A. Segmentation/'
                       '1. Original Images/a. Training Set/IDRiD_34.jpg')
    image = np.array(image)
    imgs, pos = slide_image(image, 512, 0.2)
    result_patchs = []

    heatmap = np.zeros((1, image.shape[0]//16, image.shape[1]//16))
    counter = np.zeros((1, image.shape[0]//16, image.shape[1]//16))
    for img_patchs, p in zip(imgs, pos):
        cls, reg = det.predict(img_patchs)
        p = tuple(i // 16 for i in p)
        hh = cls[:, 1, :, :]-cls[:, 0, :, :]
        heatmap[:, p[0]:p[0]+32, p[1]:p[1]+32] += hh
        counter[:, p[0]:p[0]+32, p[1]:p[1]+32] += 1
    heatmap /= counter
    for iarchor in range(15):
        plt.figure()
        plt.imshow(heatmap[iarchor, :, :])
        plt.colorbar()
        plt.figure()
        plt.imshow(counter[iarchor, :, :])
        plt.show()
