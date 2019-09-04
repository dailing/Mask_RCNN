import math
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
import matplotlib.pyplot as plt
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from util.logs import get_logger
from voc2012 import VOC
from tensorboardX import SummaryWriter
from util.npdraw import draw_bounding_box
from functools import reduce
import random
import matplotlib.pyplot as plt


logger = get_logger('fuck me')


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
                    layers += [
                        conv2d,
                        nn.BatchNorm2d(v),
                        nn.ReLU(inplace=True)]
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

        for irow, icol, iarc in product(
                range(arow),
                range(acol),
                range(self.n_archer)):
            abox = (
                center_rows[irow],
                center_cols[icol],
                *self.archers[iarc]
            )
            iou_on_each_bbox = np.array(
                [mean_iou(abox, label_box) for label_box in bbox])
            iou_map[iarc, 0, irow, icol] = np.max(iou_on_each_bbox)
            if iou_map[iarc, 0, irow, icol] > 0.5:
                arc_to_bbox_map[iarc, 0, irow, icol] = np.argmax(
                    iou_on_each_bbox)
        positive = iou_map > 0.5
        negative = iou_map < 0.3
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

        # logger.info(n_postive, np.sum(negative), np.sum(train_mask))
        # logger.info(positive.shape)

        archor_cls = np.concatenate(
            (negative.astype(np.int), positive.astype(np.int)),
            axis=1)

        for irow, icol, iarc in product(
                range(arow),
                range(acol),
                range(self.n_archer)):
            if not train_mask[iarc, 0, irow, icol]:
                continue
            current_bbox = bbox[arc_to_bbox_map[iarc, 0, irow, icol]]
            t_row = (current_bbox[0] - center_rows[irow]) / \
                self.archers[iarc][0]
            t_col = (current_bbox[1] - center_cols[icol]) / \
                self.archers[iarc][1]
            t_row_len = math.log(current_bbox[2] / self.archers[iarc][0])
            t_col_len = math.log(current_bbox[3] / self.archers[iarc][1])
            archor_reg[iarc, :, irow, icol] = (
                t_row, t_col, t_row_len, t_col_len)
        # logger.info(archor_cls.shape)

        archor_cls = archor_cls.astype(np.float32)
        train_mask = train_mask.astype(np.float32)
        return image, (archor_cls, archor_reg, train_mask)

    def __len__(self):
        return self.data.__len__()


def restore_box_reg(t_row, t_col, t_lrow, t_lcol, arow, acol, a_lrow, a_lcol):
    row = t_row * a_lrow + arow
    col = t_col * a_lcol + acol
    lrow = math.exp(t_lrow) * a_lrow
    lcol = math.exp(t_lcol) * a_lcol
    return row, col, lrow, lcol


class Mrcnn:
    def __init__(
            self,
            device=None,
            model=None,
            train_data=None,
            test_data=None,
            ratio=None,
            net=None):
        if device is None:
            device = 'cuda'
        self.device = torch.device(device)
        if net is None:
            self.net = VGG()
        else:
            self.net = net
        if model is not None:
            logger.info(f'loading form {model}')
            dd = torch.load(model)
            self.net.load_state_dict(dd)
        self.net.to(self.device)
        self.epoach = 0
        if train_data is None:
            self.train_data = BBloader('train', 'voc')
        else:
            self.train_data = train_data
        if test_data is None:
            self.test_data = BBloader('train', 'voc')
        else:
            self.test_data = test_data
        if ratio is None:
            self.ratio = self.test_data.ratio
        else:
            self.ratio = ratio
        self.train_loader = DataLoader(
            self.train_data,
            3,
            True,
            num_workers=12)
        self.test_loader = DataLoader(self.test_data, 1, True, num_workers=12)
        self.optm = SGD(
            self.net.parameters(),
            lr=0.0003,
            momentum=0.9,
            nesterov=True,
            weight_decay=0.00005
        )
        self.logWriter = SummaryWriter(logdir=f'log/fuck')
        self.thresh = 0.999

    def train(self):
        self.net.train()
        tt = tqdm(self.train_loader, total=len(self.train_loader))
        for idx, (img, (cls, reg, mask)) in enumerate(tt):
            mask = mask.numpy()
            mask_sum = mask.sum()
            rand_mask = np.random.rand(*mask.shape) > (self.thresh) ** 0.02
            rand_mask_sum = rand_mask.sum()
            self.thresh += (rand_mask_sum - mask_sum) / mask.size
            mask += rand_mask
            mask = mask > 0
            mask_sum = mask.sum()
            if mask_sum == 0:
                continue

            mask = torch.Tensor(mask.astype(np.float32))
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
            loss = L_cls + (0.5 * L_reg if positive_sum > 0 else 0)
            loss.backward()
            self.optm.step()

            loss = loss.detach().cpu()
            tt.set_postfix_str(str(loss))
            self.logWriter.add_scalar(
                'train/loss',
                float(loss),
                (self.epoach - 1) * len(self.train_loader) + idx)

    def predict(self, image: np.ndarray):
        # plt.figure()
        # plt.imshow(image)
        # logger.info(image.shape)
        image = image.transpose(2, 0, 1)
        image = image[np.newaxis, ::]
        image = image.astype(np.float32)
        image = torch.Tensor(image)
        image = image.to(self.device)
        cls, reg = self.net(image)
        cls = cls.detach().cpu().numpy()
        reg = reg.detach().cpu().numpy()
        cls = cls.reshape((
            cls.shape[1] // 2,
            2,
            *cls.shape[2:]
        ))
        return cls, reg
        # logger.info(cls.shape)
        # idx = 0
        # plt.figure()
        # plt.imshow(cls[idx, 1] - cls[idx, 0], cmap='gray')
        # plt.colorbar()
        # plt.show()
        # return None
        # reg = reg.reshape((
        #     reg.shape[1] // 4,
        #     4,
        #     *reg.shape[2:]
        # ))
        # logger.info(cls.shape)
        # result = []
        # for iarch, irow, icol in product(
        #         range(cls.shape[0]),
        #         range(cls.shape[2]),
        #         range(cls.shape[3])):
        #     if cls[iarch, 1, irow, icol] > cls[iarch, 0, irow, icol]:
        #         reg[iarch, :, irow, icol]
        #         logger.info(f'{iarch}, {len(self.test_data.archers)}')
        #         self.test_data.archers[iarch]
        #         regresult = restore_box_reg(
        #             *reg[iarch, :, irow, icol].tolist(),
        #             self.ratio // 2 + irow * self.ratio,
        #             self.ratio // 2 + icol * self.ratio,
        #             *self.test_data.archers[iarch],
        #         )
        #         result.append(regresult)
        return result

    def test(self):
        tt = tqdm(self.test_loader, total=len(self.test_loader))
        for idx, (img, (cls, reg, mask)) in enumerate(tt):
            img = img.to(self.device)
            real_img = img.detach().numpy()
            real_img = real_img[0, :, :, :].transpose(1, 2, 0)
            pcls, preg = self.net(img)

            pcls = cls.detach().numpy()
            pcls = pcls.reshape(self.test_data.n_archer, 2, *pcls.shape[-2:])
            preg = preg.detach().numpy()
            preg = preg.reshape(self.test_data.n_archer, 4, *preg.shape[-2:])
            for iarch, irow, icol in product(
                    range(pcls.shape[0]),
                    range(pcls.shape[2]),
                    range(pcls.shape[3])):
                if pcls[iarch, 1, irow, icol] > pcls[iarch, 0, irow, icol]:
                    regresult = restore_box_reg(
                        *preg[iarch, :, irow, icol].tolist(),
                        self.ratio // 2 + irow * self.ratio,
                        self.ratio // 2 + icol * self.ratio,
                        *self.test_data.archers[iarch],
                    )
                    logger.info(f'draw bounding box {iarch} {irow}, {icol}')
                    real_img = draw_bounding_box(real_img, *regresult,
                                                 (0, 1, 0, 0.5),
                                                 bg_color=(1, 0, 0, 0.02))
            logger.info(pcls.shape)

            plt.figure()
            plt.imshow(real_img)
            plt.show()

    def step(self, n=300):
        for i in range(n):
            self.epoach += 1
            self.train()
            torch.save(
                self.net.state_dict(),
                f'runs/model_{self.epoach:04}.model')


if __name__ == '__main__':
    import sys
    from fire import Fire

    Fire(Mrcnn)
    sys.exit(0)

    traner = Mrcnn()
    traner.step()

    sys.exit(0)

    n = VGG()
    dataset = BBloader('train', 'voc')
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for img, (cls, reg, mask) in loader:
        logger.info(img.dtype)
        pcls, preg = n(img)
        img = img.detach().numpy()[0, :, :, :]
        img = img.transpose(1, 2, 0)
        logger.info(img.shape)
        plt.figure()
        plt.imshow(img)
        # plt.show()

        mask = mask.detach().numpy()
        mask = mask[0, ::]
        cls = cls.detach().numpy()
        cls = cls[0, ::]
        # cls = cls.reshape(9, 2, *cls.shape[2:])
        reg = reg.detach().numpy()
        reg = reg[0, ::]
        # reg = reg.reshape(9, 4, *reg.shape[2:])
        logger.info(cls.shape)
        logger.info(reg.shape)
        logger.info(mask.shape)
        logger.info(np.sum(mask))
        logger.info(np.sum(cls[:, 1, :, :]))
        for irow, icol, iarch in product(
                range(mask.shape[2]),
                range(mask.shape[3]),
                range(mask.shape[0])):
            # if mask[iarch, 0, irow, icol] == 0:
            #     continue
            if cls[iarch, 1, irow, icol] == 0:
                continue
            rr = restore_box_reg(
                *reg[iarch, :, irow, icol].tolist(),
                dataset.ratio // 2 + dataset.ratio * irow,
                dataset.ratio // 2 + dataset.ratio * icol,
                *dataset.archers[iarch],
            )
            logger.info('drawing bounding box')
            img = draw_bounding_box(
                img,
                dataset.ratio // 2 + dataset.ratio * irow,
                dataset.ratio // 2 + dataset.ratio * icol,
                *dataset.archers[iarch],
                color=(0, 1, 0, 0.4)
            )
            img = draw_bounding_box(
                img,
                *rr,
                color=(1, 0, 0, 1)
            )
        plt.figure()
        plt.imshow(img)
        plt.show()
        break
    logger.info(mean_iou(
        (1000, 1000, 100, 100),
        (1000, 1000, 100, 100)
    ) - 1)
    logger.info(mean_iou(
        (1010, 1010, 100, 100),
        (1000, 1000, 100, 100)
    ) - 0.81 / 1.19)
    logger.info(mean_iou(
        (1010, 1000, 100, 100),
        (1000, 1000, 100, 100)
    ) - 0.9 / 1.1)
    logger.info(mean_iou(
        (1010, 1010, 10, 10),
        (1000, 1000, 100, 100)
    ) - 0.01)
    logger.info(mean_iou(
        (2010, 1010, 10, 10),
        (1000, 1000, 100, 100)
    ) - 0)
