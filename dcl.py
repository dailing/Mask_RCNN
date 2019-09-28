import random
from io import BytesIO
from itertools import product
from os.path import join as pjoin

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sqlitedict import SqliteDict
from tensorboardX import SummaryWriter
from torch.optim import Adam, SGD
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from util.tasks import MServiceInstance, Task
from util.config import Configure

from util.augment import ResizeKeepAspectRatio, Compose, \
    RandomNoise, RandFlip, ToTensor, ToFloat, FundusAOICrop, \
    Resize, RandRotate, RangeCenter, RandomCrop, Normalize
from util.logs import get_logger
from scipy.special import softmax
import sys
from dataset import datasets as avaliable_datasets
import argparse
from sklearn.metrics import average_precision_score, roc_auc_score
import util.bconfig
import model


logger = get_logger('fff')
device = torch.device('cuda')


class NetModel(nn.Module):
    def __init__(self, config, with_feature=False):
        super().__init__()
        if config.net_parameters is None:
            config.net_parameters = {}
        self.base_net = config.basenet(**config.net_parameters)
        self.config = config
        self.with_feature = with_feature
        for i in self.config.outputs:
            logger.info(f'{i.layer_parameters}')
            layer = i.model(**i.layer_parameters)
            self.__setattr__(i.name_output, layer)

    def forward(self, x):
        values = self.base_net(x)
        outputs = {}
        for i in self.config.outputs:
            layer = self.__getattr__(i.name_output)
            assert i.name_input in values, \
                f'require {i.name_input}, has {list(values.keys())}'
            out = layer(values[i.name_input])
            values[i.name_output] = out
            outputs[i.name_output] = out
        if self.with_feature:
            return outputs, values
        return outputs


class TrainEvalDataset(Dataset):
    def __init__(self, data_reader, config, N=7, __k=2,
                 augment=False, split='train', root=None, swap=False):
        super().__init__()
        self.config = config
        self.data_reader = data_reader
        self.N = N
        self.k = __k
        self.swap_img = swap
        resiez_size = int(self.config.input_size*1.2)
        if split != 'train':
            resiez_size = self.config.input_size
        self.transform = [
            ToFloat(),
            Resize(resiez_size),
            # RangeCenter(),
            Normalize(
                # mean=np.array([0.485, 0.456, 0.406], dtype=np.float32),
                # std=np.array([0.229, 0.224, 0.225], dtype=np.float32)),
                mean=np.array([0.5, 0.5, 0.5], dtype=np.float32),
                std=np.array([0.5, 0.5, 0.5], dtype=np.float32)),
        ]
        if split == 'train':
            self.transform += [
                RandomNoise(),
                RandFlip(),
                # RandRotate(),
                RandomCrop(self.config.input_size)
            ]
        self.transform += [
            ToTensor()
        ]
        self.transform = Compose(self.transform)

    def _generate_random_perm(self):
        original = np.mgrid[0:self.N, 0:self.N]
        new_order = np.mgrid[0:self.N, 0:self.N]
        perm_row = ((np.random.rand(self.N)-0.5)*2*self.k +
                    np.arange(self.N)).argsort()
        perm_col = ((np.random.rand(self.N)-0.5)*2*self.k +
                    np.arange(self.N)).argsort()

        new_order[:, np.arange(self.N), :] = new_order[:, perm_row, :]
        new_order[:, :, np.arange(self.N)] = new_order[:, :, perm_col]
        original = original.astype(np.int32)
        new_order = new_order.astype(np.int32)
        return original, new_order

    def swap(self, image):
        original_matrix, swap_matrix = self._generate_random_perm()
        stride = int(image.shape[1] / self.N)
        new_img = np.zeros(image.shape, image.dtype)
        for row, col in product(range(self.N), range(self.N)):
            n_row, n_col = swap_matrix[:, row, col]
            new_img[
                :,
                (row*stride):(row*stride+stride),
                (col*stride):(col*stride+stride), ] = image[
                :,
                (n_row*stride):(n_row*stride+stride),
                (n_col*stride):(n_col*stride+stride),
            ]
        return new_img, swap_matrix, original_matrix

    def __getitem__(self, index):
        try:
            image, label = self.data_reader[index]
            image = self.transform(image)
            # logger.info(image.shape)
            # assert len(image.shape) == 3
            # assert image.shape[0] == 3
            return image, label
        except Exception as e:
            logger.error(e, exc_info=True)
            # return self.__getitem__(index-1)
        # if image is None:
        #     return self.__getitem__(index-1)

    def __len__(self):
        return self.data_reader.__len__()


def metrics_acc(output, label):
    output = output.argmax(axis=1)
    return (output == label).mean()


def metrics_AP(output, label):
    n_class = output.shape[1]
    AP = {}
    for i in range(n_class):
        AP[i] = average_precision_score(label == i, output[:, i])
    logger.info(AP)
    return np.mean(list(AP.values()))


def metrics_mRoc(output, label):
    n_class = output.shape[1]
    mROC = {}
    for i in range(n_class):
        try:
            mROC[i] = roc_auc_score(label == i, output[:, i])
        except ValueError as e:
            mROC[i] = 0
    logger.info(mROC)
    return np.mean(list(mROC.values()))


def calculate_metrics(config, output_all, label_all):
    result = {}
    for metric in config:
        output = np.array(output_all[metric.predicts])
        label = np.array(label_all[metric.ground_truth])
        result[metric.name] = metric.func(output, label)
    # result['mapping'] = calculate_mapping(output_all, label_all)
    return result


def test(config, net, data_loader, epoach):
    net.eval()
    global_step = epoach * len(data_loader)
    cls_predict, cls_gt = {}, {}
    for batch_cnt, batch in tqdm(
            enumerate(data_loader), total=len(data_loader)):
        image, label = batch
        image = image.to(device)
        for k, v in label.items():
            if isinstance(v, torch.Tensor):
                label[k] = label[k].to(device)
        net_out = net(image)
        loss_sum, loss_map = calculate_loss(
            config.net.loss, net_out, label)
        wtire_summary(loss_map, 'test', global_step)
        for k, v in net_out.items():
            if k not in cls_predict:
                cls_predict[k] = []
            cls_predict[k] += v.detach().cpu().numpy().tolist()
        for k, v in label.items():
            if k not in cls_gt:
                cls_gt[k] = []
            if not isinstance(v, torch.Tensor):
                continue
            cls_gt[k] += v.detach().cpu().numpy().tolist()
        global_step += 1
    test_metrix = calculate_metrics(config.metrics, cls_predict, cls_gt)
    logger.info(test_metrix)
    for k, v in test_metrix.items():
        summery_writer.add_scalar(
            f'metrics/{k}',
            v,
            epoach
        )


def predict(
        config, dataset, model_file,
        num_workers=10, batch_size=1, gpu=True):
    predict_device = torch.device('cuda' if gpu else 'cpu')
    loader = DataLoader(
        TrainEvalDataset(
            dataset,
            config),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers)
    net = NetModel(config.net, with_feature=True)
    net = nn.DataParallel(net)
    unused = net.load_state_dict(
        {(('module.base_net.'+k) if not
            k.startswith('module.base_net') else k): v
            for k, v in torch.load(model_file, map_location='cpu').items()},
        strict=False)
    logger.info(unused)
    net = net.to(predict_device)
    net.eval()
    outputs = []
    features = []
    for batch_cnt, batch in tqdm(enumerate(loader), total=len(loader)):
        image, label = batch
        image = image.to(predict_device)
        net_out, feature = net(image)
        for k, v in net_out.items():
            net_out[k] = v.detach().cpu()
        for k, v in feature.items():
            feature[k] = v.detach().cpu()
        outputs.append(net_out)
        features.append(feature)
    record = {}
    feature = {}
    for k in outputs[0].keys():
        record[k] = torch.cat([i[k] for i in outputs])
    for k in features[0].keys():
        feature[k] = torch.cat([i[k] for i in features])
    return record, feature


def calculate_loss(config, net_out, label):
    loss = {}
    loss_sum = None
    for cfg in config:
        assert cfg.input in net_out, cfg.input
        input = net_out[cfg.input]
        target = label[cfg.target]
        loss_val = cfg.loss_type(input, target)
        loss_val *= cfg.weight
        assert cfg.name not in loss
        loss[cfg.name] = loss_val
        if loss_sum is None:
            loss_sum = loss_val
        else:
            loss_sum += loss_val
    return loss_sum, loss


def wtire_summary(loss_map, tag='train', step=None):
    for k, v in loss_map.items():
        summery_writer.add_scalar(
            f'{tag}/{k}_loss',
            v.detach().cpu().numpy(),
            global_step=step
        )


def train(config):
    n_clsaa = 6
    Learn_Swaped = False

    loader = DataLoader(
        TrainEvalDataset(
            config.dataset(split='train', **config.dataset_parameter),
            config),
        config.batch_size, True, num_workers=20)
    test_loader = DataLoader(
        TrainEvalDataset(
            config.dataset(split='test', **config.dataset_parameter),
            config),
        2, False, num_workers=20)
    net = NetModel(config.net)
    net = nn.DataParallel(net)
    unused, unused1 = net.load_state_dict(
        {(('module.base_net.'+k) if not
            k.startswith('module.base_net') else k): v
            for k, v in torch.load(config.net.pre_train).items()},
        strict=False)
    logger.info(unused)
    logger.info(unused1)
    net = net.to(device)
    optimizer = SGD(net.parameters(), 0.01, 0.9)
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, 0.95)

    storage_dict = SqliteDict(f'{config.output_dir}/dcl_snap.db')
    start_epoach = 0
    if len(storage_dict) > 0:
        kk = list(storage_dict.keys())
        # net.load_state_dict(
        #     torch.load(BytesIO(storage_dict[38])))
        net.load_state_dict(
            torch.load(BytesIO(storage_dict[kk[-1]])))
        start_epoach = int(kk[-1]) + 1
        logger.info(f'loading from epoach{start_epoach}')
    global_step = 0
    for epoach in (range(start_epoach, 500)):
        net.train()
        for batch_cnt, batch in tqdm(enumerate(loader), total=len(loader)):
            image, label = batch
            image = image.to(device)
            for k, v in label.items():
                if isinstance(v, torch.Tensor):
                    label[k] = label[k].to(device)
            optimizer.zero_grad()
            net_out = net(image)
            loss_sum, loss_map = calculate_loss(
                config.net.loss, net_out, label)
            loss_sum.backward()
            optimizer.step()
            global_step += 1
            wtire_summary(loss_map, 'train', global_step)
        exp_lr_scheduler.step(epoach)
        logger.debug(f'saving epoach {epoach}')
        buffer = BytesIO()
        torch.save(net.state_dict(), buffer)
        buffer.seek(0)
        storage_dict[epoach] = buffer.read()
        storage_dict.commit()
        test(config, net, test_loader, epoach)


class DCLCONFIG(util.bconfig.Config):
    class MetricsDef(util.bconfig.Config):
        name = util.bconfig.Value('acc')
        predicts = util.bconfig.Value('level_ce')
        ground_truth = util.bconfig.Value('label')
        func = util.bconfig.ValueMap(
            'acc',
            acc=metrics_acc, mAP=metrics_AP, mROC=metrics_mRoc,
        )

    class NetDef(util.bconfig.Config):
        class LossDef(util.bconfig.Config):
            input = util.bconfig.Value('level_ce')
            target = util.bconfig.Value('level')
            name = util.bconfig.Value('level_ce_loss')
            weight = util.bconfig.Value(1)
            loss_type = util.bconfig.ValueMap(
                'cross_entropy',
                cross_entropy=nn.CrossEntropyLoss(),
                smooth_l1_loss=nn.SmoothL1Loss(),
            )

        class OutputsDef(util.bconfig.Config):
            layer_parameters = util.bconfig.Value({})
            name_input = util.bconfig.Value('feature')
            name_output = util.bconfig.Value('level_ce')
            model = util.bconfig.ValueMap(
                'fc', fc=nn.Linear, softmax=nn.Softmax)

        basenet = util.bconfig.ValueMap('resnet', **model.models)
        net_parameters = util.bconfig.Value(dict(num_classes=6))
        pre_train = util.bconfig.Value('runs/resnet50-19c8e357.pth')
        outputs = util.bconfig.ValueList(OutputsDef)
        loss = util.bconfig.ValueList(LossDef)

    metrics = util.bconfig.ValueList(MetricsDef)
    dataset = util.bconfig.ValueMap('kaggle_dr', **avaliable_datasets)
    batch_size = util.bconfig.Value(10)
    dataset_parameter = util.bconfig.Value({})
    net = NetDef()
    output_dir = util.bconfig.Value('log/')
    input_size = util.bconfig.Value(224)
    cmd = util.bconfig.Value('train')
    config = util.bconfig.Value('configure.yaml')


def serve():
    pass


if __name__ == "__main__":
    import os
    import sys
    cfg = DCLCONFIG.build()
    cfg.parse_args()
    if os.path.exists(cfg.config):
        cfg.from_yaml(cfg.config)
        cfg.parse_args()
    if cfg.cmd == 'gen_config':
        print(cfg.dump_yaml())
        sys.exit(0)
    elif cfg.cmd == 'train':
        config = cfg
        config.from_yaml(cfg.config)
        summery_writer = SummaryWriter(logdir=f'{config.output_dir}/log')
        with open(f'{config.output_dir}/config.yaml', 'w') as f:
            f.write(config.dump_yaml())
        train(config)
