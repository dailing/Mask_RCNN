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

from util.augment import augment_map, Compose
from util.logs import get_logger
from scipy.special import softmax
import sys
from dataset import datasets as avaliable_datasets
import argparse
from sklearn.metrics import average_precision_score, roc_auc_score
import util.bconfig
import model
from model.deeplab_v3 import CrossEntropy2d
import pickle
from itertools import chain
from model.experiment import ExperimentLoss
from os import cpu_count

logger = get_logger('fff')
device = torch.device('cuda')
num_processor = cpu_count()

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
        return values


class TrainEvalDataset(Dataset):
    def __init__(self, data_reader, config, N=7, __k=2,
                 augment=False, split='train', root=None, swap=False):
        super().__init__()
        self.config = config
        self.data_reader = data_reader
        self.N = N
        self.k = __k
        self.swap_img = swap
        # resiez_size = int(self.config.input_size*1.2)
        # if split != 'train':
        #     resiez_size = self.config.input_size
        if split == 'train':
            transform = config.train_transform
        else:
            transform = config.test_transform
        transform = [
            transform_cfg.op(**transform_cfg.parameters)
            for transform_cfg in transform
        ]
        self.transform = Compose(transform)

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


def metrics_iou(output, label):
    iou = {}
    numClasses = output.shape[1]
    prediction = np.argmax(output, axis=1)
    logger.debug(f'shape is {prediction.shape}; {label.shape}')
    for i in range(numClasses):
        intersection = float(np.sum(np.logical_and(prediction == i, label == i)))
        union = float(np.sum(np.logical_or(prediction == i, label == i)))
        logger.debug(f'intersecton {intersection}, union {union}')
        if union == 0:
            iou[f'{i}'] = 1
        else:
            iou[f'{i}'] = intersection / union
    return iou


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


def test(config, net, data_loader, epoach, loss_calculator):
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
        loss_sum, loss_map = loss_calculator(net_out, label)
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
        if isinstance(v, dict):
            summery_writer.add_scalars(
                f'metrics/{k}',
                v, epoach
            )
        elif isinstance(v, list):
            summery_writer.add_scalars(
                f'metrics/{k}',
                {i: j for i, j in enumerate(v)},
                epoach
            )
        else:
            summery_writer.add_scalar(
                f'metrics/{k}',
                v,
                epoach
            )


def predict(config):
    device = torch.device('cuda')
    loader = DataLoader(
        TrainEvalDataset(
            config.dataset(split='train', **config.dataset_parameter),
            config),
        1000, False, num_workers=num_processor)
    test_loader = DataLoader(
        TrainEvalDataset(
            config.dataset(split='test', **config.dataset_parameter),
            config),
        1000, False, num_workers=num_processor)
    net = NetModel(config.net)
    net = net.to(device)

    storage_dict = SqliteDict(f'{config.output_dir}/dcl_snap.db')
    if len(storage_dict) > 0:
        kk = list(storage_dict.keys())
        if config.predict.load_epoach == -1:
            config.predict.load_epoach = kk[-1]
        net.load_state_dict(
            torch.load(BytesIO(storage_dict[config.predict.load_epoach]), map_location=device), strict=True)
        logger.info(f'loading from epoach {config.predict.load_epoach}')

    net.eval()

    outputs = {}
    keys = ['softmax']
    if config.predict.values_to_save is not None and len(config.predict.values_to_save) > 0:
        keys = config.predict.values_to_save
    for k in keys:
        outputs[k] = []
    for batch_cnt, batch in tqdm(enumerate(chain(loader, test_loader)), total=len(loader)+len(test_loader)):
        image, label = batch
        image = image.to(device)
        net_out = net(image)
        net_out.update(label)
        out = {}
        for k in keys:
            outputs[k].append(net_out[k].detach().cpu())
    # record = {}
    for k in keys:
        outputs[k] = torch.cat(outputs[k])
    pickle.dump((outputs), open(f'{config.output_dir}/predict.pkl', 'wb'))


class LossCalculator():
    def __init__(self, config):
        self.config = config
        self.loss_instance = []
        for cfg in config:
            instance = cfg.loss_type(**cfg.loss_parameters)
            self.loss_instance.append(instance)

    def __call__(self, net_out, label):
    # def calculate_loss(config, net_out, label):
        loss = {}
        loss_sum = None
        for cfg, loss_ins in zip(self.config, self.loss_instance):
            assert cfg.input == '*' or cfg.input in net_out, f'{cfg.input} not in {list(net_out.keys())}'
            if cfg.input == '*':
                input = net_out
            else:
                input = net_out[cfg.input]
            if cfg.target == '*':
                target = label
            else:
                target = label[cfg.target]
            loss_val = loss_ins(input, target)
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
    loader = DataLoader(
        TrainEvalDataset(
            config.dataset(split='train', **config.dataset_parameter),
            config),
        config.batch_size, True, num_workers=num_processor)
    test_loader = DataLoader(
        TrainEvalDataset(
            config.dataset(split='test', **config.dataset_parameter),
            config),
        config.batch_size, False, num_workers=num_processor)
    net = NetModel(config.net)
    loss_calculator = LossCalculator(config.net.loss)
    # net = nn.DataParallel(net)
    logger.info(config.net.pre_train)
    logger.info(type(config.net.pre_train))
    if config.net.pre_train is not None and os.path.exists(config.net.pre_train):
        unused, unused1 = net.load_state_dict(
            {(('base_net.'+k) if not
                k.startswith('base_net') else k): v
                for k, v in torch.load(config.net.pre_train).items()},
            strict=False)
        logger.info(unused)
        logger.info(unused1)
    net = net.to(device)
    optimizer = SGD(net.parameters(), config.lr, 0.9, weight_decay=0.0005)
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
    for epoach in (range(start_epoach, config.max_it)):
        net.train()
        for batch_cnt, batch in tqdm(enumerate(loader), total=len(loader)):
            image, label = batch
            if isinstance(image, torch.Tensor):
                image = image.to(device)
            elif isinstance(image, dict):
                for k, v in image.items():
                    if isinstance(v, torch.Tensor):
                        image[k] = image[k].to(device)
            elif isinstance(image, list):
                for v in image:
                    v.to(device)
            for k, v in label.items():
                if isinstance(v, torch.Tensor):
                    label[k] = label[k].to(device)
            optimizer.zero_grad()
            net_out = net(image)
            loss_sum, loss_map = loss_calculator(net_out, label)
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
        test(config, net, test_loader, epoach, loss_calculator)


class DCLCONFIG(util.bconfig.Config):
    class MetricsDef(util.bconfig.Config):
        name = util.bconfig.Value('acc')
        predicts = util.bconfig.Value('level_ce')
        ground_truth = util.bconfig.Value('label')
        func = util.bconfig.ValueMap(
            'acc',
            acc=metrics_acc, mAP=metrics_AP, mROC=metrics_mRoc,
            iou=metrics_iou,
        )

    class NetDef(util.bconfig.Config):
        class LossDef(util.bconfig.Config):
            input = util.bconfig.Value('level_ce')
            target = util.bconfig.Value('level')
            name = util.bconfig.Value('level_ce_loss')
            weight = util.bconfig.Value(1)
            loss_type = util.bconfig.ValueMap(
                'cross_entropy',
                cross_entropy=nn.CrossEntropyLoss,
                smooth_l1_loss=nn.SmoothL1Loss,
                yolo=model.dark_53.YOLOLayer,
                cross_entropy2d=CrossEntropy2d,
                exp_loss=ExperimentLoss,
            )
            loss_parameters = util.bconfig.Value({})

        class OutputsDef(util.bconfig.Config):
            layer_parameters = util.bconfig.Value({})
            name_input = util.bconfig.Value('feature')
            name_output = util.bconfig.Value('level_ce')
            model = util.bconfig.ValueMap(
                'fc', fc=nn.Linear,
                softmax=nn.Softmax,
                relu=nn.ReLU,
                conv=nn.Conv2d,
                yolo=model.dark_53.YOLOLayer,
                yolo_box=model.dark_53.YOLO2Boxes,
                dropout=nn.Dropout,
            )

        basenet = util.bconfig.ValueMap('resnet', **model.models)
        net_parameters = util.bconfig.Value(dict(num_classes=6))
        pre_train = util.bconfig.Value('runs/resnet50-19c8e357.pth')
        outputs = util.bconfig.ValueList(OutputsDef)
        loss = util.bconfig.ValueList(LossDef)

    class AugmentDef(util.bconfig.Config):
        op = util.bconfig.ValueMap('None', **augment_map)
        parameters = util.bconfig.Value({})

    class PredictDef(util.bconfig.Config):
        load_epoach = util.bconfig.Value(-1)
        values_to_save = util.bconfig.ValueList(util.bconfig.Value)

    train_transform = util.bconfig.ValueList(AugmentDef)
    test_transform = util.bconfig.ValueList(AugmentDef)
    metrics = util.bconfig.ValueList(MetricsDef)
    dataset = util.bconfig.ValueMap('kaggle_dr', **avaliable_datasets)
    batch_size = util.bconfig.Value(10)
    dataset_parameter = util.bconfig.Value({})
    net = NetDef()
    output_dir = util.bconfig.Value('log/')
    # input_size = util.bconfig.Value(224)
    cmd = util.bconfig.Value('train')
    config = util.bconfig.Value('configure.yaml')
    max_it = util.bconfig.Value(14)
    lr = util.bconfig.Value(0.001)
    predict = PredictDef()


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
        logger.debug(config.dump_yaml())
        config.parse_args()
        summery_writer = SummaryWriter(logdir=f'{config.output_dir}/log')
        with open(f'{config.output_dir}/config.yaml', 'w') as f:
            f.write(config.dump_yaml())
        logger.debug(config.dump_yaml())
        train(config)
    elif cfg.cmd == 'predict':
        config = cfg
        config.from_yaml(cfg.config)
        config.parse_args()
        logger.debug(config.dump_yaml())
        predict(config)
