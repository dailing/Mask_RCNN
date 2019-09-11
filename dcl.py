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
    Resize, RandRotate, RangeCenter, RandomCrop
from util.logs import get_logger
from scipy.special import softmax
import sys
from model.pasnet import PNASNet5Large
from model.inceptionv4 import InceptionV4
from dataset import datasets as avaliable_datasets
import argparse
from sklearn.metrics import average_precision_score, roc_auc_score


logger = get_logger('fff')
device = torch.device('cuda')


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            num_classes=6,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
            with_regression=False):

        super(ResNet, self).__init__()
        layers = [3, 4, 6, 3]
        block = Bottleneck
        self.overall_stride = 32
        self.input_width = 448
        self.N = 7
        self.with_regression = with_regression
        self.groups = groups

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(f"replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got"
                             " {replace_stride_with_dilation}")
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2,
            padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.construction_learning_conv = nn.Conv2d(
            512*block.expansion, 2, kernel_size=1, stride=1)
        self.construction_learning_relu = nn.ReLU()
        self.construction_learning_pool = \
            nn.AdaptiveAvgPool2d((self.N, self.N))

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(
            self.inplanes, planes, stride, downsample, self.groups,
            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation,
                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feature_map = x
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        feature_vector = x
        results = dict(feature_map=feature_map, feature=feature_vector)

        return results


class NetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.net_parameters is None:
            config.net_parameters = {}
        self.base_net = config.basenet(**config.net_parameters)
        self.config = config
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
        return outputs


class PilLoader:
    def pil_loader(self, imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


class DRDatasetRemote(PilLoader):
    def __init__(self, split=None):
        self.reader = None
        if split is None:
            split = 'train'
        self.split = split
        label_file = {
            'train': 'dataset/grade/trainLabels.csv',
            'test': 'dataset/grade/retinopathy_solution.csv',
        }
        # self.cache = SqliteDict('log/drcache.db')
        self.files = pd.read_csv(label_file[split])
        # self.files.columns = ('image', 'label', *self.files.columns[2:])
        self.remote_task = None

    def __getitem__(self, index):
        if self.remote_task is None:
            self.remote_task = Task(
                task_name='get_file',
                redis_host='192.168.3.40',
                redis_port=16379,
            )
        if index >= len(self.files):
            raise IndexError
        row = self.files.iloc[index]
        resp = self.remote_task.issue(row)
        # if fname in self.cache:
        #     return self.cache[fname]
        # self.cache[fname] = (img, row)
        # self.cache.commit()
        return resp.get()

    def __len__(self):
        return len(self.files)


class CUBBirdDataset(PilLoader):
    def __init__(self, split=None, root=None):
        if root is None:
            root = 'dataset/CUB_200_2011'
        image_list = pd.read_csv(
            pjoin(root, 'images.txt'),
            sep=' ',
            header=None,
            names=['image_id', 'image_name'],
            index_col=0)
        image_list = image_list.join(pd.read_csv(
            pjoin(root, 'train_test_split.txt'),
            sep=' ',
            header=None,
            names=['image_id', 'split'],
            index_col=0))
        image_list = image_list.join(pd.read_csv(
            pjoin(root, 'image_class_labels.txt'),
            sep=' ',
            header=None,
            names=['image_id', 'label'],
            index_col=0))
        if split is None:
            split = 'train'
        if split == 'train':
            image_list = image_list[image_list.split == 1]
        else:
            image_list = image_list[image_list.split == 0]
        self.image_list = image_list
        self.root = root

    def __getitem__(self, index):
        if index >= len(self.image_list):
            raise IndexError
        row = self.image_list.iloc[index]
        fname = pjoin(self.root, 'images', row.image_name)
        image = cv2.imread(fname, cv2.IMREAD_COLOR)
        if image is None:
            raise Exception(f'cannot read file {fname}')
        return image, row.to_dict()

    def __len__(self):
        return len(self.image_list)


class RubbishDataset(PilLoader):
    def __init__(
            self, split=None, root=None,
            test_split=None, csv_file='label_with_split.csv'):
        if root is None:
            root = '../dataset'
        image_list = pd.read_csv(
            pjoin(root, f'{root}/{csv_file}'))
        image_list.sample()
        if split is None:
            split = 'train'
        if split == 'train':
            image_list = image_list[image_list.split != test_split]
        else:
            image_list = image_list[image_list.split == test_split]
        self.image_list = image_list
        self.root = root

    def __getitem__(self, index):
        if index >= len(self.image_list):
            raise IndexError
        row = self.image_list.iloc[index]
        # logger.info(row)
        fname = pjoin(self.root, row.image)
        image = np.array(Image.open(open(fname, 'rb')))
        if len(image.shape) < 2:
            return self.__getitem__(index-1)
        if len(image.shape) != 3:
            image = np.stack((image, image, image), axis=2)
        image = image[:, :, :3]
        if image is None:
            raise Exception(f'cannot read file {fname}')
        return image, row.to_dict()

    def __len__(self):
        return len(self.image_list)


class TrainEvalDataset(Dataset):
    def __init__(self, data_reader, config, N=7, __k=2,
                 augment=False, split='train', root=None, swap=False):
        super().__init__()
        self.config = config
        self.data_reader = data_reader
        self.N = N
        self.k = __k
        self.swap_img = swap
        self.transform = [
            ToFloat(),
            Resize(int(self.config.input_size)),
            # RandomCrop(self.config.input_size),
            RangeCenter()
        ]
        if split == 'train':
            self.transform += [
                RandomNoise(),
                RandFlip(),
                RandRotate(),
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
        config.batch_size, False, num_workers=20)
    net = NetModel(config.net)
    net = nn.DataParallel(net)
    unused = net.load_state_dict(
        {(('module.base_net.'+k) if not
            k.startswith('module.base_net') else k): v
            for k, v in torch.load(config.net.pre_train).items()},
        strict=False)
    logger.info(unused)
    net = net.to(device)
    optimizer = SGD(net.parameters(), 0.001, 0.9)
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, 0.97)

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


def get_configure():
    cfg = Configure()
    cfg.\
        add_mapping('dataset', avaliable_datasets, default_value='kaggle_dr').\
        add_multi(batch_size=dict(default_value=10)).\
        add('dataset_parameter', default_value=dict())
    cfg.add_subconfigure('net').\
        add_mapping(
            'basenet',
            default_value='res_net',
            name_mapping=dict(
                inception4=InceptionV4,
                res_net=ResNet,
                pans=PNASNet5Large)).\
        add('net_parameters', default_value=dict(
            num_classes=6,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
            with_regression=False)).\
        add_multi(
            pre_train='runs/resnet50-19c8e357.pth'
        )
    cfg.net.add_list(
        'outputs',
        lambda: Configure().
        add_multi(
            layer_parameters=dict(default_value=dict(
                in_features=2048,
                out_features=10,
                bias=True
            )),
            name_input=dict(default_value='feature'),
            name_output=dict(default_value='level_ce')).
        add_mapping(
            'model',
            default_value='fc',
            name_mapping=dict(
                fc=nn.Linear,
                softmax=nn.Softmax,
            ))
    ).add_list(
        'loss',
        lambda: Configure().
        add_multi(
            input='level_ce',
            target='level',
            name='level_ce_loss',
            weight=1,
        ).
        add_mapping('loss_type', name_mapping=dict(
            cross_entropy=nn.CrossEntropyLoss(),
            smooth_l1_loss=nn.SmoothL1Loss()
        ), default_value='cross_entropy')
    )
    cfg.add_list(
        'metrics',
        lambda: Configure().add_multi(
            name='acc',
            predicts='level_ce',
            ground_truth='label',
        ).add_mapping('func', name_mapping=dict(
            acc=metrics_acc,
            mAP=metrics_AP,
            mROC=metrics_mRoc,
        ), default_value='acc')
    )
    cfg.add_multi(
        output_dir='log/',
        input_size=224,
    )
    return cfg


def serve():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('cmd', type=str, choices=['train', 'gen_config'])
    parser.add_argument(
        '-config', type=str, required=False, help='configure file')
    args = parser.parse_args()

    if args.cmd == 'gen_config':
        print(get_configure().make_sample_yaml())
        sys.exit(0)
    elif args.cmd == 'train':
        config = get_configure()
        config.from_yaml(args.config)
        # logger.info(config.to_yaml())
        summery_writer = SummaryWriter(logdir=f'{config.output_dir}/log')
        # net = NetModel(config.net)
        # xx = np.random.rand(1, 3, 448, 448).astype(np.float32)
        # xx = torch.from_numpy(xx)
        # out = net(xx)
        # logger.info(out)
        train(config)
        # pp = Predictor()
        # pp.init_env()
