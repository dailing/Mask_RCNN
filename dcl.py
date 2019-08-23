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

from util.augment import ResizeKeepAspectRatio, Compose, \
    RandomNoise, RandFlip, ToTensor, ToFloat, FundusAOICrop, \
    Resize, RandRotate, RangeCenter, RandomCrop
from util.logs import get_logger
from scipy.special import softmax


logger = get_logger('fff')
summery_writer = SummaryWriter(logdir='log/dcl_log')

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

    def __init__(self, num_classes=1000,
                 groups=1, width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None,
                 with_regression=False):
        super(ResNet, self).__init__()
        layers = [3, 4, 6, 3]
        block = Bottleneck
        self.overall_stride = 32
        self.input_width = 448
        self.N = 7

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
        self.groups = groups
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

        self.fc_cls = nn.Linear(512 * block.expansion, num_classes)
        self.fc_adv = nn.Linear(512 * block.expansion, 2)
        self.with_regression = with_regression
        if self.with_regression:
            self.fc_regression = nn.Linear(512 * block.expansion, 1)
        self.fc_threshold = nn.Linear(512*block.expansion, 5)

        # construction learning
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
        results = {}

        results['cls'] = self.fc_cls(feature_vector)
        results['adv'] = self.fc_adv(feature_vector)
        construction_learning = \
            self.construction_learning_conv(feature_map)
        construction_learning = \
            self.construction_learning_relu(construction_learning)
        construction_learning = \
            self.construction_learning_pool(construction_learning)
        results['ctl'] = construction_learning
        results['reg'] = None
        results['threshold'] = self.fc_threshold(feature_vector)
        if self.with_regression:
            results['reg'] = self.fc_regression(feature_vector)
        return results


class PilLoader:
    def pil_loader(self, imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


class DRDataset(PilLoader):
    def __init__(self, split=None, root=None):
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
        self.files.columns = ('image', 'label', *self.files.columns[2:])
        # self.files.label
        # self.transform = Compose(
        #     (FundusAOICrop(), Resize(224))
        # )
        self.files_by_level = [
            self.files[self.files.label == i] for i in range(5)
        ]
        self.length = len(self.files)

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError
        if self.split == 'train':
            level = index % 5
            index_in_level = random.randint(0, len(self.files_by_level[level])-1)
            row = self.files_by_level[level].iloc[index_in_level]
        else:
            row = self.files.iloc[index]
        fname = f'/share/small_pic/kaggle_dr@448/{row.image}'
        file_content = open(fname, 'rb').read()
        if file_content is None:
            raise Exception(f'file {fname} not found')
        img = cv2.imdecode(
            np.frombuffer(file_content, np.uint8),
            cv2.IMREAD_COLOR)
        # img = self.transform(img)
        return img, row

    def __len__(self):
        return self.length


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
        return image, row

    def __len__(self):
        return len(self.image_list)


class TrainEvalDataset(Dataset):
    def __init__(self, data_reader_class, N=7, __k=2,
                 augment=False, split='train', root=None, swap=False):
        super().__init__()
        self.data_reader = data_reader_class(split=split, root=root)
        self.N = N
        self.k = __k
        self.swap_img = swap
        self.transform = [
            # Resize(224),
            # RandomCrop(448),
            ToFloat(),
            RangeCenter()
        ]
        if augment:
            self.transform += [
                # RandomNoise(),
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
        image, label = self.data_reader[index]
        image = self.transform(image)
        if self.swap_img:
            swaped_image, swap_matrix, original_matrix = self.swap(image)
            original_matrix = (original_matrix / 1).astype(np.float32)
            swap_matrix = (swap_matrix / 1).astype(np.float32)
        else:
            swaped_image, swap_matrix, original_matrix = 0, 0, 0
            original_matrix = 0
            swap_matrix = 0
        return image, original_matrix,\
            swaped_image, swap_matrix,\
            label.label

    def __len__(self):
        return self.data_reader.__len__()


def calculate_loss(
        net_out,
        label=None, matrix=None, label_adv=None,
        global_step=None,
        learn_swapped=True):
    active_cls = net_out['cls']
    active_adv = net_out['adv']
    active_reg = net_out['reg']
    active_matrix = net_out['ctl']
    activa_th = net_out['threshold']
    loss_cls = nn.functional.cross_entropy(active_cls, label)
    if learn_swapped:
        loss_adv = nn.functional.cross_entropy(active_adv, label_adv)
        loss_ctl = nn.functional.l1_loss(active_matrix, matrix)
        loss = loss_cls + loss_adv + loss_ctl
        loss_ctl = loss_ctl.detach().cpu().numpy()
        loss_adv = loss_adv.detach().cpu().numpy()
        loss_cls = loss_cls.detach().cpu().numpy()
    else:
        loss = loss_cls
        loss_adv = 0
        loss_ctl = 0
    if active_reg is not None:
        loss_reg = nn.functional.smooth_l1_loss(active_reg, label.float().view_as(active_reg))
        loss += loss_reg
        loss_reg = loss_reg.detach().cpu().numpy()
    else:
        loss_reg = 0
    if activa_th is not None:
        _, yv = torch.meshgrid([torch.arange(0,label.shape[0]), torch.arange(0, 5)])
        yv = yv.to(device)
        label_th = torch.stack([label] * 5, dim=1)
        label_th = (label_th > yv).float()
        loss_threshold = nn.functional.binary_cross_entropy_with_logits(activa_th,label_th)
    return loss, dict(loss=loss, loss_cls=loss_cls,
                      loss_adv=loss_adv, loss_ctl=loss_ctl,
                      loss_reg=loss_reg, loss_threshold=loss_threshold)


def test(net, data_loader, epoach):
    net.eval()
    global_step = epoach * len(data_loader)
    cls_predict, cls_gt = [], []
    for batch_cnt, batch in enumerate(tqdm(data_loader)):
        image, matrix, swapped_image, swap_matrix, label = batch
        label_adv = torch.LongTensor([0]*image.shape[0])
        image = (image)
        matrix = (matrix)
        label = (label)

        image = image.to(device)
        matrix = matrix.to(device)
        label = label.to(device)
        label_adv = label_adv.to(device)

        net_out = net(image)
        active_cls = net_out['cls']
        active_th = net_out['threshold']
        active_reg = net_out['reg']

        loss, loss_dict = calculate_loss(
            net_out,
            label=label,
            matrix=matrix,
            label_adv=label_adv, learn_swapped=False)

        for name, loss_val in loss_dict.items():
            summery_writer.add_scalar(
                f'test/{name}', loss_val,
                global_step=global_step
            )
        label = label.detach().cpu().numpy()
        active_cls = active_cls.detach().cpu().numpy()
        result_cls = np.argmax(active_cls, axis=1)

        # active_adv = active_adv.detach().cpu().numpy()
        # result_adv = np.argmax(active_adv, axis=1)
        # label_adv = label_adv.detach().cpu().numpy()
        active_th = active_th.detach().cpu().numpy()
        active_th = 1. / (1. + np.exp(-active_th))
        active_th = active_th > 0.5
        label_th = np.stack([label] * 5, axis=1)
        for i in range(5):
            label_th[:, i] = label_th[:, i] > i
        acc_th = np.mean(active_th == label_th, axis=0)
        summery_writer.add_scalar(
            'test/class_acc', np.mean(result_cls == label),
            global_step=global_step
        )
        cls_gt += label.tolist()
        cls_predict += result_cls.tolist()
        # summery_writer.add_scalar(
        #     'test/class_adv', np.mean(result_adv == label_adv),
        #     global_step=global_step
        # )
        if active_reg is not None:
            active_reg = np.round(active_reg.detach().cpu().numpy())
            summery_writer.add_scalar(
                'test/class_reg_acc', np.mean(active_reg == label),
                global_step=global_step
            )
        summery_writer.add_scalars(
            'test/bin_classification_acc',
            {f'acc_{i}': acc_th[i] for i in range(5)},
            global_step=global_step
        )
        global_step += 1
    cls_gt = np.array(cls_gt)
    cls_predict = np.array(cls_predict)
    summery_writer.add_scalar(
        'test/acc_all',
        np.mean(cls_gt == cls_predict),
        epoach
    )
    # summery_writer.add_scalar(
    #     'test/f1',
    #     sklearn
    # )


# noinspection PyArgumentList
def train():
    n_clsaa = 6
    Learn_Swaped = False
    with_regresion = True

    loader = DataLoader(
        TrainEvalDataset(DRDataset, split='train', augment=True, swap=True),
        10, True, num_workers=20)
    test_loader = DataLoader(
        TrainEvalDataset(DRDataset, split='test'), 10, True, num_workers=20)
    net = ResNet(num_classes=n_clsaa, with_regression=with_regresion)
    net.load_state_dict(torch.load('runs/resnet50-19c8e357.pth'), strict=False)
    net = net.to(device)
    net = nn.DataParallel(net)

    ignored_params1 = list(map(id, net.module.fc_adv.parameters()))
    ignored_params2 = list(map(id, net.module.fc_cls.parameters()))
    ignored_params3 = list(map(id, net.module.fc_regression.parameters()))
    ignored_params4 = list(
        map(id, net.module.construction_learning_conv.parameters()))
    ignored_params = ignored_params1 + ignored_params2 + ignored_params3 + ignored_params4
    base_params = filter(
        lambda p: id(p) not in ignored_params,
        net.module.parameters())

    base_lr = 0.01
    lr_ratio = 1

    def lr_policy(step):
        if step < 20:
            return 0.01 * 1.259 ** step
        return 0.3 ** (step // 200)

    learning_parametars = [
        {'params': base_params},
        {'params': net.module.fc_cls.parameters(), 'lr': lr_ratio*base_lr},
        {'params': net.module.fc_regression.parameters(), 'lr': 0.1*base_lr},
        {'params': net.module.fc_adv.parameters(), 'lr': 0.1*base_lr},
        {'params': net.module.construction_learning_conv.parameters(),
            'lr': 0.1*base_lr},
    ]
    optimizer = SGD(net.parameters(), 0.01, 0.9)
    # optimizer = Adam(learning_parametars, base_lr)
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, 0.97)

    storage_dict = SqliteDict('./log/dcl_snap.db')
    start_epoach = 0
    if len(storage_dict) > 0:
        kk = list(storage_dict.keys())
        net.load_state_dict(
            torch.load(BytesIO(storage_dict[kk[-1]])))
        start_epoach = int(kk[-1]) + 1
        logger.info(f'loading from epoach{start_epoach}')
    global_step = 0
    for epoach in (range(start_epoach, 500)):
        net.train()
        for batch_cnt, batch in tqdm(enumerate(loader), total=len(loader)):
            image, matrix, swapped_image, swap_matrix, label = batch
            if Learn_Swaped:
                label_adv = torch.LongTensor(
                    [0]*image.shape[0] + [1]*image.shape[0])
                image = torch.cat((image, swapped_image), dim=0)
                matrix = torch.cat((matrix, swap_matrix), dim=0)
                label = torch.cat((label, label), dim=0)
            else:
                label_adv = torch.LongTensor(
                    [0] * image.shape[0])

            image = image.to(device)
            label = label.to(device)
            matrix = matrix.to(device)
            label_adv = label_adv.to(device)

            optimizer.zero_grad()

            net_out = net(image)
            active_cls, active_adv, active_matrix = net_out['cls'], net_out['adv'], net_out['ctl']
            active_th = net_out['threshold']
            active_reg = net_out['reg']

            loss, loss_dict = calculate_loss(
                net_out,
                label=label,
                matrix=matrix,
                label_adv=label_adv,
                learn_swapped=Learn_Swaped)

            loss.backward()
            optimizer.step()

            for name, loss_val in loss_dict.items():
                summery_writer.add_scalar(
                    f'train/{name}', loss_val,
                    global_step=global_step
                )
            label = label.detach().cpu().numpy()
            active_cls = active_cls.detach().cpu().numpy()
            result_cls = np.argmax(active_cls, axis=1)

            active_adv = active_adv.detach().cpu().numpy()
            result_adv = np.argmax(active_adv, axis=1)
            label_adv = label_adv.detach().cpu().numpy()
            active_th = active_th.detach().cpu().numpy()
            active_th = 1. / (1. + np.exp(-active_th))
            active_th = active_th > 0.5
            label_th = np.stack([label] * 5, axis=1)
            # logger.info(label_th)
            for i in range(5):
                label_th[:,i] = label_th[:,i] > i
            # logger.info(label_th)
            # logger.info(active_th)
            # logger.info(label_th == active_th)
            acc_th = np.mean(active_th==label_th, axis=0)
            if active_reg is not None:
                active_reg = np.round(active_reg.detach().cpu().numpy())
                summery_writer.add_scalar(
                    'train/class_reg_acc', np.mean(active_reg == label),
                    global_step=global_step
                )
            summery_writer.add_scalar(
                'train/class_acc', np.mean(result_cls == label),
                global_step=global_step
            )
            summery_writer.add_scalar(
                'train/class_adv', np.mean((result_adv == label_adv)),
                global_step=global_step
            )
            summery_writer.add_scalars(
                'train/bin_classification_acc',
                {f'acc_{i}': acc_th[i] for i in range(5)},
                global_step=global_step
            )
            global_step += 1
        exp_lr_scheduler.step(epoach)
        logger.debug(f'saving epoach {epoach}')
        buffer = BytesIO()
        torch.save(net.state_dict(), buffer)
        buffer.seek(0)
        storage_dict[epoach] = buffer.read()
        storage_dict.commit()
        #test(net, test_loader, epoach)


class Predictor(MServiceInstance):
    def init_env(self):
        net = ResNet(num_classes=6, with_regression=True)
        net = nn.DataParallel(net)
        net.load_state_dict(torch.load(self.snap_file))
        self.transform = Compose((
            FundusAOICrop(),
            Resize(448),
            ToFloat(),
            RangeCenter(),
            ToTensor()
        ))
        self.net = net
        self.net.eval()

    def __call__(self, arg):
        if self.net is None:
            self.init_env()
        if type(arg) is np.ndarray:
            arg = (arg,)
        elif type(arg) in (list, tuple):
            pass
        else:
            logger.error(f'FUCK the arg is {type(arg)}')
        result = []
        for img in arg:
            img = self.transform(img)
            img = img[np.newaxis,:,:,:]
            img = torch.Tensor(img)
            net_result = self.net(img)
            rr = net_result['reg'].detach().cpu().numpy().item()
            cls = net_result['cls'].detach().cpu().numpy()
            logger.info(rr)
            logger.info(cls)
            result.append(dict(
                level=round(rr),
                raw=rr,
                prob=softmax(cls).tolist()
            ))
            return result



    def __init__(self, snap_file='snap.pkl'):
        self.transform = None
        self.snap_file = snap_file
        self.net = None



if __name__ == "__main__":
    train()
    pp = Predictor()
    pp.init_env()

