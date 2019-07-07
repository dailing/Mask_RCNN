import numpy as np
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from util.process_pool import run_once
from util.logs import get_logger
from PIL import Image
import torch.nn.functional as F
from abc import ABC, abstractmethod
import torch
import pickle
from os.path import join as pjoin
from os.path import exists
from os import makedirs
from hashlib import md5
import shutil
import sys
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from torch.optim import lr_scheduler
from util.process_pool import MongoFileReader

from tensorboardX import SummaryWriter
from abc import ABC, abstractmethod
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
# import torchvision.models.resnet
from util.augment import ResizeKeepAspectRatio, RandomCrop, Compose, \
    RandomNoise, RandFlip, ToTensor, ToFloat, GlobalNorm, FundusAOICrop,\
    Resize
from itertools import product
from sqlitedict import SqliteDict
from io import BytesIO
from tensorboardX import SummaryWriter
from tqdm import tqdm
# import ray
# ray.init('localhost:6379')

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


# def one_hot(labels: torch.Tensor,
#             num_classes: int,
#             device: Optional[torch.device] = None,
#             dtype: Optional[torch.dtype] = None,
#             eps: Optional[float] = 1e-6) -> torch.Tensor:
#     r"""
#     Converts an integer label 2D tensor to a one-hot 3D tensor.
#     code from :
#     https://torchgeometry.readthedocs.io/en/latest/_modules/kornia/utils/one_hot.html

#     Args:
#         labels (torch.Tensor) : tensor with labels of shape :math:`(N, H, W)`,
#                                 where N is batch siz. Each value is an integer
#                                 representing correct classification.
#         num_classes (int): number of classes in labels.
#         device (Optional[torch.device]): the desired device of returned tensor.
#          Default: if None, uses the current device for the default tensor type
#          (see torch.set_default_tensor_type()). device will be the CPU for CPU
#          tensor types and the current CUDA device for CUDA tensor types.
#         dtype (Optional[torch.dtype]): the desired data type of returned
#          tensor. Default: if None, infers data type from values.

#     Returns:
#         torch.Tensor: the labels in one hot tensor.

#     Examples::
#         >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
#         >>> kornia.losses.one_hot(labels, num_classes=3)
#         tensor([[[[1., 0.],
#                   [0., 1.]],
#                  [[0., 1.],
#                   [0., 0.]],
#                  [[0., 0.],
#                   [1., 0.]]]]
#     """
#     if not torch.is_tensor(labels):
#         raise TypeError("Input labels type is not a torch.Tensor. Got {}"
#                         .format(type(labels)))
#     if not len(labels.shape) == 3:
#         raise ValueError("Invalid depth shape, we expect BxHxW. Got: {}"
#                          .format(labels.shape))
#     if not labels.dtype == torch.int64:
#         raise ValueError(
#             "labels must be of the same dtype torch.int64. Got: {}" .format(
#                 labels.dtype))
#     if num_classes < 1:
#         raise ValueError("The number of classes must be bigger than one."
#                          " Got: {}".format(num_classes))
#     batch_size, height, width = labels.shape
#     one_hot = torch.zeros(batch_size, num_classes, height, width,
#                           device=device, dtype=dtype)
#     return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


# class FocalLoss(nn.Module):
#     r"""Criterion that computes Focal loss.
#     code from https://torchgeometry.readthedocs.io/en/latest/_modules/kornia/losses/focal.html#focal_loss

#     According to [1], the Focal loss is computed as follows:

#     .. math::

#         \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

#     where:
#        - :math:`p_t` is the model's estimated probability for each class.


#     Arguments:
#         alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
#         gamma (float): Focusing parameter :math:`\gamma >= 0`.
#         reduction (Optional[str]): Specifies the reduction to apply to the
#          output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
#          ‘mean’: the sum of the output will be divided by the number of elements
#          in the output, ‘sum’: the output will be summed. Default: ‘none’.

#     Shape:
#         - Input: :math:`(N, C, H, W)` where C = number of classes.
#         - Target: :math:`(N, H, W)` where each value is
#           :math:`0 ≤ targets[i] ≤ C−1`.

#     Examples:
#         >>> N = 5  # num_classes
#         >>> args = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
#         >>> loss = kornia.losses.FocalLoss(*args)
#         >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
#         >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
#         >>> output = loss(input, target)
#         >>> output.backward()

#     References:
#         [1] https://arxiv.org/abs/1708.02002
#     """

#     def __init__(self, alpha: float, gamma: Optional[float] = 2.0,
#                  reduction: Optional[str] = 'none') -> None:
#         super(FocalLoss, self).__init__()
#         self.alpha: float = alpha
#         self.gamma: torch.Tensor = torch.tensor(gamma)
#         self.reduction: Optional[str] = reduction
#         self.eps: float = 1e-6

#     def forward(  # type: ignore
#             self,
#             input: torch.Tensor,
#             target: torch.Tensor) -> torch.Tensor:
#         if not torch.is_tensor(input):
#             raise TypeError("Input type is not a torch.Tensor. Got {}"
#                             .format(type(input)))
#         if not len(input.shape) == 4:
#             raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
#                              .format(input.shape))
#         if not input.shape[-2:] == target.shape[-2:]:
#             raise ValueError("input and target shapes must be the same. Got: {}"
#                              .format(input.shape, input.shape))
#         if not input.device == target.device:
#             raise ValueError(
#                 "input and target must be in the same device. Got: {}" .format(
#                     input.device, target.device))
#         # compute softmax over the classes axis
#         input_soft = F.softmax(input, dim=1) + self.eps

#         # create the labels one hot tensor
#         target_one_hot = one_hot(target, num_classes=input.shape[1],
#                                  device=input.device, dtype=input.dtype)

#         # compute the actual focal loss
#         weight = torch.pow(torch.tensor(1.) - input_soft,
#                            self.gamma.to(input.dtype))
#         focal = -self.alpha * weight * torch.log(input_soft)
#         loss_tmp = torch.sum(target_one_hot * focal, dim=1)

#         if self.reduction == 'none':
#             loss = loss_tmp
#         elif self.reduction == 'mean':
#             loss = torch.mean(loss_tmp)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss_tmp)
#         else:
#             raise NotImplementedError("Invalid reduction mode: {}"
#                                       .format(self.reduction))
#         return loss


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample
        # layers downsample the input when stride != 1
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
                 zero_init_residual=False,
                 groups=1, width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
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

        # construction learning
        self.construction_learning_conv = nn.Conv2d(
            512*block.expansion, 2, kernel_size=1, stride=1)
        self.construction_learning_relu = nn.ReLU()
        self.construction_learning_pool = \
            nn.AdaptiveAvgPool2d((self.N, self.N))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3%
        # according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                # elif isinstance(m, BasicBlock):
                #     nn.init.constant_(m.bn2.weight, 0)

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
        active_cls = self.fc_cls(feature_vector)
        active_adv = self.fc_adv(feature_vector)

        construction_learning = \
            self.construction_learning_conv(feature_map)
        construction_learning = \
            self.construction_learning_relu(construction_learning)
        construction_learning = \
            self.construction_learning_pool(construction_learning)
        return active_cls, active_adv, construction_learning


class PilLoader():
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
        self.files.label += 1
        self.transform = Compose(
            (FundusAOICrop(), Resize(512))
        )

    def __getitem__(self, index):
        if self.reader is None:
            self.reader = SqliteDict('dataset/kaggle.db')
        if index >= len(self.files):
            raise IndexError
        row = self.files.iloc[index]
        fname = row.image
        # if fname in self.cache:
        #     return self.cache[fname]
        file_content = self.reader[fname]
        if file_content is None:
            raise Exception(f'file {fname} not found')
        # file_content = BytesIO(file_content)
        img = cv2.imdecode(
            np.frombuffer(file_content, np.uint8),
            cv2.IMREAD_COLOR)
        img = self.transform(img)
        # self.cache[fname] = (img, row)
        # self.cache.commit()
        return img, row

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
                 augment=False, split='train', root=None):
        super().__init__()
        self.data_reader = data_reader_class(split=split, root=root)
        self.N = N
        self.k = __k
        self.transform = [
            ResizeKeepAspectRatio(512),
            RandomCrop(448),
            ToFloat(),
        ]
        if augment:
            self.transform += [
                RandomNoise(),
                RandFlip(),
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
        swaped_image, swap_matrix, original_matrix = self.swap(image)
        original_matrix = (original_matrix / 1).astype(np.float32)
        swap_matrix = (swap_matrix / 1).astype(np.float32)
        return image, original_matrix,\
            swaped_image, swap_matrix,\
            label.label - 1

    def __len__(self):
        return self.data_reader.__len__()


def calculate_loss(
        matrix, label, label_adv,
        active_cls, active_adv, active_matrix, global_step,
        test=False):
    if test:
        ratio = 1
    else:
        # ratio = 1
        ratio = (1-np.exp(-global_step * 0.001))
        # ratio = (1/(1 + np.exp(-global_step / 4000 * 20 + 10)))
    loss_cls = 1.0 * ratio * \
        nn.functional.cross_entropy(active_cls, label)
    loss_adv = 1.0 * ratio * \
        nn.functional.cross_entropy(active_adv, label_adv)
    loss_ctl = 1.0 * ratio * \
        nn.functional.l1_loss(active_matrix, matrix)
    # loss = loss_cls
    loss = loss_cls + loss_adv + loss_ctl
    return loss, dict(loss=loss, loss_cls=loss_cls,
                      loss_adv=loss_adv, loss_ctl=loss_ctl)


def calculate_matrix(
        matrix, label, label_adv,
        active_cls, active_adv, active_matrix):
    pass


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

        active_cls, active_adv, active_matrix = net(image)
        # loss_cls = 1.0 * nn.functional.cross_entropy(active_cls, label)
        # loss_adv = 1.0 * nn.functional.cross_entropy(active_adv, label_adv)
        # loss_ctl = 1.0 * nn.functional.l1_loss(active_matrix,
        #                                        swap_matrix)
        # loss = loss_cls
        # loss = loss_cls + loss_adv + loss_ctl
        loss, loss_dict = calculate_loss(matrix, label, label_adv,
                                         active_cls, active_adv, active_matrix,
                                         global_step, test=True)
        for name, loss_val in loss_dict.items():
            summery_writer.add_scalar(
                f'test/{name}', loss_val.detach().cpu().numpy(),
                global_step=global_step
            )
        label = label.detach().cpu().numpy()
        active_cls = active_cls.detach().cpu().numpy()
        result_cls = np.argmax(active_cls, axis=1)
        # logger.debug(result_cls)
        # logger.debug(label)

        active_adv = active_adv.detach().cpu().numpy()
        result_adv = np.argmax(active_adv, axis=1)
        label_adv = label_adv.detach().cpu().numpy()
        # logger.debug(result_adv)
        # logger.debug(label_adv)
        # logger.debug(active_matrix)
        # logger.debug(matrix)
        summery_writer.add_scalar(
            'test/class_acc', np.mean(result_cls == label),
            global_step=global_step
        )
        cls_gt += label.tolist()
        cls_predict += result_cls.tolist()
        summery_writer.add_scalar(
            'test/class_adv', np.mean(result_adv == label_adv),
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


def train():
    n_clsaa = 5
    Learn_Swaped = True

    data = TrainEvalDataset(DRDataset, augment=True)
    loader = DataLoader(data, 2, True, num_workers=12)
    test_loader = DataLoader(
        TrainEvalDataset(
            DRDataset,
            augment=False, split='test'),
        2, True, num_workers=12)
    net = ResNet(num_classes=n_clsaa)
    net = net.to(device)
    net = nn.DataParallel(net)

    ignored_params1 = list(map(id, net.module.fc_adv.parameters()))
    ignored_params2 = list(map(id, net.module.fc_cls.parameters()))
    ignored_params3 = list(
        map(id, net.module.construction_learning_conv.parameters()))
    ignored_params = ignored_params1 + ignored_params2 + ignored_params3
    base_params = filter(
        lambda p: id(p) not in ignored_params,
        net.module.parameters())

    base_lr = 0.01
    lr_ratio = 10

    def lr_policy(step):
        if step < 20:
            return 0.01 * 1.259 ** step
        return 0.3 ** (step // 200)

    learning_parametars = [
        {'params': base_params},
        {'params': net.module.fc_adv.parameters(), 'lr': 0.1*base_lr},
        {'params': net.module.fc_cls.parameters(), 'lr': lr_ratio*base_lr},
        {'params': net.module.construction_learning_conv.parameters(),
            'lr': 0.1*base_lr},
    ]
    # optimizer = SGD(net.parameters(), 0.01, 0.9, weight_decay=0.01)
    optimizer = Adam(learning_parametars, base_lr)
    exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer, lr_policy)

    storage_dict = SqliteDict('./log/dcl_snap.db')
    start_epoach = 0
    if len(storage_dict) > 0:
        kk = list(storage_dict.keys())
        net.load_state_dict(
            torch.load(BytesIO(storage_dict[kk[-1]])))
        start_epoach = int(kk[-1]) + 1
        logger.info(f'loading from epoach{start_epoach}')
    global_step = 0
    for epoach in tqdm(range(start_epoach, 500), total=500):
        net.train()
        for batch_cnt, batch in tqdm(enumerate(loader), total=len(loader)):
            image, matrix, swapped_image, swap_matrix, label = batch
            logger.info(label)
            if Learn_Swaped:
                label_adv = torch.LongTensor(
                    [0]*image.shape[0] + [1]*image.shape[0])
                image = torch.cat((image, swapped_image), dim=0)
                matrix = torch.cat((matrix, swap_matrix), dim=0)
                label = torch.cat((label, label), dim=0)
            else:
                label_adv = torch.LongTensor(
                    [0]*image.shape[0])

            image = image.to(device)
            matrix = matrix.to(device)
            label = label.to(device)
            label_adv = label_adv.to(device)

            optimizer.zero_grad()
            active_cls, active_adv, active_matrix = net(image)
            loss, loss_dict = \
                calculate_loss(matrix, label, label_adv,
                               active_cls, active_adv, active_matrix,
                               global_step)

            loss.backward()
            optimizer.step()

            for name, loss_val in loss_dict.items():
                summery_writer.add_scalar(
                    f'train/{name}', loss_val.detach().cpu().numpy(),
                    global_step=global_step
                )
            label = label.detach().cpu().numpy()
            active_cls = active_cls.detach().cpu().numpy()
            result_cls = np.argmax(active_cls, axis=1)
            # logger.debug(result_cls)
            # logger.debug(label)

            active_adv = active_adv.detach().cpu().numpy()
            result_adv = np.argmax(active_adv, axis=1)
            label_adv = label_adv.detach().cpu().numpy()
            # logger.debug(result_adv)
            # logger.debug(label_adv)
            # logger.debug(active_matrix)
            # logger.debug(matrix)
            summery_writer.add_scalar(
                'train/class_acc', np.mean(result_cls == label),
                global_step=global_step
            )
            summery_writer.add_scalar(
                'train/class_adv', np.mean((result_adv == label_adv)),
                global_step=global_step
            )
            # image = image.detach().cpu().numpy()
            # figure = plt.figure()
            # plt.imshow(image[-1,0,:,:])
            # plt.colorbar()
            # summery_writer.add_figure('train/image', figure, global_step)
            # plt.close(figure)
            # active_matrix = active_matrix.detach().cpu().numpy()
            # matrix = matrix.detach().cpu().numpy()
            # figure = plt.figure()
            # plt.imshow(active_matrix[-1, 0, :, :])
            # plt.colorbar()
            # summery_writer.add_figure('train/result_ctl', figure, global_step)
            # plt.close(figure)
            # figure = plt.figure()
            # plt.imshow(matrix[-1, 0, :, :])
            # plt.colorbar()
            # summery_writer.add_figure('train/label_ctl', figure, global_step)
            # plt.close(figure)
            global_step += 1
        exp_lr_scheduler.step()
        logger.debug(f'saving epoach {epoach}')
        buffer = BytesIO()
        torch.save(net.state_dict(), buffer)
        buffer.seek(0)
        storage_dict[epoach] = buffer.read()
        storage_dict.commit()
        test(net, test_loader, epoach)

if __name__ == "__main__":
    train()

# ss = TrainEvalDataset(CUBBirdDataset)

# image, original_matrix, swaped_image, swap_matrix, label = ss[4]
# print(label)
# plt.figure('matrix')
# plt.imshow(swap_matrix[0,:,:])
# plt.colorbar()
# plt.figure()
# plt.imshow(swaped_image[0,:,:])
# plt.show()

# img, row = ss[5]
# print(img.dtype)
# resize = ResizeKeepAspectRatio(512)
# crop = RandomCrop(448)
# img = resize(img)
# print(img.shape)
# plt.figure()
# plt.imshow(img)

# for i in range(4):
#     plt.figure()
#     plt.imshow(crop(img))

# plt.show()

# random_sample = np.random.rand(1,3,448,448).astype(np.float32)
# random_sample = torch.Tensor(random_sample)
# xx = ResNet(num_classes=10)
# out = xx(random_sample)
# print(out)
