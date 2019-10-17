import torch
from torch import nn
from util.logs import get_logger
import numpy as np


logger = get_logger('dark loss')

def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(
            in_num, out_num, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())


# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()
        reduced_channels = int(in_channels/2)
        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out


class Darknet53(nn.Module):
    def __init__(self, block=DarkResidualBlock, num_classes=1000):
        super(Darknet53, self).__init__()

        self.num_classes = num_classes

        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=8)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)
        # self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        out = self.conv5(out)
        out = self.residual_block4(out)
        out = self.conv6(out)
        out = self.residual_block5(out)

        return dict(feature=out)

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)


def darknet53(num_classes):
    return Darknet53(DarkResidualBlock, num_classes)



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
    """
        boxdef x, y, width, height
    """
    intersection = get_intersection(box1[0], box1[2], box2[0], box2[2]) * \
                   get_intersection(box1[1], box1[3], box2[1], box2[3])
    union = box1[2] * box1[3] + box2[2] * box2[3] - intersection
    return intersection / union


def bbox_wh_iou(wh1, wh2):
    # wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[:,0], wh2[:,1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


class YoloLoss(nn.Module):
    def __init__(
            self,
            anchors=[],
            num_class=1):
        super(YoloLoss, self).__init__()
        self.anchors = anchors
        self.num_class = num_class
        self.ignore_thres = 0.5
    
    def forward(self, x, target):
        target = torch.cat(target, dim=0)
        
        nBatch = x.size(0)
        nAnchor = len(self.anchors) // 2
        nClass = self.num_class
        nRow = x.size(-2)
        nCol = x.size(-1)
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        target = target.to(x.device)

        # Output tensors
        obj_mask = ByteTensor(nBatch, nAnchor, nRow, nCol).fill_(0)
        noobj_mask = ByteTensor(nBatch, nAnchor, nRow, nCol).fill_(1)
        class_mask = FloatTensor(nBatch, nAnchor, nRow, nCol).fill_(0)
        iou_scores = FloatTensor(nBatch, nAnchor, nRow, nCol).fill_(0)
        tx = FloatTensor(nBatch, nAnchor, nRow, nCol).fill_(0)
        ty = FloatTensor(nBatch, nAnchor, nRow, nCol).fill_(0)
        tw = FloatTensor(nBatch, nAnchor, nRow, nCol).fill_(0)
        th = FloatTensor(nBatch, nAnchor, nRow, nCol).fill_(0)
        tcls = FloatTensor(nBatch, nAnchor, nRow, nCol, nClass).fill_(0)

        instance_map = torch.cat([
            torch.arange(x.size(0), dtype=torch.float32) 
            for _ in range(int(target.size(0) / x.size(0)))])
        logger.info(instance_map)
        target[:,0] = instance_map
        # target is: image_index, class_index, x,y,w,h
        target = target[target[:,1]>-1, ...]
        arow, acol = x.size(2), x.size(3)
        assert x.size(1) == len(self.anchors) * (self.num_class + 5), \
            f"expected channels is: {len(self.anchors) * (self.num_class + 5)}, got {x.size(1)}"
        # x should be: batch_size, anchors, class+x,y,h,w,
        x = x.view(x.size(0), len(self.anchors), (self.num_class+5), x.size(2), x.size(3))

        logger.info(x.shape)
        logger.info(f'\n{target}')

        image_id = target[:,0].long()
        logger.info(image_id)
        class_id = target[:,1].long()
        grid_scale = FloatTensor([nRow, nCol])
        target_xy = target[:,2:4] * grid_scale
        target_wh = target[:,4:6] * grid_scale
        grid_xy = target_xy.long()
        anchors = FloatTensor(self.anchors).view(len(self.anchors)//2, 2) * grid_scale
        logger.info(f'\n{anchors}')
        logger.info(f'\n{target_wh}')


        # ious should be: nAnchor*nBox
        ious = torch.stack([bbox_wh_iou(anchor, target_wh) for anchor in anchors])
        logger.info(f'\n{ious}')
        # get best anchor for each box
        best_ious, best_n = ious.max(0)
        logger.info(best_ious)
        logger.info(best_n)

        obj_mask[image_id, best_n, grid_xy[:,0], grid_xy[:,1]] = 1
        noobj_mask[image_id, best_n, grid_xy[:,0], grid_xy[:,1]] = 0

        for i, anchor_ious in enumerate(ious.t()):
            noobj_mask[image_id[i], anchor_ious > self.ignore_thres, grid_xy[:,0], grid_xy[:,1]] = 0


        
