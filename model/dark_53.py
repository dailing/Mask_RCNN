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
        logger.info(f'{x.mean()}, {x.max()}, {x.min()}')
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
        logger.info(f'{out.mean()}, {out.max()}, {out.min()}')

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
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
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


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output



def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = (pred_boxes.size(2), pred_boxes.size(3))

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG[0], nG[1]).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG[0], nG[1]).fill_(1)
    class_mask = FloatTensor(nB, nA, nG[0], nG[1]).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG[0], nG[1]).fill_(0)
    tx = FloatTensor(nB, nA, nG[0], nG[1]).fill_(0)
    ty = FloatTensor(nB, nA, nG[0], nG[1]).fill_(0)
    tw = FloatTensor(nB, nA, nG[0], nG[1]).fill_(0)
    th = FloatTensor(nB, nA, nG[0], nG[1]).fill_(0)
    tcls = FloatTensor(nB, nA, nG[0], nG[1], nC).fill_(0)

    # Convert to position relative to box
    target = torch.cat(target, dim=0)
    target = target.to(pred_boxes.device)
    instance_map = torch.cat([
        torch.arange(nB, dtype=torch.float32) 
        for _ in range(int(target.size(0) / nB))])
    target[:,0] = instance_map
    target = target[target[:,1]>-1, ...]

    logger.info(target)
    target_boxes = target[:, 2:6] * FloatTensor(nG+nG)
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gj, gi = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    logger.info(f'{ious.shape}, {gi.shape}, {gi}, {gj}')
    for i, anchor_ious in enumerate(ious.t()):
        # logger.info(anchor_ious.shape)
        # logger.info(ignore_thres)
        # logger.info(noobj_mask.shape)
        # logger.info(anchor_ious > torch.FloatTensor(ignore_thres))
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


def to_cpu(tensor):
    return tensor.detach().cpu()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, stride=32):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.anchors = [ (anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2) ]
        self.num_anchors = len(self.anchors)
        logger.info((self.num_anchors))
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 1
        self.metrics = {}
        self.stride = stride
        self.grid_size = (-1, -1)  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        # self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g[0]).repeat(g[1], 1).view([1, 1, g[0], g[1]]).type(FloatTensor)
        self.grid_y = torch.arange(g[1]).repeat(g[0], 1).t().contiguous().view([1, 1, g[0], g[1]]).type(FloatTensor)

        self.scaled_anchors = FloatTensor([(a_w * self.grid_size[0], a_h * self.grid_size[1]) for a_w, a_h in self.anchors])
        logger.info(f'\n{self.scaled_anchors}')
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = (x.size(2), x.size(3))

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, int(grid_size[0]), int(grid_size[1]))
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if not (grid_size == self.grid_size):
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
        )

        if targets is None:
            return output
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            logger.info(tw[obj_mask])
            logger.info(w[obj_mask])

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            # logger.info(pred_conf)
            # logger.info(f'{pred_conf[obj_mask]}, {tconf[obj_mask]}')
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }
            logger.info(self.metrics)

            return total_loss


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
        target = target.to(x.device)
        instance_map = torch.cat([
            torch.arange(nBatch, dtype=torch.float32) 
            for _ in range(int(target.size(0) / nBatch))])
        target[:,0] = instance_map
        target = target[target[:,1]>-1, ...]

        nBatch = x.size(0)
        nAnchor = len(self.anchors) // 2
        nClass = self.num_class
        nRow = x.size(-2)
        nCol = x.size(-1)
        assert x.size(1) == len(self.anchors) * (self.num_class + 5), \
            f"expected channels is: {len(self.anchors) * (self.num_class + 5)}, got {x.size(1)}"
        # x should be: batch_size, anchors, class+x,y,h,w,
        x = x.view(x.size(0), len(self.anchors), x.size(2), x.size(3), (self.num_class+5))

        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor

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

        logger.info(instance_map)
        # target[:,0] = instance_map
        instance_map = instance_map.long()
        # target is: image_index, class_index, x,y,w,h
        # arow, acol = x.size(2), x.size(3)


        logger.info(x.shape)
        logger.info(f'\n{target}')

        image_id = target[:,0].long()
        logger.info(image_id)
        class_id = target[:,1].long()
        grid_scale = FloatTensor([nRow, nCol])
        target_xy = target[:,2:4] * grid_scale
        target_wh = target[:,4:6] * grid_scale

        gx, gy = target_xy.t()
        gw, gh = target_wh.t()
        gi, gj = target_xy.long().t()


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

        logger.info(grid_scale)
        logger.info(grid_xy)
        obj_mask[image_id, best_n, gi, gj] = 1
        noobj_mask[image_id, best_n, gi, gj] = 0

        # logger.info(f'\n{obj_mask}')

        for i, anchor_ious in enumerate(ious.t()):
            noobj_mask[image_id[i], anchor_ious > self.ignore_thres, gi, gj] = 0

        tx[instance_map, best_n, gi, gj] = gx - gx.floor()
        ty[instance_map, best_n, gi, gj] = gy - gy.floor()

        tw[instance_map, best_n, gi, gj] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
        th[instance_map, best_n, gi, gj] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)

        tcls[instance_map, best_n, gi, gj, class_id] = 1
        logger.info(gi)
        logger.info(x.shape)
        logger.info(x[instance_map, best_n, gi, gj].shape)
        x[instance_map, best_n, gi, gj] 
        logger.info(iou_scores[instance_map, best_n, gi, gj].shape) 
        target[2:6]
        iou_scores[instance_map, best_n, gi, gj] = bbox_iou(x[instance_map, best_n, gi, gj, class_id], target[2:6], x1y1x2y2=False)

        return noobj_mask, obj_mask, tcls


