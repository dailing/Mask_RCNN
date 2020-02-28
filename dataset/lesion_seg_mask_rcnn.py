import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle
import torch
from os.path import exists
from torch.utils.data import Dataset
from util.augment import ResizeKeepAspectRatio

def find_boxes(image, klass=0):
    if image is None:
        return {}
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (image>0).astype(np.uint8))
#     print(labels.shape, labels.max())
    stats[:,2] += stats[:,0]
    stats[:,3] += stats[:,1]
    stats[:,[0,1,2,3]] = stats[:,[1,0,3,2]]
    # stats[:,[0,1,2,3]] = stats[:,[1,3,0,2]]
    # stats[:,[0,1,2,3]] = stats[:,[0,2,1,3]]
    stats = stats[:, :-1]
    boxes = stats[1:]
    masks = np.zeros((len(boxes), *image.shape), dtype=np.uint8)
    for i in range(len(boxes)):
        masks[i, boxes[i,0]:boxes[i,2], boxes[i, 1]:boxes[i, 3]] = (labels[boxes[i,0]:boxes[i,2], boxes[i, 1]:boxes[i, 3]] == (i+1))
    boxes[:,[0,1,2,3]] = boxes[:,[1,0,3,2]]
    klass = np.array([klass] * len(boxes))
    return dict(
        boxes=torch.from_numpy(boxes.astype(np.float32)),
        masks=torch.from_numpy(masks),
        labels=torch.from_numpy(klass.astype(np.int64)))


class LesionSegMask(Dataset):
    def __init__(self, split=None, root=None):
        if root is None:
            root = '../data/challenge'
        self.images = [(
                f'{root}/all_images/IDRiD_{i:02d}.jpg', 
                f'{root}/all_images/IDRiD_{i:02d}_EX.tif',
                f'{root}/all_images/IDRiD_{i:02d}_HE.tif',
                f'{root}/all_images/IDRiD_{i:02d}_MA.tif',
                f'{root}/all_images/IDRiD_{i:02d}_OD.tif',
                f'{root}/all_images/IDRiD_{i:02d}_SE.tif',
            ) for i in range(1, 82)]
        if split == 'train':
            self.images = self.images[:60]
        elif split == 'test':
            self.images = self.images[60:]
        else:
            print("##################FUCK####################")
        self.ratio = 0.5
    
    def __getitem__(self, index):
        i = self.images[index]
        image = cv2.imread(i[0], cv2.IMREAD_COLOR)
        dsize = (int(image.shape[1] * self.ratio), int(image.shape[0] * self.ratio))
        image = cv2.resize(image, dsize=dsize)
        label = {}
        for cls, path in enumerate(i[1:]):
            if not exists(path):
                continue
            mask_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.resize(mask_img, dsize=dsize)
            res = find_boxes(mask_img, cls)
            for k, v in res.items():
                if k in label:
                    label[k] = torch.cat((label[k], v))
                else:
                    label[k] = v
        return image, label

    def __len__(self):
        return len(self.images)


# def predict()

    
if __name__ == "__main__":
    ds = LesionSegMask(split='train')
    image, label = ds[0]
    print(image.shape)
    print(label)