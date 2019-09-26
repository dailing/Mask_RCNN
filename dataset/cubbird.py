from PIL import Image
import PIL
import pandas as pd
import random
import cv2
import numpy as np
from os.path import join as pjoin


class PilLoader:
    def pil_loader(self, imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


class CUBBirdDataset(PilLoader):
    def __init__(self, split=None, root=None):
        if root is None:
            root = '../data/CUB_200_2011'
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


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    for i in CUBBirdDataset():
        pass
