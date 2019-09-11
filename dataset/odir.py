
import PIL
import pandas as pd
import random
import cv2
import numpy as np
import pickle
from torch.utils.data import Dataset


class ODIR():
    def __init__(self, split=None, root=None, test_split=0):
        self.reader = None
        if split is None:
            split = 'train'
        self.split = split
        self.files = pd.read_excel(
            '../data/ODIR/ODIR-5K_training-Chinese2.xlsx')
        if split == 'train':
            self.files = self.files[self.files.split != test_split]
        else:
            self.files = self.files[self.files.split == test_split]
        self.length = len(self.files)

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError
        row = self.files.iloc[index]
        fname = f'../data/ODIR/img_stack@299/{row["编号"]}.pkl'
        img = pickle.load(open(fname, 'rb'))
        return img, row.to_dict()

    def __len__(self):
        return self.length


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    dr = ODIR()
    img, label = dr[0]
    print(img.shape)
    print(label)
    dd = DataLoader(dr, 32, num_workers=os.cpu)
    for i, (img, label) in enumerate(dd):
        if i > 1000:
            break
        pass
