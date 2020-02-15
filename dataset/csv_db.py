import PIL
import pandas as pd
import random
import cv2
import numpy as np
from os.path import join as pjoin
from random import randint

class CsvDataset():
    def __init__(self, split=None, root=None):
        self.reader = None
        if split is None:
            split = 'train'
        self.root = root
        self.split = split
        label_file =  pjoin(root, f'{split}.csv')
        self.files = pd.read_csv(label_file)
        self.length = len(self.files)

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError
        row = self.files.iloc[index]
        fname = pjoin(self.root, row.image)
        file_content = open(fname, 'rb').read()
        if file_content is None:
            raise Exception(f'file {fname} not found')
        img = cv2.imdecode(
            np.frombuffer(file_content, np.uint8),
            cv2.IMREAD_COLOR)
        if img is None:
            return self.__getitem__(randint(0, self.length-1))        
        return img, row.to_dict()

    def __len__(self):
        return self.length


if __name__ == "__main__":
    dr = CsvDataset()
    xx = dr[0]
