import PIL
import pandas as pd
import random
import cv2
import numpy as np


class DRDataset():
    def __init__(self, split=None, root=None):
        self.reader = None
        if split is None:
            split = 'train'
        self.split = split
        label_file = {
            'train': '../data/grade/trainLabels.csv',
            'test': '../data/grade/retinopathy_solution.csv',
        }
        self.files = pd.read_csv(label_file[split])
        self.files.columns = ('image', *self.files.columns[1:])
        self.files_by_level = [
            self.files[self.files.level == i] for i in range(5)
        ]
        self.length = len(self.files)

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError
        if self.split == 'train':
            level = index % 5
            index_in_level = random.randint(
                0, len(self.files_by_level[level])-1)
            row = self.files_by_level[level].iloc[index_in_level]
        else:
            row = self.files.iloc[index]
        fname = f'../data/grade/images@299/{row.image}'
        file_content = open(fname, 'rb').read()
        if file_content is None:
            raise Exception(f'file {fname} not found')
        img = cv2.imdecode(
            np.frombuffer(file_content, np.uint8),
            cv2.IMREAD_COLOR)
        # img = self.transform(img)
        return img, row.to_dict()

    def __len__(self):
        return self.length


if __name__ == "__main__":
    dr = DRDataset()
    xx = dr[0]
