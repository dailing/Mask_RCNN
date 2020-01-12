import cv2
import numpy as np
from PIL import Image


class DRIVE():
    def __init__(self, split=None, root=None, test_split=0):
        if root is None:
            root = '/dataset/DRIVE'
        self.reader = None
        if split is None:
            split = 'train'
        self.split = split

        if split == 'train':
            self.files = [(
                f'{root}/training/images/{ii}_training.tif',
                f'{root}/training/1st_manual/{ii}_manual1.gif')
                for ii in range(21, 41)]
        else:
            self.files = [(
                f'{root}/test/images/{ii:02d}_test.tif',
                f'{root}/test/1st_manual/{ii:02d}_manual1.gif')
                for ii in range(1, 21)]
        self.length = len(self.files)
        self.images = [(
            cv2.imread(self.files[index][0]),
            (np.array(Image.open(open(self.files[index][1], 'rb'))) / 255 ).astype(np.int64),
        ) for index in range(self.length)]

    def __getitem__(self, index):
        img1, label = self.images[index]
        return img1, dict(vessel=label)

    def __len__(self):
        return self.length


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dd = DRIVE()
    dl = DataLoader(dd, batch_size=3)
    print(dd[0][1]['vessel'].shape)
    print(dd[0][0].shape)
    for img, label in dl:
        print(img.shape)
        print(label['vessel'].min())
        break
