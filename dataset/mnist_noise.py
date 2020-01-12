from torch.utils.data import Dataset
import numpy as np
from os.path import join as pjoin
import numpy as np
import pickle

class MnistNoise(Dataset):
    def __init__(self, split='train'):
        self.data = pickle.load(open('runs/mnist_data_with_noise_label.pkl', 'rb'))
        if split == 'train':
            for k in self.data:
                self.data[k] = self.data[k][:60000]
        elif split == 'test':
            for k in self.data:
                self.data[k] = self.data[k][60000:]
        else:
            raise Exception("FUCK YOU!")
        self.label_keys = ['label', 'noise_label']

    def __getitem__(self, index):
        if index >= self.__len__():
            raise IndexError()
        return self.data['image'][index], {
            key:self.data[key][index] for key in self.label_keys}

    def __len__(self):
        return self.data['image'].shape[0]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import dataset.mnist
    mnist = dataset.mnist.Mnist()
    img, label = mnist[0]
    print(label)
    plt.imshow(img)
