from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
# import nvidia.dali.ops as ops
# import nvidia.dali.types as types
# from nvidia.dali.pipeline import Pipeline
from torch.utils.data._utils.collate import default_collate
from util.logs import get_logger
import torch
# from nvidia.dali.plugin.pytorch import DALIGenericIterator
from tqdm import tqdm


logger = get_logger('annotate')


# class ExternalInputIterator(object):
#     def __init__(
#             self, batch_size, split='train', test_split=0,
#             needed_labels=[]):
#         self.split = 'train' if split is None else split
#         self.root = '../data/annotation4'
#         files = pd.read_csv(f'{self.root}/dataset.csv')
#         if self.split == 'train':
#             self.image_list = files[files.split != test_split]
#         else:
#             self.image_list = files[files.split == test_split]
#         self.batch_size = batch_size

#     def __iter__(self):
#         self.i = 0
#         self.n = len(self.image_list)
#         return self

#     def __next__(self):
#         batch = []
#         labels = []
#         for _ in range(self.batch_size):
#             row = self.image_list.iloc[self.i]
#             label = []
#             image = np.frombuffer(open(
#                 f'{self.root}/images@299/{row.path}', 'rb').read(),
#                 dtype=np.uint8)
#             batch.append(image)

#             for i in ['lesionHa', 'lesionSex', 'lesionHex', 'lesionMa']:
#                 val = f'{row[i]:04b}'[::-1]
#                 for j in range(4):
#                     row[f'{i}_{j}'] = int(val[j])
#                     label.append(int(val[j]))
#             labels.append(np.array(label))
#             self.i = (self.i + 1) % self.n

#         print(len(batch))
#         return (batch, labels)

#     next = __next__


# class AnnoPipeline(Pipeline):
#     def __init__(self, batch_size, num_threads, device_id):
#         super().__init__(
#             batch_size, num_threads, device_id, seed=12)
#         self.input = ops.ExternalSource()
#         self.input_label = ops.ExternalSource()
#         self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
#         self.cast = ops.Cast(device="gpu", dtype=types.FLOAT)
#         self.eii = ExternalInputIterator(batch_size)
#         self.iterator = iter(self.eii)

#     def define_graph(self):
#         self.jpegs = self.input()
#         self.labels = self.input_label()
#         images = self.decode(self.jpegs)
#         images = self.cast(images)
#         return (images, self.labels)

#     def iter_setup(self):
#         (images, labels) = self.iterator.next()
#         # logger.info(labels)
#         self.feed_input(self.jpegs, images)
#         self.feed_input(self.labels, labels)


class OldAnno(Dataset):
    def __init__(self, split=None, root=None, test_split=0):
        self.split = 'train' if split is None else split
        self.root = '../data/annotation4' if root is None else root
        files = pd.read_csv(f'{self.root}/dataset.csv')
        if self.split == 'train':
            self.image_list = files[files.split != test_split]
        else:
            self.image_list = files[files.split == test_split]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        row = self.image_list.iloc[index]
        image = np.array(Image.open(open(
            f'{self.root}/images@299/{row.image}', 'rb')))
        row = row.to_dict()
        for i in ['lesionHa', 'lesionSex', 'lesionHex', 'lesionMa']:
            val = f'{row[i]:04b}'[::-1]
            for j in range(4):
                row[f'{i}_{j}'] = int(val[j])
        return image, row


if __name__ == "__main__":
    # anno = AnnoPipeline(batch_size=3, num_threads=12, device_id=0)
    # anno.build()
    # iiter = DALIGenericIterator(anno, ['data', 'label'], 9999999999)
    # out = anno.run()
    # logger.info(torch.(out))

    for data in tqdm(DataLoader(OldAnno(), 32, True, num_workers=12)):
        # logger.info(data[0]['data'].shape)
        pass
