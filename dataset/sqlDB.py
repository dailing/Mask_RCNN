import PIL
import pandas as pd
import cv2
import numpy as np
import pickle
from torch.utils.data import Dataset
from main import ImageAnnotation, psql_db, ImageStorage
import playhouse.db_url
from util.logs import get_logger
import torch
import datetime
import random


logger = get_logger('fuck sql db')


class SqlDB(Dataset):
    def __init__(
            self,
            split='train',
            db_url='postgresql://db_user:123456@localhost:25068/fuckdb',
            table_name='imageannotation',
            max_box=2):
        self.db_url = db_url
        self.max_box = max_box

    def _init_env(self):
        if not hasattr(self, 'images'):
            if psql_db.is_closed():
                psql_db.initialize(playhouse.db_url.connect(self.db_url))
            result = ImageAnnotation.select().where(
                    (ImageAnnotation.session_name=='disk_set') and 
                    (ImageAnnotation.points[0]!='-1')).\
                dicts().\
                execute()
            result = list(result)
            # logger.info(result)
            # self.export_labels = export_labels
            self.annotations = {}
            for i in result:
                iid = i['image_id']
                if iid not in self.annotations:
                    self.annotations[iid] = {'detection': []}
                self.annotations[iid]['detection'].append( np.array((
                    iid,
                    i['class_id'],
                    *i['points']), dtype=np.float32))
            for k, v in self.annotations.items():
                assert len(v['detection']) <= self.max_box
                while len(v['detection']) < self.max_box:
                    v['detection'].append(np.array((-1,-1,-1,-1,-1,-1), np.float32))
            self.images = list(self.annotations.keys())
            
    def __getitem__(self, index):
        self._init_env()
        if index >= self.__len__():
            raise IndexError()
        img_record = ImageStorage.get_by_id(self.images[index])
        img = cv2.imdecode(np.frombuffer(img_record.payload.tobytes(), np.uint8), cv2.IMREAD_ANYCOLOR)
        anno = self.annotations[self.images[index]]
        return img, anno

    def __len__(self):
        self._init_env()
        return len(self.images)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    db = SqlDB()
    img, label = db[0]
    print(label)
    print(img.shape)
    # plt.imshow(img)
    # plt.show()
