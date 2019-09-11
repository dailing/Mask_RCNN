from flask import Flask, request
from util.logs import get_logger
import json
from base64 import b64decode
import cv2
import numpy as np
from dcl import get_configure, TrainEvalDataset
from util.tasks import Task
from util.augment import ImageReader
from torch.utils.data import Dataset, DataLoader
from util.tasks import MServiceInstance
from dcl import NetModel
import torch.nn as nn
import torch
from PIL import Image


class DummyData(Dataset):
    """
        use this with dataloader only when
        num_worker = 0
    """
    def __init__(self):
        super().__init__()
        self.items = []

    def __getitem__(self, index):
        if len(self.items) <= 0:
            raise IndexError()
        val = self.items[0]
        self.items = self.items[1:]
        return val, dict()

    def set_data(self, data):
        self.items = data

    def __len__(self):
        return 99999


class FlaskConfig(object):
    """Base config, uses staging database server."""
    DEBUG = False
    TESTING = False
    REDIS_SERVER_ADDRESS = 'localhost'
    REDIS_SERVER_PORT = 6379

    @property
    def DATABASE_URI(self):         # Note: all caps
        return 'mysql://user@{}/foo'


class Predictor(MServiceInstance):
    def init_env(self):
        net = NetModel(self.config.net)
        net = nn.DataParallel(net)
        net.load_state_dict(torch.load(self.snap_file, map_location='cpu'))
        self.o_dataset = DummyData()
        self.dataset = TrainEvalDataset(self.o_dataset, self.config)
        self.dataloader = DataLoader(
            self.dataset, num_workers=0, shuffle=False, batch_size=4)
        self.dataloader_iter = self.dataloader.__iter__()
        self.net = net
        self.net.eval()
        self.reader = ImageReader()

    def _read(self, fname):
        image = np.array(Image.open(open(fname, 'rb')))
        if len(image.shape) < 2:
            return self.__getitem__(index-1)
        if len(image.shape) != 3:
            image = np.stack((image, image, image), axis=2)
        image = image[:, :, :3]
        return image

    def __call__(self, arg):
        if self.net is None:
            self.init_env()
        if type(arg) is np.ndarray:
            arg = (arg,)
        elif type(arg) in (list, tuple):
            arg = [self.reader(x) for x in arg]
        elif type(arg) is str:
            arg = (self._read(arg),)
        else:
            logger.error(f'FUCK the arg is {type(arg)}')
        result = []
        for img in arg:
            images = []
            i_mid = img.shape[0] // 2
            j_mid = img.shape[1] // 2
            images.append(img[:i_mid, :j_mid, :])
            images.append(img[:i_mid, j_mid:, :])
            images.append(img[i_mid:, :j_mid, :])
            images.append(img[i_mid:, j_mid:, :])
            self.o_dataset.set_data(images)
            img, _ = self.dataloader_iter.__next__()
            logger.info(img.shape)
            net_result = self.net(img)
            for k in net_result.keys():
                net_result[k] = nn.functional.softmax(net_result[k], 1)[:, 1].\
                    detach().cpu().numpy().tolist()
            logger.info(net_result)
            result.append(net_result)
        return result

    def __init__(self, config):
        self.config = config
        self.transform = None
        self.snap_file = config.net.pre_train
        self.net = None


def process_data(data):
    if type(data) is dict:
        iter_obj = data.items()
    elif type(data) is list:
        iter_obj = enumerate(data)
    else:
        logger.info(type(data))
        return

    for k, v in iter_obj:
        if type(v) is str and len(v) > 100:
            try:
                v = b64decode(v)
                v = np.frombuffer(v, dtype=np.uint8)
                v = cv2.imdecode(v, cv2.IMREAD_COLOR)
                if v is None:
                    raise Exception('Decode Error')
                data[k] = v
            except Exception as e:
                logger.info(e, exc_info=True)
            logger.info(f'passing image succ!')
        elif type(v) in [dict, list, tuple]:
            process_data(v)


config = get_configure()
config.from_yaml('config_files/quater.yaml')
pp = Predictor(config)
pp.init_env()

app = Flask(__name__)
logger = get_logger('server logger')
redis_host = 'localhost'
app.config.from_object(FlaskConfig())
logger.info('fuck')


@app.route("/")
def fuck():
    return 'fuck  you !'


@app.route("/api/grade", methods=['POST'])
def grade_one():
    data = request.get_data(as_text=True)
    data = json.loads(data)
    logger.info(data)
    process_data(data)
    resp = pp(data)
    return json.dumps(resp)


@app.route("/api/<string:taskname>", methods=['POST'])
def serve(taskname):
    logger.info(f'taskname {taskname}')
    logger.info(request.content_length)
    task = Task(
        taskname,
        redis_host=app.config['REDIS_SERVER_ADDRESS'],
        redis_port=app.config['REDIS_SERVER_PORT'])
    data = request.get_data(as_text=True)
    data = json.loads(data)
    logger.info(data)
    process_data(data)
    resp = task.issue(data)
    rr = resp.get()
    return json.dumps(rr)


if __name__ == "__main__":
    resp = pp('../data/annotation4/images/20170313_709_张文彪_0848504596_R')
