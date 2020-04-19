from util.logs import get_logger
from torch.utils.data import DataLoader
from dataset.lesion_seg_mask_rcnn import LesionSegMask
from os import cpu_count
from torch.utils.data import Dataset
from util.augment import Compose, ToFloat, ToTensor
from model.maskrcnn import MaskRCNN
from torch.optim import Adam, SGD
from sqlitedict import SqliteDict
from io import BytesIO
from tqdm import tqdm
import torch
import os
from tensorboardX import SummaryWriter

logger = get_logger('main')
num_processor = 2
device = torch.device('cuda')
summery_writer = SummaryWriter(logdir=f'log/maskrcnn/log')

def wtire_summary(loss_map, tag='train', step=None):
    for k, v in loss_map.items():
        summery_writer.add_scalar(
            f'{tag}/{k}_loss',
            v.detach().cpu().numpy(),
            global_step=step
        )



class TrainEvalDataset(Dataset):
    def __init__(self, data_reader, split='train'):
        super().__init__()
        self.data_reader = data_reader
        if split == 'train':
            transform = [
                ToFloat(),
                ToTensor(),
            ]
        else:
            transform = [
                ToFloat(),
                ToTensor(),
            ]
        self.transform = Compose(transform)

    def __getitem__(self, index):
        try:
            image, label = self.data_reader[index]
            image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(e, exc_info=True)

    def __len__(self):
        return self.data_reader.__len__()


def train():
    loader = DataLoader(
        TrainEvalDataset(
            LesionSegMask(split='train')),
        batch_size= 1, shuffle=False, num_workers=num_processor)
    test_loader = DataLoader(
        TrainEvalDataset(
            LesionSegMask(split='test')),
        batch_size= 1, shuffle=False, num_workers=num_processor)

    net = MaskRCNN(num_class=5)
    net = net.to(device)
    optimizer = SGD(net.parameters(), 0.001, 0.9, weight_decay=0.00001)
    # exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, 0.95)

    storage_dict = SqliteDict(f'log/maskrcnn/dcl_snap.db')
    start_epoach = 0
    # if len(storage_dict) > 0:
    #     kk = list(storage_dict.keys())
    #     # net.load_state_dict(
    #     #     torch.load(BytesIO(storage_dict[38])))
    #     net.load_state_dict(
    #         torch.load(BytesIO(storage_dict[kk[-1]])))
    #     start_epoach = int(kk[-1]) + 1
    #     logger.info(f'loading from epoach{start_epoach}')
    global_step = 0
    for epoach in (range(start_epoach, 120)):
        net.train()
        for batch_cnt, batch in tqdm(enumerate(loader), total=len(loader)):
            image, label = batch
            for k, v in label.items():
                label[k] = v.squeeze()
            image = image.to(device)
            for k, v in label.items():
                if isinstance(v, torch.Tensor):
                    label[k] = label[k].to(device)
            # print(label['boxes'].shape)
            optimizer.zero_grad()
            net_out = net(image, [label])
            loss = 0
            for i in net_out.values():
                loss += i  
            net_out['loss_sum'] = loss
            loss.backward()
            wtire_summary(net_out, 'train', global_step)
            optimizer.step()
            global_step += 1
        # exp_lr_scheduler.step(epoach)
        logger.debug(f'saving epoach {epoach}')
        buffer = BytesIO()
        torch.save(net.state_dict(), buffer)
        buffer.seek(0)
        storage_dict[epoach] = buffer.read()
        storage_dict.commit()
        # test(config, net, test_loader, epoach, loss_calculator)

if __name__ == "__main__":
    train()