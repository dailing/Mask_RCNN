import click

from util.augment import Compose, FundusAOICrop, Resize, ImageReader, ImageSaver, ToFloat
from os import walk, cpu_count, makedirs
from os.path import join as pjoin
from os.path import abspath, split
from multiprocessing import Pool
import numpy as np
import cv2

def worker(args):
    in_img = args[0]
    transform = Compose([
        ImageReader(),
        ToFloat(),
    ])

    output = transform(in_img)
    if output is None:
        return np.array((0,0,0)), np.array((1,1,1))
    return output.mean(axis=(0,1)), output.std(axis=(0,1))

@click.command()
@click.option('--data_dir', help='the dataset dir')
def crop_image(data_dir):
    data_dir = abspath(data_dir)
    images = []
    for dirpath, _, filenames in walk(data_dir):
        for f in filenames:
            if '.' not in f or f.split('.')[-1].lower() in ['jpg', 'jpeg', 'png', 'tiff']:
                pp = pjoin(dirpath, f)
                # assert pp.startswith(data_dir)
                # pp = pp[len(data_dir) + 1:]
                images.append((pp,) )
    pool = Pool(processes=cpu_count())

    out = pool.map_async(worker, images, chunksize=5 * cpu_count())
    out.wait()
    
    mean = np.array([i[0] for i in out.get()])
    std =  np.array([i[1] for i in out.get()])
    # print(std[0])
    print('Mean BGR:', mean.mean(axis=0))
    print('Std BGR:', std.mean(axis=0))

if __name__ == "__main__":
    crop_image()