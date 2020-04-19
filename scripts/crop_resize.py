import click

from util.augment import Compose, FundusAOICrop, Resize, ImageReader, ImageSaver
from os import walk, cpu_count, makedirs
from os.path import join as pjoin
from os.path import abspath, split
from multiprocessing import Pool
import cv2

def walker(args):
    in_img, out_img, size = args
    o_dir, _ = split(out_img)
    try:
        makedirs(o_dir, exist_ok=True)
    except Exception as e:
        pass
    transform = Compose([
        ImageReader(),
        FundusAOICrop(),
        Resize(size),
        ImageSaver(),
    ])

    output = transform(in_img)
    if output is None:
        return False
    with open(out_img, 'wb') as f:
        f.write(output)
    return True

@click.command()
@click.option('--size', help='the output size of the image', type=int)
@click.option('--output_dir', help='the output dir of the dataset')
@click.option('--data_dir', help='the dataset dir')
def crop_image(size, output_dir, data_dir):
    output_dir = abspath(output_dir)
    data_dir = abspath(data_dir)
    assert output_dir != data_dir
    images = []
    for dirpath, dirnames, filenames in walk(data_dir):
        for f in filenames:
            if '.' not in f or f.split('.')[-1].lower() in ['jpg', 'jpeg', 'png', 'tiff']:
                pp = pjoin(dirpath, f)
                assert pp.startswith(data_dir)
                pp = pp[len(data_dir) + 1:]
                images.append((pjoin(data_dir, pp), pjoin(output_dir, pp), size) )
    pool = Pool(processes=cpu_count())

    out = pool.map_async(walker, images, chunksize=5 * cpu_count())
    out.wait()
    for res, (in_img, _, _) in zip(out.get(), images):
        if not res:
            print(f"FUCK {in_img}!!!")


if __name__ == "__main__":
    crop_image()