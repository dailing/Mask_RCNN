import os.path
import cv2
from tqdm import tqdm
from random import shuffle

from util.augment import Compose, FundusAOICrop, Resize

input_dir = '/share/kaggke_dr'
out_dir = '/share/small_pic/kaggle_dr@448'

os.makedirs(out_dir, exist_ok=True)

files = os.listdir(input_dir)
shuffle(files)


for i in tqdm(files):
    ofile = f'{out_dir}/{i}'
    if os.path.exists(ofile):
        continue
    transform = Compose(
        (FundusAOICrop(), Resize(448))
    )
    img = cv2.imread(f'{input_dir}/{i}', cv2.IMREAD_COLOR)
    img = transform(img)
    ok,content = cv2.imencode('.png', img)
    assert  ok is True
    with open(ofile, 'wb') as f:
        f.write(content)
    # cv2.imwrite(ofile, img, dict(ext='png'))

