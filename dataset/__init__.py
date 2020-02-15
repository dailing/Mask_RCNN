from .old_anno import OldAnno
from .kaggle_dr import DRDataset
from .cubbird import CUBBirdDataset
from .mnist import Mnist
from .sqlDB import SqlDB
from .vessel_seg import DRIVE
from .mnist_noise import MnistNoise
from .csv_db import CsvDataset

datasets = dict(
    old_annotation=OldAnno,
    kaggle_dr=DRDataset,
    CUBBird=CUBBirdDataset,
    Mnist=Mnist,
    dql_db=SqlDB,
    DRIVE=DRIVE,
    MnistNoise=MnistNoise,
    CsvDataset=CsvDataset,
)
