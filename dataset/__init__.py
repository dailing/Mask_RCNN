from .old_anno import OldAnno
from .kaggle_dr import DRDataset
from .cubbird import CUBBirdDataset
from .mnist import Mnist
from .sqlDB import SqlDB

datasets = dict(
    old_annotation=OldAnno,
    kaggle_dr=DRDataset,
    CUBBird=CUBBirdDataset,
    Mnist=Mnist,
    dql_db=SqlDB,
)
