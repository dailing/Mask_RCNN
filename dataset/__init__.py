from .old_anno import OldAnno
from .kaggle_dr import DRDataset
from .cubbird import CUBBirdDataset

datasets = dict(
    old_annotation=OldAnno,
    kaggle_dr=DRDataset,
    CUBBird=CUBBirdDataset,
)
