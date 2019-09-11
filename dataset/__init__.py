from .old_anno import OldAnno
from .kaggle_dr import DRDataset

datasets = dict(
    old_annotation=OldAnno,
    kaggle_dr=DRDataset,
)
