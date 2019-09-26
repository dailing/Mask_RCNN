from .inceptionv4 import InceptionV4
from .resnet import ResNet50, ResNet101
from .pasnet import PNASNet5Large

models = dict(
    inceptionv4=InceptionV4,
    resnet50=ResNet50,
    resnet101=ResNet101,
    pnasnet5=PNASNet5Large,
)
