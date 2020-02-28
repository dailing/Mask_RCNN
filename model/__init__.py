from .inceptionv4 import InceptionV4
from .resnet import ResNet50, ResNet101, ResNet18
from .pasnet import PNASNet5Large
from .alexnet import AlexNet
from .lenet import LeNet
from .dark_53 import Darknet53
from .identityLayer import IdentityNet
from .deeplab_v3 import DeepLab
from .experiment import ExperimentModel
from .maskrcnn import MaskRCNN

models = dict(
    inceptionv4=InceptionV4,
    resnet50=ResNet50,
    resnet18=ResNet18,
    resnet101=ResNet101,
    pnasnet5=PNASNet5Large,
    alexnet=AlexNet,
    lenet=LeNet,
    darknet=Darknet53,
    deeplab=DeepLab,
    expModel=ExperimentModel,
    MaskRCNN=MaskRCNN,
)
