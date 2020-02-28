
from util.registry import Registry

MODEL_REGISTRY = Registry('MODEL')

models = MODEL_REGISTRY
# (
#     # inceptionv4=InceptionV4,
#     # resnet50=ResNet50,
#     # resnet18=ResNet18,
#     # resnet101=ResNet101,
#     # pnasnet5=PNASNet5Large,
#     # alexnet=AlexNet,
#     # lenet=LeNet,
#     # darknet=Darknet53,
#     # deeplab=DeepLab,
#     # expModel=ExperimentModel,
#     # MaskRCNN=MaskRCNN,
# )


from . import inceptionv4 
from . import resnet
from . import pasnet 
from . import alexnet 
from . import lenet 
from . import dark_53 
# from . import identityLayer 
from . import deeplab_v3 
from . import experiment 
from . import maskrcnn 
