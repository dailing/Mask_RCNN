from torchvision.models.detection import MaskRCNN as MaskRCNN_
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from . import MODEL_REGISTRY


model_urls = {
    'maskrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
}


@MODEL_REGISTRY.register()
class MaskRCNN(MaskRCNN_):
    def __init__(self, num_class=10):
        backbone = resnet_fpn_backbone('resnet50', False)
        anchor_sizes = ((8, 16, 32, 64, 128), )
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios
        )
        super(MaskRCNN, self).__init__(
            backbone,
            num_class,
            rpn_anchor_generator=None)
        state_dict = load_state_dict_from_url(
            model_urls['maskrcnn_resnet50_fpn_coco'],
            model_dir='asserts/',
            progress=True)
        del state_dict['roi_heads.box_predictor.cls_score.weight']
        del state_dict['roi_heads.box_predictor.cls_score.bias']
        del state_dict['roi_heads.box_predictor.bbox_pred.weight']
        del state_dict['roi_heads.box_predictor.bbox_pred.bias']
        del state_dict['roi_heads.mask_predictor.mask_fcn_logits.weight']
        del state_dict['roi_heads.mask_predictor.mask_fcn_logits.bias']
        unused = self.load_state_dict(state_dict, strict=False)
        # print("### unused  parameters ", unused)

