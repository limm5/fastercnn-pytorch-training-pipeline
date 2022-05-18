import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

def create_model(num_classes, pretrained=True, coco_model=False, freeze_backbone=False):
    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v3_small(pretrained=pretrained).features
    backbone.out_channels = 576
    if freeze_backbone:
        count = 0
        for param in backbone.parameters():            
                if count <3:
                    param.requires_grad = False
                else:
                    break

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                    num_classes=num_classes,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)
    return model
