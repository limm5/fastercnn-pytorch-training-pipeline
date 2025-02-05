from models import *

def return_fasterrcnn_resnet50_fpn(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_resnet50_fpn.create_model(
        num_classes, pretrained=pretrained, coco_model=coco_model
    )
    return model

def return_fasterrcnn_mobilenetv3_large_fpn(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_mobilenetv3_large_fpn.create_model(
        num_classes, pretrained=pretrained, coco_model=coco_model
    )
    return model

def return_fasterrcnn_mobilenetv3_large_320_fpn(
    num_classes, pretrained=True, coco_model=False
):    
    model = fasterrcnn_mobilenetv3_large_320_fpn.create_model(
        num_classes
    )
    return model

def return_fasterrcnn_resnet50(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_resnet50.create_model(
        num_classes, pretrained, coco_model
    )
    return model

def return_fasterrcnn_resnet18(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_resnet18.create_model(
        num_classes, pretrained, coco_model
    )
    return model

def return_fasterrcnn_custom_resnet(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_custom_resnet.create_model(
        num_classes, pretrained, coco_model
    )
    return model

def return_fasterrcnn_darknet(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_darknet.create_model(
        num_classes, pretrained, coco_model
    )
    return model

create_model = {
    'fasterrcnn_resnet50_fpn': return_fasterrcnn_resnet50_fpn,
    'fasterrcnn_mobilenetv3_large_fpn': return_fasterrcnn_mobilenetv3_large_fpn,
    'fasterrcnn_mobilenetv3_large_320_fpn': return_fasterrcnn_mobilenetv3_large_320_fpn,
    'fasterrcnn_resnet50': return_fasterrcnn_resnet50,
    'fasterrcnn_resnet18': return_fasterrcnn_resnet18,
    'fasterrcnn_custom_resnet': return_fasterrcnn_custom_resnet,
    'fasterrcnn_darknet': return_fasterrcnn_darknet
}