import torch
import torchvision
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_frcnn_mobilenet(num_classes: int):
    """ Build a Faster R-CNN model with a MobileNetV3-Large FPN backbone."""
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_model(num_classes: int, architecture: str):
    print(f"Building model: {architecture}")
    return get_frcnn_mobilenet(num_classes)


def build_model(num_classes: int, architecture: str):
    return get_model(num_classes, architecture)