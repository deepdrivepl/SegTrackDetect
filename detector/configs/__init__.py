from .yolo import *


"""
DETECTION_MODELS

A dictionary that maps dataset names to their respective detection models configuration dictionaties.

Keys:
    str: Dataset names (e.g., "MTSD", "ZeF20", "SDS", "DC").
    
Values:
    type: The corresponding detection model configuration, which includes:
        - MTSD: Mapillary Traffic Sign Dataset detection model for multi-class traffic sign detection.
        - ZeF20: ZebraFish detection model configuration for small fish detection.
        - SDS: SeaDronesSee detection model configuration for drone-based object detection.
        - DC: DroneCrowd detection model configuration designed for drone-based people detection.

Usage:
    Access the model configuration by referring to its corresponding dataset name. For example:
        model_config = DETECTION_MODELS["MTSD"]
    
    You can add custom models by defining them in `yolo.py`, then use them for inference in `inference_vid.py`. For example:
        CustomModel = dict(
            weights = "weights/my_custom_model_weights.pt",
            in_size = (512, 512),
            conf_thresh = 0.1,
            iou_thresh = 0.45,
            multi_label = True,
            labels = [],
            merge = False,
            agnostic = False,
            transform = T.ToTensor(),
            postprocess = yolo_postprocess,
            classes = ['class_a', 'class_b', 'class_c'],
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        )
        
        DETECTION_MODELS = {
            "MTSD": MTSD,
            "ZeF20": ZeF20,
            "SDS": SeaDronesSee,
            "DC": DroneCrowd,
            "CUSTOM": CustomModel,
        }
"""
DETECTION_MODELS = {
    "MTSD": MTSD,
    "ZeF20": ZeF20,
    "SDS": SeaDronesSee,
    "DC": DroneCrowd,
}
