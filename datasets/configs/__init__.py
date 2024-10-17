from .datasets import *


"""
DATASETS

A dictionary that maps dataset names to their respective dataset configuration dictionaries.

Keys:
    str: Dataset names (e.g., "DroneCrowd", "SeaDronesSee", "ZebraFish", "MTSD").
    
Values:
    type: The corresponding dataset class, which includes:
        - DroneCrowd: Dataset for drone-based people detection in crowded environments.
        - SeaDronesSee: Dataset for object detection in maritime drone imagery.
        - ZebraFish: Dataset for detecting small fish.
        - MTSD: Mapillary Traffic Sign Dataset for multi-class traffic sign detection.

Usage:
    Access the dataset class by referring to its corresponding name. For example:
        dataset_class = DATASETS["MTSD"]
    
    You can add custom datasets by defining them in your code and extending the `DATASETS` dictionary. For example:
        CustomDataset = CustomDatasetClass(data_root="/path/to/data", colors=[(255, 0, 0),(0, 255, 0),(0, 0, 255)]])
        
        DATASETS = {
            "DroneCrowd": DroneCrowd,
            "SeaDronesSee": SeaDronesSee,
            "ZebraFish": ZebraFish,
            "MTSD": MTSD,
            "CUSTOM": CustomDataset,
        }
"""
DATASETS = {
    "DroneCrowd": DroneCrowd,
    "SeaDronesSee": SeaDronesSee,
    "ZebraFish": ZebraFish,
    "MTSD": MTSD
}