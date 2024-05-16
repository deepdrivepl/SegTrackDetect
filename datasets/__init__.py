from .DroneCrowd import DroneCrowdDataset
from .SeaDronesSee import SeaDronesSeeDataset
from .ZebraFish import ZebraFishDataset
from .MTSD import MTSDDataset

import inspect


DATASETS = {
    "DroneCrowd": DroneCrowdDataset,
    "SeaDronesSee": SeaDronesSeeDataset,
    "ZebraFish": ZebraFishDataset,
    "MTSD": MTSDDataset
}