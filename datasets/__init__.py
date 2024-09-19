from .DroneCrowd import DroneCrowdDataset
from .SeaDronesSee import SeaDronesSeeDataset
from .ZebraFish import ZebraFishDataset
from .MTSD import MTSDDataset
from .dataset import ROIDataset, WindowDetectionDataset


DATASETS = {
    "DroneCrowd": DroneCrowdDataset,
    "SeaDronesSee": SeaDronesSeeDataset,
    "ZebraFish": ZebraFishDataset,
    "MTSD": MTSDDataset
}