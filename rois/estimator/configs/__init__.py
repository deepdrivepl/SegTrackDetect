from .unet import *
from .u2net import *



ESTIMATOR_MODELS = {
    "MTSD": MTSD,
    "ZeF20": ZeF20,
    "SDS_tiny": SeaDronesSee_tiny,
    "SDS_small": SeaDronesSee_small,
    "SDS_medium": SeaDronesSee_medium,
    "SDS_large": SeaDronesSee_large,
    "DC_tiny": DroneCrowd_tiny,
    "DC_small": DroneCrowd_small,
    "DC_medium": DroneCrowd_medium,
}