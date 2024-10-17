from .unet import *
from .u2net import *


"""
ESTIMATOR_MODELS

A dictionary that maps model names to their respective ROI estimator configurations.

Keys:
    str: Model names (e.g., "MTSD", "ZeF20", "SDS_tiny", "DC_small").

Values:
    type: The corresponding ROI estimator configuration, which includes:
        - MTSD: ROI estimator configuration for Mapillary Traffic Sign Dataset.
        - ZeF20: ROI estimator configuration for ZebraFish detection model.
        - SDS_tiny: SeaDronesSee ROI model configuration with tiniest input size during training (check `unet.py` for details).
        - SDS_small: SeaDronesSee ROI model configuration with small input size during training (check `unet.py` for details).
        - SDS_medium: SeaDronesSee ROI model configuration with medium input size during training (check `unet.py` for details).
        - SDS_large: SeaDronesSee ROI model configuration with large input size during training (check `unet.py` for details).
        - DC_tiny: DroneCrowd ROI model configuration with tiny input size during training (check `unet.py` for details).
        - DC_small: DroneCrowd ROI model configuration with small input size during training (check `unet.py` for details).
        - DC_medium: DroneCrowd ROI model configuration with medium input size during training (check `unet.py` for details).

Usage:
    Access the model configuration by referring to its corresponding model name. For example:
        model_config = ESTIMATOR_MODELS["MTSD"]
    
    You can add custom models by defining them in the respective module files, then use them for inference in your applications. For example:
        CustomModel = dict(
            weights = "weights/my_custom_model_weights.pt",
            in_size = (512, 512),
            postprocess = my_postprocess,
            postprocess_args = dict(
                ...
            ),
            preprocess = my_preprocess,
            preprocess_args = dict(
                ...
            )
        )
        
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
            "CUSTOM": CustomModel,
        }
"""
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
