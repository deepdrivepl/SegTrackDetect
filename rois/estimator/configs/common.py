from torchvision import transforms as T



def estimator_preprocess(h, w):
    """
    Creates a transformation pipeline for image preprocessing for the 
    UNet and U2Net Estimation models.

    Args:
        h (int): The desired height of the output image after resizing.
        w (int): The desired width of the output image after resizing.

    Returns:
        torchvision.transforms.Compose: A composed transform that includes resizing 
        and normalization.
    """
    transform = T.Compose([
        T.Resize((h, w), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform