from torchvision import transforms as T



def estimator_transform(h,w):
    transform = T.Compose([
        T.Resize((h, w), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform