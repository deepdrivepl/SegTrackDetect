from torchvision import transforms as T


def roi_transform(h,w):
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((h, w), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

u2net = dict(
    weights = "trained_models/DistanceAwareCELoss-aug-scheduler-u2netp-00000056.pth",
    in_size = (576,576),
    thresh = 0.5,
    jit = False,
    module = "models",
    class_name = "U2NETP",
    args = dict(
        in_ch = 3,
        out_ch = 1,
    ),
    transform = roi_transform,
    sigmoid_included = True,
)


unet = dict(
    weights = "trained_models/unet-r18_005_best_model_loss.pt",
    in_size = (160,256),
    thresh = 0.5,
    jit = True,
    module = None,
    class_name = None,
    args = None,
    transform = roi_transform,
    sigmoid_included = False
)


yolov7_w6 = dict(
    weights = "trained_models/yolov7-w6-640-best.torchscript.pt",
    jit = True,
    module = None,
    class_name = None,
    args = None,
    in_size = (448,704),
    conf_thresh = 0.001, # can vary!!!
    iou_thresh = 0.65,
    transform = T.ToTensor(),
)


yolov7_tiny = dict(
    weights = "trained_models/yolov7-tiny-300-best.torchscript.pt",
    jit = True,
    module = None,
    class_name = None,
    args = None,
    in_size = (160,256),
    conf_thresh = 0.001, # can vary!!!
    iou_thresh = 0.65, # can vary!!!
    transform = T.ToTensor(),
)


ROI_MODELS = {
    "u2net": u2net,
    "unet": unet,
}


DET_MODELS = {
    "yolov7_w6": yolov7_w6,
    "yolov7_tiny": yolov7_tiny
}