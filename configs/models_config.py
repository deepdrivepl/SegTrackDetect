import torch
from torchvision import transforms as T


def roi_transform(h,w):
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((h, w), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


def unet_postprocess(output, ori_shape, sigmoid_included=False, thresh=None):
    output = torch.squeeze(output)
    if not sigmoid_included:
        output = output.sigmoid()
    if thresh:
        output = torch.where(output > thresh, 1.0, 0.0)

    output = 255 * output.detach().cpu().numpy().astype(np.uint8) 
    output_fullres = cv2.resize(output, (ori_shape[1], ori_shape[2]))
    return output_fullres, output



def u2net_postprocess(output, ori_shape, sigmoid_included=True, thresh=0.5):
    output = output[0] 
    output = torch.squeeze(output)
    if not sigmoid_included:
        output = output.sigmoid()
    if thresh:
        output = torch.where(output > thresh, 1.0, 0.0)
    output = 255 * output.detach().cpu().numpy().astype(np.uint8)
    output_fullres = cv2.resize(output, (ori_shape[1], ori_shape[2]))
    return output_fullres, output



def yolov7_postprocess(output):
    return output[0]
    

    
u2net = dict(
    weights = "trained_models/u2netp.pt", #trained_models/DistanceAwareCELoss-aug-scheduler-u2netp-00000056.pth",
    in_size = (576,576),
    thresh = 0.5,
    transform = roi_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
)


unet = dict(
    weights = "trained_models/trained_models/unet-r18_005_best_model_loss.pt",
    in_size = (160,256),
    thresh = None,
    args = None,
    transform = roi_transform,
    sigmoid_included = False,
    postprocess = u2net_postprocess,

)


yolov7_w6 = dict(
    weights = "trained_models/yolov7-w6-640-best.torchscript.pt",
    in_size = (448,704),
    conf_thresh = 0.001,
    iou_thresh = 0.65,
    transform = T.ToTensor(),
    postprocess=yolov7_postprocess,
)


yolov7_tiny = dict(
    weights = "trained_models/yolov7-tiny-300-best.torchscript.pt",
    in_size = (160,256),
    conf_thresh = 0.001,
    iou_thresh = 0.65,
    transform = T.ToTensor(),
    postprocess=yolov7_postprocess,
)


yolov4 = dict(
    weights = "trained_models/mtsd001_best.pt",
    in_size = (960,960),
    conf_thresh = 0.001,
    iou_thresh = 0.65,
    transform = T.ToTensor(),
    postprocess=yolov7_postprocess,
)


ROI_MODELS = {
    "u2net": u2net,
    "unet": unet,
}


DET_MODELS = {
    "yolov7_w6": yolov7_w6,
    "yolov7_tiny": yolov7_tiny,
    "yolov4": yolov4
}