import cv2
import numpy as np

import torch
from torchvision import transforms as T


def roi_transform(h,w):
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((h, w), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


def det_transform(h,w):
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((h, w), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform



def unet_postprocess(output, ori_shape, sigmoid_included=False, thresh=None, dilate=False, k_size=3, iter=1):
    output = torch.squeeze(output)
    if not sigmoid_included:
        output = output.sigmoid()
    if thresh:
        output = torch.where(output > thresh, 1.0, 0.0)
    output = 255 * output.detach().cpu().numpy().astype(np.uint8) 
    if dilate:
        kernel = np.ones((k_size, k_size), np.uint8)
        output = cv2.dilate(output, kernel, iterations = iter)

    output_fullres = cv2.resize(output, (ori_shape[1], ori_shape[0]))
    
    return output_fullres, output



def u2net_postprocess(output, ori_shape, sigmoid_included=True, thresh=0.5, dilate=False, k_size=3, iter=1):
    output = output[0] 
    output = torch.squeeze(output)
    if not sigmoid_included:
        output = output.sigmoid()
    if thresh:
        output = torch.where(output > thresh, 1.0, 0.0)
    output = 255 * output.detach().cpu().numpy().astype(np.uint8)
    if dilate:
        kernel = np.ones((k_size, k_size), np.uint8)
        output = cv2.dilate(output, kernel, iterations = iter)
    output_fullres = cv2.resize(output, (ori_shape[1], ori_shape[0]))
    return output_fullres, output



def yolov7_postprocess(output):
    return output[0]
    

    
u2net = dict(
    weights = "weights/u2netp_MTSD.pt",
    in_size = (576,576),
    thresh = 0.5,
    transform = roi_transform,
    sigmoid_included = True,
    postprocess = u2net_postprocess,
)


unet = dict(
    weights = "weights/unetR18-ZebraFish.pt",
    in_size = (160,256),
    thresh = None,
    args = None,
    transform = roi_transform,
    sigmoid_included = False,
    postprocess = unet_postprocess,
    # dilate = True, # TODO - best metrics (padding 10, dilate 7, letterbox)
    # k_size = 7,
    # iter = 1

)


unet_DC0 = dict(
    weights = "weights/DroneCrowd-000-R18-192x320-best-loss.pt",
    in_size = (192,320),
    thresh = 0.5,
    args = None,
    transform = roi_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
)

unet_DC1 = dict(
    weights = "weights/DroneCrowd-001-R18-384x640-best-loss.pt",
    in_size = (384,640),
    thresh = 0.5,
    args = None,
    transform = roi_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
)

unet_DC1_tiny = dict(
    weights = "weights/DroneCrowd-001-R18-96x160-best-loss.pt",
    in_size = (96,160),
    thresh = 0.5,
    args = None,
    transform = roi_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
)

unet_DC1_small = dict(
    weights = "weights/DroneCrowd-001-R18-192x320-best-loss.pt",
    in_size = (192,320),
    thresh = 0.5,
    args = None,
    transform = roi_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
)

unet_DC1_medium = dict(
    weights = "weights/DroneCrowd-001-R18-384x640-best-loss.pt",
    in_size = (384,640),
    thresh = 0.5,
    args = None,
    transform = roi_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
)


unet_DC2 = dict(
    weights = "weights/DroneCrowd-002-R34-384x640-best-loss.pt",
    in_size = (384,640),
    thresh = 0.5,
    args = None,
    transform = roi_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
)


unet_SDS0 = dict(
    weights = "weights/SeaDronesSee-000-R18-224x384-best-loss.pt",
    in_size = (224,384),
    thresh = 0.5,
    args = None,
    transform = roi_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
)


unet_SDS_tiny = dict(
    weights = "ablation/SEG-RES/SEG_RES/SeaDronesSee-000-R18-64x96-best-loss.pt",
    in_size = (64,96),
    thresh = 0.5,
    args = None,
    transform = roi_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
)

unet_SDS_small = dict(
    weights = "ablation/SEG-RES/SEG_RES/SeaDronesSee-000-R18-128x192-best-loss.pt",
    in_size = (128,192),
    thresh = 0.5,
    args = None,
    transform = roi_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
)

unet_SDS_medium = dict(
    weights = "ablation/SEG-RES/SEG_RES/SeaDronesSee-000-R18-224x384-best-loss.pt",
    in_size = (224,384),
    thresh = 0.5,
    args = None,
    transform = roi_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
)

unet_SDS_large = dict(
    weights = "ablation/SEG-RES/SEG_RES/SeaDronesSee-000-R18-448x768-best-loss.pt",
    in_size = (448,768),
    thresh = 0.5,
    args = None,
    transform = roi_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
)


unet_SDS1 = dict(
    weights = "weights/SeaDronesSee-001-R18-448x768-best-loss.pt",
    in_size = (448,768),
    thresh = 0.5,
    args = None,
    transform = roi_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
)

unet_SDS1_trt = dict(
    weights = "onnx-bs1/SDS-unetR18-001-fp32.engine",
    in_size = (448,768),
    thresh = 0.5,
    args = None,
    transform = roi_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
)

unet_SDS1_trt_fp16 = dict(
    weights = "onnx-bs1/SDS-unetR18-001-fp16.engine",
    in_size = (448,768),
    thresh = 0.5,
    args = None,
    transform = roi_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
)

unet_SDS1_trt_int8 = dict(
    weights = "onnx-bs1/SDS-unetR18-001-int8.engine",
    in_size = (448,768),
    thresh = 0.5,
    args = None,
    transform = roi_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
)

unet_SDS2 = dict(
    weights = "weights/SeaDronesSee-002-R34-448x768-best-loss.pt",
    in_size = (448,768),
    thresh = 0.5,
    args = None,
    transform = roi_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
)


unet_SDS0_masks = dict(
    weights = "weights/SeaDronesSee-003-R18-224x384-sam-best-loss.pt",
    in_size = (224,384),
    thresh = 0.5,
    args = None,
    transform = roi_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
)

unet_SDS1_masks = dict(
    weights = "weights/SeaDronesSee-004-R18-448x768-sam-best-loss.pt",
    in_size = (448,768),
    thresh = 0.5,
    args = None,
    transform = roi_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
)

unet_SDS2_masks = dict(
    weights = "weights/SeaDronesSee-005-R34-448x768-sam-best-loss.pt",
    in_size = (448,768),
    thresh = 0.5,
    args = None,
    transform = roi_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
)


yolov7_tiny = dict(
    weights = "weights/yolov7t-ZebraFish.pt",
    in_size = (160,256),
    conf_thresh = 0.01,
    iou_thresh = 0.65,
    transform = det_transform, # T.ToTensor(),
    postprocess=yolov7_postprocess,
)

yolov7_tiny_SDS = dict(
    weights = "weights/000-SeaDronesSee-yolov7-tiny-320x512-best.torchscript.pt",
    in_size = (320,512),
    conf_thresh = 0.01,
    iou_thresh = 0.65,
    transform = det_transform, # T.ToTensor(),
    postprocess=yolov7_postprocess,
)

# yolov7_tiny_SDS_crops = dict(
#     weights = "weights/001-SeaDronesSee-yolov7-tiny-320x512-crops-only-best.torchscript.pt",
#     in_size = (320,512),
#     conf_thresh = 0.01,
#     iou_thresh = 0.65,
#     transform = det_transform, # T.ToTensor(),
#     postprocess=yolov7_postprocess,
# )

yolov7_tiny_SDS_crops = dict(
    weights = "weights/001-SeaDronesSee-yolov7-tiny-512x512-crops-only-best.torchscript.pt",
    in_size = (512,512),
    conf_thresh = 0.01,
    iou_thresh = 0.65,
    transform = det_transform, # T.ToTensor(),
    postprocess=yolov7_postprocess,
)

yolov7_tiny_SDS_crops_mul_scales = dict(
    weights = "weights/003-SeaDronesSee-yolov7-tiny-512x512-crops-only-multiple-scales-last.torchscript.pt",
    in_size = (512,512),
    conf_thresh = 0.01,
    iou_thresh = 0.65,
    transform = det_transform, # T.ToTensor(),
    postprocess=yolov7_postprocess,
)

yolov7_tiny_SDS_crops_mul_scales_100 = dict(
    weights = "weights/004-SeaDronesSee-yolov7-tiny-512x512-crops-only-multiple-scales-100ep-last.torchscript.pt",
    in_size = (512,512),
    conf_thresh = 0.01,
    iou_thresh = 0.65,
    transform = det_transform, # T.ToTensor(),
    postprocess=yolov7_postprocess,
)


yolov7_tiny_SDS_crops_mul_scales_100_best = dict(
    weights = "weights/004-SeaDronesSee-yolov7-tiny-512x512-crops-only-multiple-scales-100ep-best.torchscript.pt",
    in_size = (512,512),
    conf_thresh = 0.01,
    iou_thresh = 0.65,
    transform = det_transform, # T.ToTensor(),
    postprocess=yolov7_postprocess,
)


yolov7_tiny_SDS_crops_mul_scales_300 = dict(
    weights = "weights/006-SeaDronesSee-yolov7-tiny-512x512-crops-only-multiple-scales-300ep-best.torchscript.pt",
    in_size = (512,512),
    conf_thresh = 0.01,
    iou_thresh = 0.65,
    transform = det_transform, # T.ToTensor(),
    postprocess=yolov7_postprocess,
)

yolov7_tiny_SDS_crops_mul_scales_300_trt = dict(
    weights = "onnx-bs1/SDS-v7t-006-fp32.engine",
    in_size = (512,512),
    conf_thresh = 0.01,
    iou_thresh = 0.65,
    transform = det_transform, # T.ToTensor(),
    postprocess=yolov7_postprocess,
)

yolov7_tiny_SDS_crops_mul_scales_300_trt_fp16 = dict(
    weights = "onnx-bs1/SDS-v7t-006-fp16.engine",
    in_size = (512,512),
    conf_thresh = 0.01,
    iou_thresh = 0.65,
    transform = det_transform, # T.ToTensor(),
    postprocess=yolov7_postprocess,
)

yolov7_tiny_SDS_crops_mul_scales_300_trt_int8 = dict(
    weights = "onnx-bs1/SDS-v7t-006-int8.engine",
    in_size = (512,512),
    conf_thresh = 0.01,
    iou_thresh = 0.65,
    transform = det_transform, # T.ToTensor(),
    postprocess=yolov7_postprocess,
)

yolov7_tiny_SDS_crops_mul_scales_100_bigger = dict(
    weights = "weights/007-SeaDronesSee-yolov7-tiny-512x512-crops-only-multiple-scales-bigger-100ep-best.torchscript.pt",
    in_size = (512,512),
    conf_thresh = 0.01,
    iou_thresh = 0.65,
    transform = det_transform, # T.ToTensor(),
    postprocess=yolov7_postprocess,
)


yolov7_SDS_crops_mul_scales = dict(
    weights = "weights/005-SeaDronesSee-yolov7-512x512-crops-only-multiple-scales-best.torchscript.pt",
    in_size = (512,512),
    conf_thresh = 0.01,
    iou_thresh = 0.65,
    transform = det_transform, # T.ToTensor(),
    postprocess=yolov7_postprocess,
)

yolov7_tiny_DC = dict(
    weights = "weights/000-DroneCrowd-yolov7-tiny-320x512-best.torchscript.pt",
    in_size = (320,512),
    conf_thresh = 0.01,
    iou_thresh = 0.65,
    transform = det_transform, # T.ToTensor(),
    postprocess=yolov7_postprocess,
)

# yolov7_tiny_DC_crops = dict(
#     weights = "weights/001-DroneCrowd-yolov7-tiny-512x512-crops-only-best.torchscript.pt",
#     in_size = (512,512),
#     conf_thresh = 0.01,
#     iou_thresh = 0.65,
#     transform = det_transform, # T.ToTensor(),
#     postprocess=yolov7_postprocess,
# )

yolov7_tiny_DC_crops = dict(
    weights = "weights/DC/001-DroneCrowd-yolov7-tiny-512x512-crops-only-best.torchscript.pt",
    in_size = (512,512),
    conf_thresh = 0.01,
    iou_thresh = 0.65,
    transform = det_transform, # T.ToTensor(),
    postprocess=yolov7_postprocess,
)

yolov7_tiny_DC_crops_rect = dict(
    weights = "weights/DC/001-DroneCrowd-yolov7-tiny-320x512-crops-only-best.torchscript.pt",
    in_size = (320,512),
    conf_thresh = 0.01,
    iou_thresh = 0.65,
    transform = det_transform, # T.ToTensor(),
    postprocess=yolov7_postprocess,
)

yolov7_tiny_DC_crops_003 = dict(
    weights = "weights/DC/003-DroneCrowd-yolov7-tiny-512x512-crops-only-multiple-scales-100ep-best.torchscript.pt",
    in_size = (512,512),
    conf_thresh = 0.01,
    iou_thresh = 0.65,
    transform = det_transform, # T.ToTensor(),
    postprocess=yolov7_postprocess,
)

yolov7_tiny_DC_crops_004 = dict(
    weights = "weights/DC/004-DroneCrowd-yolov7-tiny-512x512-crops-only-multiple-scales-50ep-best.torchscript.pt",
    in_size = (512,512),
    conf_thresh = 0.01,
    iou_thresh = 0.65,
    transform = det_transform, # T.ToTensor(),
    postprocess=yolov7_postprocess,
)

yolov7_tiny_DC_crops_005 = dict(
    weights = "weights/DC/005-DroneCrowd-yolov7-tiny-512x512-crops-only-multiple-scales-10ep-best.torchscript.pt",
    in_size = (512,512),
    conf_thresh = 0.01,
    iou_thresh = 0.65,
    transform = det_transform, # T.ToTensor(),
    postprocess=yolov7_postprocess,
)

yolov7_tiny_DC_crops_006 = dict(
    weights = "weights/DC/006-DroneCrowd-yolov7-tiny-512x512-crops-only-multiple-scales-10ep-lr-best.torchscript.pt",
    in_size = (512,512),
    conf_thresh = 0.01,
    iou_thresh = 0.65,
    transform = det_transform, # T.ToTensor(),
    postprocess=yolov7_postprocess,
)

yolov4 = dict(
    weights = "weights/yolov4_MTSD.pt",
    in_size = (960,960),
    conf_thresh = 0.005,
    iou_thresh = 0.45,
    transform = T.ToTensor(),
    postprocess=yolov7_postprocess,
)


ROI_MODELS = {
    "u2net": u2net,
    "unet": unet,
    "unet-SDS0": unet_SDS0,
    "unet-SDS1": unet_SDS1,
    "unet-SDS2": unet_SDS2,
    "unet-SDS0-masks": unet_SDS0_masks,
    "unet-SDS1-masks": unet_SDS1_masks,
    "unet-SDS2-masks": unet_SDS2_masks,
    "unet-DC0": unet_DC0,
    "unet-DC1": unet_DC1,
    "unet-DC2": unet_DC2,
    "unet_SDS_tiny": unet_SDS_tiny,
    "unet_SDS_small": unet_SDS_small,
    "unet_SDS_medium": unet_SDS_medium,
    "unet_SDS_large": unet_SDS_large,
    "unet_DC1_tiny": unet_DC1_tiny,
    "unet_DC1_small": unet_DC1_small,
    "unet_DC1_medium": unet_DC1_medium,
    "unet_SDS1_trt": unet_SDS1_trt,
    "unet_SDS1_trt_fp16": unet_SDS1_trt_fp16,
    "unet_SDS1_trt_int8": unet_SDS1_trt_int8 
}


DET_MODELS = {
    "yolov7_tiny": yolov7_tiny,
    "yolov4": yolov4,
    "yolov7_tiny_DC": yolov7_tiny_DC,
    "yolov7_tiny_SDS": yolov7_tiny_SDS,
    "yolov7_tiny_SDS_crops": yolov7_tiny_SDS_crops,
    'yolov7_tiny_DC_crops': yolov7_tiny_DC_crops,
    'yolov7_tiny_DC_crops_rect': yolov7_tiny_DC_crops_rect,
    "yolov7_tiny_DC_crops_003": yolov7_tiny_DC_crops_003,
    "yolov7_tiny_DC_crops_004": yolov7_tiny_DC_crops_004,
    "yolov7_tiny_DC_crops_005": yolov7_tiny_DC_crops_005,
    "yolov7_tiny_DC_crops_006": yolov7_tiny_DC_crops_006,
    "yolov7_tiny_SDS_crops_mul_scales": yolov7_tiny_SDS_crops_mul_scales,
    "yolov7_tiny_SDS_crops_mul_scales_100": yolov7_tiny_SDS_crops_mul_scales_100,
    "yolov7_tiny_SDS_crops_mul_scales_100_best": yolov7_tiny_SDS_crops_mul_scales_100_best,
    "yolov7_tiny_SDS_crops_mul_scales_300": yolov7_tiny_SDS_crops_mul_scales_300,
    "yolov7_tiny_SDS_crops_mul_scales_100_bigger": yolov7_tiny_SDS_crops_mul_scales_100_bigger,
    "yolov7_SDS_crops_mul_scales": yolov7_SDS_crops_mul_scales,
    "yolov7_tiny_SDS_crops_mul_scales_300_trt": yolov7_tiny_SDS_crops_mul_scales_300_trt,
    "yolov7_tiny_SDS_crops_mul_scales_300_trt_fp16": yolov7_tiny_SDS_crops_mul_scales_300_trt_fp16,
    "yolov7_tiny_SDS_crops_mul_scales_300_trt_int8": yolov7_tiny_SDS_crops_mul_scales_300_trt_int8
}