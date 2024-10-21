import math 
import time

import cv2
import numpy as np

import torch
import torchvision.ops as ops
from torchvision import transforms as T



def yolo_preprocess(input_tensor, **kwargs):
    """
    Preprocesses the input tensor for YOLO model input.

    Args:
        input_tensor (torch.Tensor): The input tensor [B,C,H,W] representing the image or data to preprocess. 
        **kwargs: Additional keyword arguments for preprocessing options.

    Returns:
        torch.Tensor: The modified tensor in format [B,C,H,W]
    """
    return input_tensor


def box_iou(box1, box2):
    """Calculate Intersection over Union (IoU) for two sets of boxes.

    Args:
        box1 (Tensor[N, 4]): First set of boxes in (x1, y1, x2, y2) format.
        box2 (Tensor[M, 4]): Second set of boxes in (x1, y1, x2, y2) format.

    Returns:
        Tensor[N, M]: NxM matrix containing the pairwise IoU values for every 
        element in boxes1 and boxes2.
    """

    def box_area(box):
        """Calculate the area of the boxes.

        Args:
            box (Tensor): Box coordinates in (x1, y1, x2, y2) format.

        Returns:
            Tensor: Area of the boxes.
        """
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - 
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)

    return inter / (area1[:, None] + area2 - inter)  



def xywh2xyxy(x):
    """Convert bounding boxes from (center x, center y, width, height) to 
    (x1, y1, x2, y2) format.

    Args:
        x (Tensor or np.ndarray): nx4 boxes in [x, y, w, h] format.

    Returns:
        Tensor or np.ndarray: Converted boxes in [x1, y1, x2, y2] format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y



def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, merge=False,
                        labels=()):
    """
    Perform Non-Maximum Suppression (NMS) on inference results.
    This function is based on https://github.com/WongKinYiu/yolov7

    Args:
        prediction (Tensor): Predictions from the model.
        conf_thres (float): Confidence threshold for filtering boxes.
        iou_thres (float): IoU threshold for NMS.
        classes (list, optional): Filter by class.
        agnostic (bool, optional): If True, class-agnostic NMS is applied.
        multi_label (bool, optional): If True, allows multiple labels per box.
        merge (bool, optional): If True, merges overlapping boxes.
        labels (tuple, optional): Ground truth labels for the image.

    Returns:
        List[Tensor]: List of detections, where each detection is a 
        (n, 6) tensor per image [xyxy, conf, cls].
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 500  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    # merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc == 1:
            x[:, 5:] = x[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output



def yolo_postprocess(output, conf_thresh=0.25, iou_thresh=0.45, classes=None, agnostic=False, multi_label=False, merge=False, labels=()):
    """
    Post-process the YOLO model output.

    Args:
        output (torch.Tensor): The output tensor from the YOLO model.
        conf_thres (float): Confidence threshold for filtering boxes.
        iou_thres (float): IoU threshold for NMS.
        classes (list, optional): Filter by class.
        agnostic (bool, optional): If True, class-agnostic NMS is applied.
        multi_label (bool, optional): If True, allows multiple labels per box.
        merge (bool, optional): If True, merges overlapping boxes.
        labels (tuple, optional): Ground truth labels for the image.

    Returns:
        Returns:
        List[Tensor]: List of detections, where each detection is a 
        (n, 6) tensor per image [xyxy, conf, cls].
    """
    output = output[0]
    output = non_max_suppression(output, conf_thres=conf_thresh, iou_thres=iou_thresh, classes=classes, agnostic=agnostic, multi_label=multi_label, merge=merge, labels=labels)
    return output




"""
Dictionaries that contain the configurations for various YOLO detection models. 
All weights are torch.jit.ScriptModules to avoid model dependencies.

Keys:
- weights: Path to the pre-trained model weights (torch.jit.ScriptModule).
- in_size: Image dimensions for resizing during preprocessing.
- preprocess: Preprocessing function to be applied to the input image.
- preprocess_args: Arguments for preprocess function.
- postprocess: Post-processing function applied to the model outputs.
- postprocess_args: Arguments for postprocess function.
- classes: Class names.
- colors: Class colors for visualisation.
"""


ZeF20 = dict(
    weights = "weights/yolov7t-ZebraFish.pt",
    in_size = (160,256),
    preprocess = yolo_preprocess,
    preprocess_args = dict(),
    postprocess = yolo_postprocess,
    postprocess_args = dict(
        conf_thresh = 0.01,
        iou_thresh = 0.65,
        multi_label = True,
        labels = [],
        merge = False,
        agnostic = False,
    ),
    classes = ["fish"],
    colors = [(206, 75, 25)],
)


SeaDronesSee = dict(
    weights = "weights/006-SeaDronesSee-yolov7-tiny-512x512-crops-only-multiple-scales-300ep-best.torchscript.pt",
    in_size = (512,512),
    preprocess = yolo_preprocess,
    preprocess_args = dict(),
    postprocess = yolo_postprocess,
    postprocess_args = dict(
        conf_thresh = 0.01,
        iou_thresh = 0.65,
        multi_label = True,
        labels = [],
        merge = True,
        agnostic = False,
    ),
    classes = ['swimmer', 'swimmer with life jacket', 'boat', 'life jacket'],
    colors = [(203, 179, 11), (222, 135, 191), (40, 195, 132), (75, 140, 112)],
)



DroneCrowd = dict(
    weights = "weights/004-DroneCrowd-yolov7-tiny-512x512-crops-only-multiple-scales-50ep-best.torchscript.pt",
    in_size = (512,512),
    preprocess = yolo_preprocess,
    preprocess_args = dict(),
    postprocess = yolo_postprocess,
    postprocess_args = dict(
        conf_thresh = 0.01,
        iou_thresh = 0.65,
        multi_label = True,
        labels = [],
        merge = True,
        agnostic = False,
    ),
    classes = ["human"],
    colors = [(73, 200 , 147)],
)

MTSD = dict(
    weights = "weights/yolov4_MTSD.pt",
    in_size = (960,960),
    preprocess = yolo_preprocess,
    preprocess_args = dict(),
    postprocess = yolo_postprocess,
    postprocess_args = dict(
        conf_thresh = 0.01,
        iou_thresh = 0.45,
        multi_label = True,
        labels = [],
        merge = True,
        agnostic = False,
    ),
    classes = ['warning--double-curve-first-right--g2', 'regulatory--turn-left-ahead--g1', 'complementary--obstacle-delineator--g2', 'information--telephone--g1', 'regulatory--end-of-bicycles-only--g1', 'information--road-bump--g1', 'warning--children--g5', 'regulatory--no-heavy-goods-vehicles--g1', 'regulatory--no-pedestrians--g2', 'regulatory--end-of-maximum-speed-limit-70--g1', 'regulatory--radar-enforced--g1', 'warning--uneven-road--g1', 'regulatory--weight-limit--g1', 'regulatory--bicycles-only--g3', 'complementary--buses--g1', 'warning--divided-highway-ends--g2', 'warning--curve-left--g2', 'warning--horizontal-alignment-left--g1', 'information--parking--g3', 'regulatory--buses-only--g1', 'regulatory--left-turn-yield-on-green--g1', 'complementary--go-left--g1', 'regulatory--no-stopping--g2', 'warning--children--g2', 'complementary--trucks--g1', 'warning--traffic-signals--g2', 'regulatory--no-bicycles--g1', 'regulatory--no-left-turn--g2', 'information--hospital--g1', 'regulatory--lane-control--g1', 'information--tram-bus-stop--g2', 'regulatory--turn-left--g1', 'regulatory--no-u-turn--g2', 'warning--turn-right--g1', 'regulatory--turn-left--g3', 'complementary--distance--g1', 'regulatory--no-heavy-goods-vehicles--g5', 'regulatory--pedestrians-only--g2', 'warning--narrow-bridge--g1', 'regulatory--no-stopping--g5', 'warning--crossroads--g3', 'regulatory--turn-right-ahead--g1', 'regulatory--dual-lanes-go-straight-on-left--g1', 'regulatory--wrong-way--g1', 'complementary--go-right--g2', 'regulatory--maximum-speed-limit-led-80--g1', 'warning--railroad-crossing-without-barriers--g1', 'warning--winding-road-first-right--g1', 'regulatory--shared-path-pedestrians-and-bicycles--g1', 'regulatory--dual-lanes-go-straight-on-right--g1', 'regulatory--maximum-speed-limit-25--g2', 'warning--texts--g1', 'regulatory--no-motor-vehicles-except-motorcycles--g2', 'warning--narrow-bridge--g3', 'warning--curve-right--g1', 'regulatory--one-way-straight--g1', 'regulatory--no-parking-or-no-stopping--g1', 'regulatory--passing-lane-ahead--g1', 'regulatory--turn-right--g3', 'regulatory--road-closed-to-vehicles--g3', 'regulatory--go-straight-or-turn-right--g1', 'regulatory--no-left-turn--g1', 'warning--dual-lanes-right-turn-or-go-straight--g1', 'warning--roadworks--g4', 'complementary--trucks-turn-right--g1', 'warning--crossroads-with-priority-to-the-right--g1', 'warning--two-way-traffic--g2', 'warning--slippery-road-surface--g2', 'warning--road-bump--g2', 'regulatory--end-of-priority-road--g1', 'information--motorway--g1', 'regulatory--no-heavy-goods-vehicles--g2', 'warning--winding-road-first-right--g3', 'information--pedestrians-crossing--g1', 'warning--pedestrians-crossing--g4', 'warning--railroad-crossing--g4', 'regulatory--go-straight-or-turn-left--g3', 'regulatory--no-parking-or-no-stopping--g5', 'warning--curve-right--g2', 'regulatory--maximum-speed-limit-5--g1', 'complementary--chevron-right--g3', 'regulatory--no-parking-or-no-stopping--g2', 'regulatory--maximum-speed-limit-10--g1', 'regulatory--no-buses--g3', 'regulatory--pass-on-either-side--g1', 'regulatory--yield--g1', 'regulatory--keep-right--g2', 'complementary--distance--g3', 'information--dead-end--g1', 'warning--added-lane-right--g1', 'regulatory--go-straight--g1', 'warning--stop-ahead--g1', 'warning--curve-left--g1', 'complementary--except-bicycles--g1', 'warning--traffic-merges-right--g2', 'regulatory--maximum-speed-limit-120--g1', 'regulatory--parking-restrictions--g2', 'information--bike-route--g1', 'complementary--chevron-right--g5', 'regulatory--no-motor-vehicles-except-motorcycles--g1', 'warning--t-roads--g2', 'regulatory--keep-right--g4', 'warning--road-narrows-left--g2', 'regulatory--no-overtaking--g4', 'complementary--go-right--g1', 'regulatory--no-right-turn--g2', 'warning--road-narrows-right--g2', 'complementary--chevron-right--g2', 'regulatory--no-parking--g1', 'warning--y-roads--g1', 'regulatory--no-motorcycles--g2', 'regulatory--go-straight--g3', 'regulatory--maximum-speed-limit-70--g1', 'regulatory--no-overtaking--g5', 'regulatory--no-u-turn--g3', 'regulatory--maximum-speed-limit-110--g1', 'warning--junction-with-a-side-road-acute-left--g1', 'complementary--maximum-speed-limit-35--g1', 'warning--roadworks--g6', 'regulatory--one-way-left--g1', 'complementary--both-directions--g2', 'regulatory--stop-here-on-red-or-flashing-light--g1', 'regulatory--no-turn-on-red--g2', 'warning--slippery-road-surface--g1', 'complementary--chevron-right--g1', 'complementary--both-directions--g1', 'regulatory--stop--g10', 'complementary--maximum-speed-limit-75--g1', 'warning--traffic-signals--g3', 'regulatory--no-overtaking--g1', 'warning--school-zone--g2', 'complementary--chevron-right--g4', 'warning--road-bump--g1', 'regulatory--priority-over-oncoming-vehicles--g1', 'complementary--maximum-speed-limit-55--g1', 'warning--traffic-signals--g1', 'complementary--one-direction-left--g1', 'information--parking--g5', 'warning--pedestrians-crossing--g12', 'regulatory--no-right-turn--g1', 'regulatory--maximum-speed-limit-100--g3', 'complementary--turn-right--g2', 'warning--children--g1', 'regulatory--maximum-speed-limit-45--g3', 'warning--traffic-merges-left--g1', 'complementary--tow-away-zone--g1', 'regulatory--one-way-right--g3', 'regulatory--give-way-to-oncoming-traffic--g1', 'regulatory--no-parking--g5', 'regulatory--pass-on-either-side--g2', 'regulatory--no-turn-on-red--g1', 'regulatory--no-parking--g2', 'information--parking--g2', 'regulatory--keep-right--g6', 'regulatory--no-motorcycles--g1', 'warning--roadworks--g2', 'regulatory--pedestrians-only--g1', 'information--living-street--g1', 'warning--domestic-animals--g1', 'information--interstate-route--g1', 'warning--divided-highway-ends--g1', 'regulatory--bicycles-only--g1', 'information--gas-station--g3', 'warning--flaggers-in-road--g1', 'information--disabled-persons--g1', 'information--children--g1', 'warning--junction-with-a-side-road-perpendicular-right--g3', 'regulatory--do-not-block-intersection--g1', 'complementary--maximum-speed-limit-15--g1', 'warning--roadworks--g3', 'regulatory--reversible-lanes--g2', 'regulatory--no-heavy-goods-vehicles--g4', 'regulatory--end-of-no-parking--g1', 'regulatory--no-entry--g1', 'warning--pedestrians-crossing--g5', 'information--highway-interstate-route--g2', 'information--safety-area--g2', 'complementary--maximum-speed-limit-25--g1', 'regulatory--keep-left--g1', 'complementary--keep-right--g1', 'complementary--pass-right--g1', 'warning--railroad-crossing--g1', 'regulatory--one-way-right--g2', 'warning--junction-with-a-side-road-perpendicular-left--g1', 'complementary--maximum-speed-limit-30--g1', 'regulatory--no-u-turn--g1', 'warning--junction-with-a-side-road-perpendicular-left--g3', 'warning--wild-animals--g1', 'regulatory--no-overtaking-by-heavy-goods-vehicles--g1', 'warning--railroad-crossing-without-barriers--g3', 'regulatory--mopeds-and-bicycles-only--g1', 'warning--trucks-crossing--g1', 'regulatory--triple-lanes-turn-left-center-lane--g1', 'regulatory--go-straight-or-turn-left--g1', 'regulatory--keep-right--g1', 'information--highway-exit--g1', 'warning--junction-with-a-side-road-acute-right--g1', 'warning--roundabout--g1', 'regulatory--go-straight-or-turn-right--g3', 'warning--double-turn-first-right--g1', 'complementary--one-direction-right--g1', 'regulatory--go-straight-or-turn-left--g2', 'regulatory--no-bicycles--g2', 'warning--texts--g2', 'complementary--chevron-left--g5', 'regulatory--maximum-speed-limit-led-60--g1', 'warning--pass-left-or-right--g2', 'regulatory--no-parking-or-no-stopping--g3', 'warning--double-reverse-curve-right--g1', 'warning--kangaloo-crossing--g1', 'regulatory--dual-path-bicycles-and-pedestrians--g1', 'warning--height-restriction--g2', 'warning--bicycles-crossing--g1', 'regulatory--maximum-speed-limit-90--g1', 'warning--horizontal-alignment-right--g1', 'information--limited-access-road--g1', 'warning--roadworks--g1', 'regulatory--maximum-speed-limit-20--g1', 'warning--double-curve-first-left--g1', 'regulatory--maximum-speed-limit-55--g2', 'complementary--chevron-left--g1', 'warning--road-widens--g1', 'regulatory--maximum-speed-limit-35--g2', 'complementary--maximum-speed-limit-40--g1', 'regulatory--no-stopping--g1', 'warning--railroad-crossing-without-barriers--g4', 'regulatory--maximum-speed-limit-40--g1', 'warning--crossroads--g1', 'warning--railroad-crossing--g3', 'warning--hairpin-curve-right--g1', 'regulatory--maximum-speed-limit-30--g3', 'regulatory--stop-here-on-red-or-flashing-light--g2', 'regulatory--stop--g2', 'other-sign', 'regulatory--maximum-speed-limit-50--g1', 'complementary--turn-left--g2', 'warning--junction-with-a-side-road-perpendicular-right--g1', 'information--parking--g6', 'warning--texts--g3', 'information--end-of-motorway--g1', 'information--airport--g2', 'warning--roundabout--g2', 'regulatory--stop--g1', 'warning--other-danger--g1', 'warning--road-narrows--g2', 'information--airport--g1', 'information--emergency-facility--g2', 'regulatory--maximum-speed-limit-30--g1', 'regulatory--no-pedestrians--g1', 'regulatory--shared-path-bicycles-and-pedestrians--g1', 'warning--falling-rocks-or-debris-right--g2', 'regulatory--maximum-speed-limit-100--g1', 'warning--turn-left--g1', 'regulatory--end-of-prohibition--g1', 'information--bus-stop--g1', 'warning--double-curve-first-right--g1', 'warning--road-widens-right--g1', 'information--gas-station--g1', 'regulatory--bicycles-only--g2', 'regulatory--one-way-left--g2', 'information--parking--g1', 'regulatory--u-turn--g1', 'information--end-of-built-up-area--g1', 'warning--traffic-merges-right--g1', 'warning--railroad-intersection--g3', 'complementary--chevron-left--g4', 'regulatory--no-straight-through--g1', 'complementary--obstacle-delineator--g1', 'regulatory--road-closed--g2', 'warning--trail-crossing--g2', 'warning--two-way-traffic--g1', 'regulatory--turn-right--g1', 'warning--steep-descent--g2', 'regulatory--turn-left--g2', 'regulatory--one-way-right--g1', 'complementary--keep-left--g1', 'information--food--g2', 'complementary--distance--g2', 'regulatory--priority-road--g1', 'regulatory--no-overtaking--g2', 'regulatory--maximum-speed-limit-led-100--g1', 'warning--road-narrows--g1', 'warning--railroad-crossing-with-barriers--g1', 'warning--winding-road-first-left--g1', 'complementary--maximum-speed-limit-45--g1', 'regulatory--one-way-left--g3', 'complementary--chevron-left--g2', 'warning--road-narrows-left--g1', 'regulatory--end-of-speed-limit-zone--g1', 'warning--emergency-vehicles--g1', 'complementary--maximum-speed-limit-20--g1', 'regulatory--stop-signals--g1', 'regulatory--road-closed-to-vehicles--g1', 'complementary--maximum-speed-limit-70--g1', 'warning--bicycles-crossing--g2', 'information--pedestrians-crossing--g2', 'regulatory--keep-left--g2', 'regulatory--roundabout--g1', 'regulatory--no-turn-on-red--g3', 'warning--double-curve-first-left--g2', 'regulatory--no-parking--g6', 'information--telephone--g2', 'complementary--chevron-left--g3', 'warning--pedestrians-crossing--g1', 'regulatory--turn-right--g2', 'warning--road-narrows-right--g1', 'warning--pedestrians-crossing--g10', 'information--trailer-camping--g1', 'regulatory--no-vehicles-carrying-dangerous-goods--g1', 'warning--railroad-crossing-with-barriers--g2', 'regulatory--maximum-speed-limit-60--g1', 'regulatory--height-limit--g1', 'regulatory--maximum-speed-limit-40--g3', 'regulatory--maximum-speed-limit-80--g1'],
    colors = [(106, 153, 192), (1, 207, 118), (7, 247, 192), (74, 249, 237), (213, 151, 121), (224, 191, 56), (68, 149, 234), (35, 219, 9), 
        (186, 215, 189), (198, 114, 167), (164, 182, 111), (111, 13, 161), (101, 125, 219), (79, 86, 231), (241, 226, 214), (60, 202, 157), 
        (20, 59, 6), (195, 234, 239), (242, 31, 165), (90, 42, 51), (114, 157, 32), (200, 238, 133), (37, 60, 202), (223, 25, 50), (126, 105, 79), 
        (160, 195, 234), (164, 198, 245), (24, 108, 160), (74, 55, 237), (138, 70, 215), (124, 194, 198), (147, 239, 209), (86, 68, 238), (135, 208, 40), 
        (71, 48, 8), (16, 10, 153), (254, 203, 161), (101, 203, 167), (22, 252, 229), (93, 124, 199), (138, 250, 150), (77, 213, 199), (191, 186, 189), 
        (154, 41, 139), (123, 101, 150), (149, 192, 34), (99, 215, 175), (103, 147, 11), (162, 91, 158), (195, 22, 202), (77, 70, 224), (254, 245, 184), 
        (76, 118, 27), (126, 250, 215), (105, 52, 150), (163, 158, 1), (136, 204, 224), (75, 164, 172), (100, 234, 232), (148, 209, 40), (251, 93, 34), 
        (27, 178, 187), (142, 21, 195), (170, 222, 225), (161, 183, 233), (210, 51, 250), (235, 188, 9), (16, 31, 116), (155, 11, 156), (43, 80, 226), 
        (120, 77, 235), (71, 98, 180), (58, 240, 64), (136, 180, 249), (158, 118, 41), (2, 96, 202), (198, 152, 242), (110, 172, 250), (100, 187, 63), 
        (67, 235, 147), (212, 130, 237), (90, 197, 93), (65, 181, 210), (196, 190, 44), (127, 166, 199), (110, 18, 23), (145, 115, 23), (142, 15, 108), 
        (144, 176, 232), (192, 27, 138), (25, 34, 169), (92, 176, 166), (43, 252, 252), (103, 189, 40), (18, 214, 179), (125, 129, 11), (209, 41, 240), 
        (250, 98, 201), (149, 251, 96), (129, 127, 114), (59, 91, 81), (81, 194, 29), (74, 61, 245), (75, 43, 61), (111, 226, 8), (31, 253, 148), 
        (180, 253, 23), (2, 244, 42), (4, 11, 178), (160, 75, 15), (198, 136, 92), (121, 70, 69), (66, 18, 54), (10, 120, 65), (179, 11, 66), (229, 28, 128), 
        (137, 131, 198), (85, 223, 99), (130, 64, 237), (70, 17, 111), (12, 21, 63), (195, 210, 234), (188, 234, 104), (1, 240, 73), (165, 194, 118), 
        (138, 23, 244), (104, 98, 110), (37, 162, 74), (238, 152, 66), (135, 250, 22), (241, 101, 74), (54, 81, 102), (216, 17, 165), (123, 218, 137), 
        (167, 188, 32), (79, 91, 187), (76, 120, 182), (23, 21, 30), (193, 115, 243), (198, 151, 112), (5, 128, 97), (250, 141, 87), (184, 251, 155), 
        (101, 237, 213), (23, 231, 221), (47, 26, 211), (160, 213, 13), (104, 226, 33), (247, 190, 203), (160, 115, 74), (228, 168, 114), (28, 104, 93), 
        (5, 229, 18), (82, 18, 8), (153, 210, 118), (50, 102, 114), (11, 134, 20), (157, 197, 241), (103, 137, 217), (239, 253, 174), (23, 94, 62), 
        (194, 100, 225), (205, 79, 177), (29, 46, 131), (50, 192, 193), (91, 140, 95), (222, 140, 64), (45, 120, 69), (130, 186, 32), (185, 249, 25), 
        (31, 93, 50), (112, 20, 103), (116, 214, 251), (102, 177, 48), (108, 238, 181), (2, 93, 49), (250, 208, 179), (175, 23, 64), (119, 155, 197), 
        (3, 223, 68), (52, 52, 28), (224, 215, 152), (30, 207, 44), (207, 190, 113), (231, 82, 239), (18, 91, 178), (135, 38, 153), (192, 112, 4), 
        (81, 62, 197), (110, 232, 199), (33, 248, 203), (121, 189, 141), (212, 84, 188), (245, 41, 161), (26, 223, 93), (164, 52, 158), (209, 98, 25), 
        (237, 194, 20), (170, 134, 35), (202, 177, 74), (45, 93, 231), (126, 137, 169), (158, 221, 186), (98, 94, 183), (1, 126, 131), (127, 96, 222), 
        (180, 14, 20), (65, 114, 71), (127, 26, 46), (88, 165, 1), (0, 56, 9), (216, 108, 159), (193, 241, 211), (202, 221, 171), (59, 199, 127), 
        (253, 233, 87), (55, 236, 122), (162, 93, 213), (157, 129, 83), (153, 93, 104), (99, 10, 94), (73, 215, 182), (197, 219, 158), (144, 110, 26), 
        (102, 132, 106), (147, 167, 155), (162, 217, 35), (109, 220, 21), (142, 254, 140), (5, 162, 252), (80, 102, 189), (123, 161, 225), (167, 169, 222), 
        (128, 158, 150), (179, 52, 214), (78, 234, 86), (87, 25, 145), (44, 241, 48), (171, 139, 69), (3, 52, 63), (23, 109, 116), (149, 247, 115), 
        (183, 37, 70), (193, 58, 91), (97, 232, 220), (112, 77, 245), (160, 2, 122), (26, 133, 122), (95, 220, 121), (121, 129, 136), (183, 41, 190), 
        (0, 67, 234), (71, 103, 76), (46, 172, 227), (178, 188, 202), (229, 32, 142), (40, 159, 50), (88, 152, 124), (89, 153, 181), (72, 145, 123), 
        (234, 54, 138), (204, 4, 102), (85, 15, 168), (108, 97, 130), (188, 180, 120), (127, 0, 52), (127, 2, 67), (226, 158, 211), (165, 85, 135), 
        (96, 182, 118), (88, 254, 162), (147, 71, 121), (67, 155, 26), (246, 144, 35), (86, 118, 28), (235, 29, 161), (122, 21, 100), (106, 28, 71), 
        (130, 96, 121), (164, 168, 209), (105, 210, 116), (105, 18, 9), (235, 122, 74), (119, 111, 80), (216, 44, 199), (144, 8, 22), (123, 196, 75), 
        (29, 26, 158), (235, 219, 226), (191, 77, 116), (139, 85, 75), (11, 25, 146), (253, 66, 171), (93, 226, 144), (136, 115, 20), (130, 70, 138), 
        (146, 239, 139), (253, 4, 253), (97, 171, 134), (159, 147, 55), (251, 94, 248), (62, 210, 209), (223, 97, 4), (24, 161, 22), (208, 233, 44), 
        (63, 233, 154), (237, 128, 17), (7, 45, 109), (26, 163, 13), (18, 191, 168), (129, 142, 123), (229, 34, 187), (121, 81, 9), (20, 26, 104)],
)
