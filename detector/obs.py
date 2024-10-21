import torch



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

    


def overlapping_box_suppression(windows, bboxes, th=0.6):
    """Perform overlapping box suppression to remove redundant bounding boxes.

    This function removes redundant bounding boxes based on their Intersection over Union (IoU) 
    with the given thresholds. 

    Args:
        windows (torch.Tensor): A tensor of shape [N, 4] representing the bounding boxes to filter, 
            where each box is defined by (xmin, ymin, xmax, ymax).
        bboxes (torch.Tensor): A tensor of shape [M, 5] representing the bounding boxes with confidence scores, 
            where each box is defined by (xmin, ymin, xmax, ymax, confidence).
        th (float): IoU threshold for determining whether to suppress a box. Default is 0.6.

    Returns:
        bboxes_filtered (torch.Tensor): The filtered bounding boxes tensor with shape [M', 5], 
            where M' is the number of bounding boxes that are not suppressed.
    """

    def normalize(input_tensor):
        """Normalize a tensor to the range [0, 1].

        Args:
            input_tensor (torch.Tensor): The input tensor to normalize.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        input_tensor -= input_tensor.min(0, keepdim=True)[0]
        input_tensor /= input_tensor.max(0, keepdim=True)[0]
        return input_tensor


    if len(bboxes) == 0:
        return bboxes


    unique_windows = torch.unique(windows, dim=0)

    # Calculate intersections
    intersection_maxs = torch.min(bboxes[:, None, 2:4], unique_windows[:, 2:4])  # xmax, ymax
    intersection_mins = torch.max(bboxes[:, None, :2], unique_windows[:, :2])  # xmin, ymin
    intersections = torch.cat((intersection_mins, intersection_maxs), dim=2)

    # Set intersections with no common area to 0
    intersections[(intersections[:,:,2] - intersections[:,:,0] < 0) | (intersections[:,:,3] - intersections[:,:,1] < 0)] = 0 # no common area
    
    # Set detections from the window to 0
    window_matches = (windows[:, None] == unique_windows).all(dim=-1) 
    win_inds = torch.nonzero(window_matches, as_tuple=True)
    intersections[win_inds[0], win_inds[1], :] = 0
    
    # Calculate IoU matrix
    ious = torch.stack([box_iou(intersections[:, i, :], bboxes[:, :4]) for i in range(intersections.shape[1])], dim=1)
        
    # Set diagonal to 0 (a detection cannot be removed because of itself)
    ious.diagonal(dim1=0, dim2=2).zero_()
    
    # Identify to-delete detections
    to_del = torch.nonzero(ious > th)
    ious = ious[ious > th]
    if not ious.numel():
        return bboxes


    det_ind = to_del[:, 2].to(int)
    
    # Extract confidence scores and areas
    bboxes_det = bboxes[det_ind]
    confs = bboxes_det[:, 4]
    areas = (bboxes_det[:, 2] - bboxes_det[:, 0]) * (bboxes_det[:, 3] - bboxes_det[:, 1])

    # Normalize
    confs = 1 - normalize(confs)
    areas = 1 - normalize(areas)
    ious_vals = normalize(ious)

    # Calculate mean values
    mean_vals = torch.mean(torch.stack([ious_vals, confs, areas]), dim=0)

    # Sort to delete by mean values
    to_del = torch.hstack((to_del, mean_vals.unsqueeze(-1)))
    to_del = to_del[to_del[:, -1].sort(descending=True)[1]].int() # (N, 4)

    
    # Remove detections
    to_del_ids = set()
    del_mask = torch.zeros(bboxes.shape[0], dtype=torch.bool)
    for i in range(to_del.shape[0]):
        det_idx = to_del[i, 2].item()
        if det_idx in to_del_ids:
            continue
        if to_del[i, 0].item() in to_del_ids:
            continue
        to_del_ids.add(det_idx)
        del_mask[det_idx] = True

    # Filter bboxes
    bboxes_filtered = bboxes[~del_mask]

    return bboxes_filtered