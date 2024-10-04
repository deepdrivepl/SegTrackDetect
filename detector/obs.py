import torch

from .aggregation import box_iou


def overlapping_box_suppression(windows, bboxes, th=0.6):
    
    def normalize(input_tensor):
        input_tensor -= input_tensor.min(0, keepdim=True)[0]
        input_tensor /= input_tensor.max(0, keepdim=True)[0]
        return input_tensor

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
        return windows, bboxes


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

    # Filter windows and bboxes
    windows_filtered = windows[~del_mask]
    bboxes_filtered = bboxes[~del_mask]

    return windows_filtered, bboxes_filtered