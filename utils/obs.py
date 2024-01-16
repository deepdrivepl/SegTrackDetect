import torch

from .bboxes import box_iou


def filter_dets(windows, detections, th=0.7):
    filtered_detections, filtered_windows = torch.empty((0,6)), torch.empty((0,4))
    
    unique_windows = torch.unique(windows, dim=0)
    for w, unique_window in enumerate(unique_windows):
        ind_win = torch.unique(torch.where(windows==unique_window)[0])
        ind_notwin = torch.unique(torch.where(windows!=unique_window)[0])
        window_dets = detections[ind_win,:]
        other_dets = detections[ind_notwin,:]
        
        intersection_maxs = torch.min(other_dets[:, None, 2:4], unique_window.unsqueeze(0)[:, 2:4]) # xmax, ymax
        intersection_mins = torch.max(other_dets[:, None, :2], unique_window.unsqueeze(0)[:, :2]) # xmin, ymin
        intersections = torch.flatten(torch.cat((intersection_mins, intersection_maxs), dim=2), start_dim=0, end_dim=1)
        intersections[(intersections[:,2] - intersections[:,0] < 0) | (intersections[:,3] - intersections[:,1] < 0)] = 0

        ious = box_iou(intersections, window_dets[:,:4])
        to_del = torch.where(ious > th)[1]
        dets_filtered = window_dets[[x for x in range(window_dets.shape[0]) if x not in to_del],:]
        filtered_detections = torch.cat((filtered_detections, dets_filtered.to(filtered_detections.device)))
        filtered_detections = torch.unique(filtered_detections, dim=0)
        
        filtered_windows = torch.cat((filtered_windows.to(unique_window.device), unique_window.repeat(dets_filtered.shape[0], 1)))
        
        # avoid filtering "same detection" twice (same objects)
        to_del_orig = []
        for r,row in enumerate(detections):
            if row.tolist() in to_del.tolist():
                to_del_orig.append(r)
                
        detections = detections[[x for x in range(detections.shape[0]) if x not in to_del_orig],:]
        windows = windows[[x for x in range(windows.shape[0]) if x not in to_del_orig],:]
    return filtered_windows, filtered_detections   



def filter_dets_single_matrix(windows, bboxes, th=0.7):
    unique_windows = torch.unique(windows, dim=0)
    
    
    intersection_maxs = torch.min(bboxes[:, None, 2:4], unique_windows[:, 2:4]) # xmax, ymax
    intersection_mins = torch.max(bboxes[:, None, :2], unique_windows[:, :2]) # xmin, ymin
    intersections = torch.cat((intersection_mins, intersection_maxs), dim=2)
    intersections[(intersections[:,:,2] - intersections[:,:,0] < 0) | (intersections[:,:,3] - intersections[:,:,1] < 0)] = 0 # no common area
    
    # detections from window = 0
    for i, unique_window in enumerate(unique_windows):
        win_ind = torch.unique(torch.where(windows==unique_window)[0])
        intersections[win_ind, i, :] = 0
    # print(intersections.shape, bboxes.shape)
    
    ious = torch.empty((len(bboxes), len(unique_windows), len(bboxes)))
    for i in range(intersections.shape[1]):
        ious[:,i,...] = box_iou(intersections[:,i,:], bboxes[:,:4])
        
    for i in range(bboxes.shape[0]): # a detection cannot be removed because of itself
        ious[i,:,i] = 0
    
    #print(ious)
    to_del = torch.nonzero(ious > th)
    ious = ious[ious > th]

    to_del = torch.hstack((to_del, ious.unsqueeze(-1)))
    to_del = to_del[to_del[:, -1].sort(descending=True)[1]] # (N, 4)
    #print(to_del)
    
    to_del_ids = []
    for i in range(to_del.shape[0]):
        if to_del[i, 0].item() in to_del_ids:
            #print(f'Cause already deleted {to_del[i, 0]}')
            continue
        if to_del[i, 2].item() in to_del_ids:
            #print(f'Detection already deleted {to_del[i, 2]}')
            continue
        to_del_ids.append(int(to_del[i, 2].item()))
        #print(f'Delete {to_del[i, 2]} because of {to_del[i, 0]}, iou: {to_del[i, 3]}')
    bboxes_filtered = bboxes[[x for x in range(bboxes.shape[0]) if x not in to_del_ids],:]
    windows_filtered = windows[[x for x in range(windows.shape[0]) if x not in to_del_ids],:]
    # print(bboxes_filtered)
    return windows_filtered, bboxes_filtered


def filter_dets_single_matrix_all(windows, bboxes, th=0.6):
    
    def normalize(input_tensor):
        input_tensor -= input_tensor.min(0, keepdim=True)[0]
        input_tensor /= input_tensor.max(0, keepdim=True)[0]
        return input_tensor
    
    unique_windows = torch.unique(windows, dim=0)
    
    intersection_maxs = torch.min(bboxes[:, None, 2:4], unique_windows[:, 2:4]) # xmax, ymax
    intersection_mins = torch.max(bboxes[:, None, :2], unique_windows[:, :2]) # xmin, ymin
    intersections = torch.cat((intersection_mins, intersection_maxs), dim=2)
    intersections[(intersections[:,:,2] - intersections[:,:,0] < 0) | (intersections[:,:,3] - intersections[:,:,1] < 0)] = 0 # no common area
    
    # detections from window = 0
    for i, unique_window in enumerate(unique_windows):
        win_ind = torch.unique(torch.where(windows==unique_window)[0])
        intersections[win_ind, i, :] = 0
    # print(intersections.shape, bboxes.shape)
    
    ious = torch.empty((len(bboxes), len(unique_windows), len(bboxes)))
    for i in range(intersections.shape[1]):
        ious[:,i,...] = box_iou(intersections[:,i,:], bboxes[:,:4])
        
    for i in range(bboxes.shape[0]): # a detection cannot be removed because of itself
        ious[i,:,i] = 0
    
    #print(ious)
    to_del = torch.nonzero(ious > th)
    ious = ious[ious > th]
    if not ious.numel():
        return windows, bboxes
    
    det_ind = to_del[:,2].to(int)
    confs, areas = [], []
    for i in det_ind:
        xmin,ymin,xmax,ymax,conf = bboxes[i,:-1]
        confs.append(conf.item())
        areas.append((xmax-xmin)*(ymax-ymin))
        
    confs = 1 - normalize(torch.tensor(confs).unsqueeze(-1))
    areas = 1 - normalize(torch.tensor(areas).unsqueeze(-1))
    ious = normalize(ious.unsqueeze(-1))
    mean = torch.mean(torch.stack([ious,confs, areas]), 0)

    to_del = torch.hstack((to_del, mean))
    to_del = to_del[to_del[:, -1].sort(descending=True)[1]] # (N, 4)
    #print(to_del)
    
    to_del_ids = []
    for i in range(to_del.shape[0]):
        if to_del[i, 0].item() in to_del_ids:
            #print(f'Cause already deleted {to_del[i, 0]}')
            continue
        if to_del[i, 2].item() in to_del_ids:
            #print(f'Detection already deleted {to_del[i, 2]}')
            continue
        to_del_ids.append(int(to_del[i, 2].item()))
        #print(f'Delete {to_del[i, 2]} because of {to_del[i, 0]}, iou: {to_del[i, 3]}')
    bboxes_filtered = bboxes[[x for x in range(bboxes.shape[0]) if x not in to_del_ids],:]
    windows_filtered = windows[[x for x in range(windows.shape[0]) if x not in to_del_ids],:]
    # print(bboxes_filtered)
    return windows_filtered, bboxes_filtered


def filter_dets_single_matrix_iou(windows, bboxes, th=0.7):
    
    def normalize(input_tensor):
        input_tensor -= input_tensor.min(0, keepdim=True)[0]
        input_tensor /= input_tensor.max(0, keepdim=True)[0]
        return input_tensor
    
    unique_windows = torch.unique(windows, dim=0)
    
    intersection_maxs = torch.min(bboxes[:, None, 2:4], unique_windows[:, 2:4]) # xmax, ymax
    intersection_mins = torch.max(bboxes[:, None, :2], unique_windows[:, :2]) # xmin, ymin
    intersections = torch.cat((intersection_mins, intersection_maxs), dim=2)
    intersections[(intersections[:,:,2] - intersections[:,:,0] < 0) | (intersections[:,:,3] - intersections[:,:,1] < 0)] = 0 # no common area
    
    # detections from window = 0
    for i, unique_window in enumerate(unique_windows):
        win_ind = torch.unique(torch.where(windows==unique_window)[0])
        intersections[win_ind, i, :] = 0
    # print(intersections.shape, bboxes.shape)
    
    ious = torch.empty((len(bboxes), len(unique_windows), len(bboxes)))
    for i in range(intersections.shape[1]):
        ious[:,i,...] = box_iou(intersections[:,i,:], bboxes[:,:4])
        
    for i in range(bboxes.shape[0]): # a detection cannot be removed because of itself
        ious[i,:,i] = 0
    
    #print(ious)
    to_del = torch.nonzero(ious > th)
    ious = ious[ious > th]
    if not ious.numel():
        return windows, bboxes
    
    det_ind = to_del[:,2].to(int)
    # confs, areas = [], []
    # for i in det_ind:
    #     xmin,ymin,xmax,ymax,conf = bboxes[i,:-1]
    #     confs.append(conf.item())
    #     areas.append((xmax-xmin)*(ymax-ymin))
        
    # confs = 1 - normalize(torch.tensor(confs).unsqueeze(-1))
    # areas = 1 - normalize(torch.tensor(areas).unsqueeze(-1))
    ious = normalize(ious.unsqueeze(-1))
    mean = ious

    to_del = torch.hstack((to_del, mean))
    to_del = to_del[to_del[:, -1].sort(descending=True)[1]] # (N, 4)
    #print(to_del)
    
    to_del_ids = []
    for i in range(to_del.shape[0]):
        if to_del[i, 0].item() in to_del_ids:
            #print(f'Cause already deleted {to_del[i, 0]}')
            continue
        if to_del[i, 2].item() in to_del_ids:
            #print(f'Detection already deleted {to_del[i, 2]}')
            continue
        to_del_ids.append(int(to_del[i, 2].item()))
        #print(f'Delete {to_del[i, 2]} because of {to_del[i, 0]}, iou: {to_del[i, 3]}')
    bboxes_filtered = bboxes[[x for x in range(bboxes.shape[0]) if x not in to_del_ids],:]
    windows_filtered = windows[[x for x in range(windows.shape[0]) if x not in to_del_ids],:]
    # print(bboxes_filtered)
    return windows_filtered, bboxes_filtered



def filter_dets_single_matrix_area(windows, bboxes, th=0.7):
    
    def normalize(input_tensor):
        input_tensor -= input_tensor.min(0, keepdim=True)[0]
        input_tensor /= input_tensor.max(0, keepdim=True)[0]
        return input_tensor
    
    unique_windows = torch.unique(windows, dim=0)
    
    intersection_maxs = torch.min(bboxes[:, None, 2:4], unique_windows[:, 2:4]) # xmax, ymax
    intersection_mins = torch.max(bboxes[:, None, :2], unique_windows[:, :2]) # xmin, ymin
    intersections = torch.cat((intersection_mins, intersection_maxs), dim=2)
    intersections[(intersections[:,:,2] - intersections[:,:,0] < 0) | (intersections[:,:,3] - intersections[:,:,1] < 0)] = 0 # no common area
    
    # detections from window = 0
    for i, unique_window in enumerate(unique_windows):
        win_ind = torch.unique(torch.where(windows==unique_window)[0])
        intersections[win_ind, i, :] = 0
    # print(intersections.shape, bboxes.shape)
    
    ious = torch.empty((len(bboxes), len(unique_windows), len(bboxes)))
    for i in range(intersections.shape[1]):
        ious[:,i,...] = box_iou(intersections[:,i,:], bboxes[:,:4])
        
    for i in range(bboxes.shape[0]): # a detection cannot be removed because of itself
        ious[i,:,i] = 0
    
    #print(ious)
    to_del = torch.nonzero(ious > th)
    ious = ious[ious > th]
    if not ious.numel():
        return windows, bboxes
    
    det_ind = to_del[:,2].to(int)
    confs, areas = [], []
    for i in det_ind:
        xmin,ymin,xmax,ymax,conf = bboxes[i,:-1]
        confs.append(conf.item())
        areas.append((xmax-xmin)*(ymax-ymin))
        
    # confs = 1 - normalize(torch.tensor(confs).unsqueeze(-1))
    areas = 1 - normalize(torch.tensor(areas).unsqueeze(-1))
    # ious = normalize(ious.unsqueeze(-1))
    mean = areas #torch.mean(torch.stack([ious,confs, areas]), 0)

    to_del = torch.hstack((to_del, mean))
    to_del = to_del[to_del[:, -1].sort(descending=True)[1]] # (N, 4)
    #print(to_del)
    
    to_del_ids = []
    for i in range(to_del.shape[0]):
        if to_del[i, 0].item() in to_del_ids:
            #print(f'Cause already deleted {to_del[i, 0]}')
            continue
        if to_del[i, 2].item() in to_del_ids:
            #print(f'Detection already deleted {to_del[i, 2]}')
            continue
        to_del_ids.append(int(to_del[i, 2].item()))
        #print(f'Delete {to_del[i, 2]} because of {to_del[i, 0]}, iou: {to_del[i, 3]}')
    bboxes_filtered = bboxes[[x for x in range(bboxes.shape[0]) if x not in to_del_ids],:]
    windows_filtered = windows[[x for x in range(windows.shape[0]) if x not in to_del_ids],:]
    # print(bboxes_filtered)
    return windows_filtered, bboxes_filtered



def filter_dets_single_matrix_conf(windows, bboxes, th=0.7):
    
    def normalize(input_tensor):
        input_tensor -= input_tensor.min(0, keepdim=True)[0]
        input_tensor /= input_tensor.max(0, keepdim=True)[0]
        return input_tensor
    
    unique_windows = torch.unique(windows, dim=0)
    
    intersection_maxs = torch.min(bboxes[:, None, 2:4], unique_windows[:, 2:4]) # xmax, ymax
    intersection_mins = torch.max(bboxes[:, None, :2], unique_windows[:, :2]) # xmin, ymin
    intersections = torch.cat((intersection_mins, intersection_maxs), dim=2)
    intersections[(intersections[:,:,2] - intersections[:,:,0] < 0) | (intersections[:,:,3] - intersections[:,:,1] < 0)] = 0 # no common area
    
    # detections from window = 0
    for i, unique_window in enumerate(unique_windows):
        win_ind = torch.unique(torch.where(windows==unique_window)[0])
        intersections[win_ind, i, :] = 0
    # print(intersections.shape, bboxes.shape)
    
    ious = torch.empty((len(bboxes), len(unique_windows), len(bboxes)))
    for i in range(intersections.shape[1]):
        ious[:,i,...] = box_iou(intersections[:,i,:], bboxes[:,:4])
        
    for i in range(bboxes.shape[0]): # a detection cannot be removed because of itself
        ious[i,:,i] = 0
    
    #print(ious)
    to_del = torch.nonzero(ious > th)
    ious = ious[ious > th]
    if not ious.numel():
        return windows, bboxes
    
    det_ind = to_del[:,2].to(int)
    confs, areas = [], []
    for i in det_ind:
        xmin,ymin,xmax,ymax,conf = bboxes[i,:-1]
        confs.append(conf.item())
        areas.append((xmax-xmin)*(ymax-ymin))
        
    confs = 1 - normalize(torch.tensor(confs).unsqueeze(-1))
    # areas = 1 - normalize(torch.tensor(areas).unsqueeze(-1))
    # ious = normalize(ious.unsqueeze(-1))
    mean = confs #torch.mean(torch.stack([ious,confs, areas]), 0)

    to_del = torch.hstack((to_del, mean))
    to_del = to_del[to_del[:, -1].sort(descending=True)[1]] # (N, 4)
    #print(to_del)
    
    to_del_ids = []
    for i in range(to_del.shape[0]):
        if to_del[i, 0].item() in to_del_ids:
            #print(f'Cause already deleted {to_del[i, 0]}')
            continue
        if to_del[i, 2].item() in to_del_ids:
            #print(f'Detection already deleted {to_del[i, 2]}')
            continue
        to_del_ids.append(int(to_del[i, 2].item()))
        #print(f'Delete {to_del[i, 2]} because of {to_del[i, 0]}, iou: {to_del[i, 3]}')
    bboxes_filtered = bboxes[[x for x in range(bboxes.shape[0]) if x not in to_del_ids],:]
    windows_filtered = windows[[x for x in range(windows.shape[0]) if x not in to_del_ids],:]
    # print(bboxes_filtered)
    return windows_filtered, bboxes_filtered



OBS_SORT_TYPES = {
    'iou': filter_dets_single_matrix_iou, 
    'conf': filter_dets_single_matrix_conf, 
    'area': filter_dets_single_matrix_area, 
    'all': filter_dets_single_matrix_all, 
    'none': filter_dets
}