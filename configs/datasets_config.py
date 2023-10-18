ZeF20 = dict(
    cat2label = {'fish': 0},
    train_list = "/home/ubuntu/ZeF20/data/3DZeF20/train-gpus.txt",
    val_list = "/home/ubuntu/ZeF20/data/3DZeF20/val-gpus.txt",
    test_list = "/home/ubuntu/ZeF20/data/3DZeF20/test-gpus.txt",
    seq_pos = -3,
    img_ext = "jpg"
)


VisDroneMOT = dict(
    cat2label = {'ignored-regions': 0, 'pedestrian': 1, 'people': 2, 'bicycle': 3, 'car': 4, 'van': 5, 
                 'truck': 6, 'tricycle': 7, 'awning-tricycle': 8, 'bus': 9, 'motor': 10, 'others': 11},
    train_list = "/home/ubuntu/VisDrone2019/tinyROI/VD23-train.txt",
    val_list = "/home/ubuntu/VisDrone2019/tinyROI/VD23-val.txt",
    test_list = "/home/ubuntu/VisDrone2019/tinyROI/VD23-test-dev.txt",
    seq_pos = -2,
    img_ext = "jpg"
)

DATASETS = {
    "ZeF20": ZeF20,
    "VisDroneMOT": VisDroneMOT
}

