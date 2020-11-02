import matplotlib
matplotlib.use('Agg')               # No display
import matplotlib.pyplot as plt

from dataset import BuildDataset, BuildDataLoader
from dataset import plot_mask_batch
from rpn import RPNHead
from accuracy_tracker import AccuracyTracker

import os.path
import torch.backends.cudnn
import torch.utils.data
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm

def do_eval(dataloader, checkpoint_file, device, result_dir=None):
    if result_dir is not None:
        os.makedirs(result_dir, exist_ok=True)
    # ============================ Eval ================================
    rpn_head = RPNHead(device=device).to(device)
    checkpoint = torch.load(checkpoint_file)
    print("[INFO] Weight loaded from checkpoint file: {}".format(checkpoint_file))
    rpn_head.load_state_dict(checkpoint['model_state_dict'])
    rpn_head.eval()  # set to eval mode
    tracker = AccuracyTracker()

    for iter, data in enumerate(tqdm(dataloader), 0):
        img = data['images'].to(device)
        # label_list = [x.to(device) for x in data['labels']]
        # mask_list = [x.to(device) for x in data['masks']]
        bbox_list = [x.to(device) for x in data['bbox']]
        index_list = data['index']
        img_shape = (img.shape[2], img.shape[3])
        with torch.no_grad():
            cls_out, reg_out = rpn_head(img)
            targ_cls, targ_reg = rpn_head.create_batch_truth(bbox_list, index_list, img_shape)
            tracker.onNewBatch(cls_out, targ_cls)
            # visualization
            if result_dir is not None:
                plot_mask_batch(rpn_head, cls_out, reg_out, img, bbox_list, index_list, result_dir, top_K=20)


    print("tracker.TP_pos: {}".format(tracker.TP_pos))
    print("tracker.TP_neg: {}".format(tracker.TP_neg))
    print("tracker.tot_pos: {}".format(tracker.tot_pos))
    print("tracker.tot_neg: {}".format(tracker.tot_neg))
    return tracker.getMetric()

#reproductivity
torch.random.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

# =========================== Config ==========================
batch_size = 8
checkpoint_file = "checkpoints/epoch_{}".format(69)
assert os.path.isfile(checkpoint_file)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# =========================== Dataset ==============================
# file path and make a list
imgs_path = 'data/hw3_mycocodata_img_comp_zlib.h5'
masks_path = 'data/hw3_mycocodata_mask_comp_zlib.h5'
labels_path = "data/hw3_mycocodata_labels_comp_zlib.npy"
bboxes_path = "data/hw3_mycocodata_bboxes_comp_zlib.npy"
paths = [imgs_path, masks_path, labels_path, bboxes_path]

dataset = BuildDataset(paths,augmentation=False)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# dataset
# train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
# train_loader = train_build_loader.loader()
test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = test_build_loader.loader()


# print("Train Point-wise Accuracy: {}".format(do_eval(train_loader, checkpoint_file, device, None)))
print("Test Point-wise Accuracy: {}".format(do_eval(test_loader, checkpoint_file, device, "test_results")))
