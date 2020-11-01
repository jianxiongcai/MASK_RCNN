## Author: Lishuo Pan 2020/4/18

import torch
import torchvision.transforms.functional
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib

matplotlib.use('Agg')               # No display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.mlab as mlab
import os.path
from dataset import BuildDataset,BuildDataLoader
from tqdm import tqdm



## Visualize debugging
if __name__ == '__main__':
    # file path and make a list
    imgs_path = 'data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = 'data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = 'data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = 'data/hw3_mycocodata_bboxes_comp_zlib.npy'

    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths, augmentation=False)

    ## Visualize debugging
    # --------------------------------------------
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    dataset_scale_list = []
    dataset_ratio_list = []
    for idx, batch in enumerate(tqdm(train_loader), 0):
        boxes = batch['bbox']
                  
        batch_size=len(boxes)
        for i in range(batch_size):
            for obj_i in range(boxes[i].shape[0]):
                w=boxes[i][obj_i, 2].clone().detach().float()
                h=boxes[i][obj_i, 3].clone().detach().float()
                ratio=w/h
                scale=torch.sqrt(w*h)
                dataset_ratio_list.append(ratio.item())
                dataset_scale_list.append(scale.item())

    # todo (jianxiong): why not use np median
    # def median(lst):
    #     n = len(lst)
    #     s = sorted(lst)
    #     return (sum(s[n//2-1:n//2+1])/2.0, s[n//2])[n % 2] if n else None
    num_bins = 50

    dataset_scales = np.array(dataset_scale_list)
    dataset_ratios = np.array(dataset_ratio_list)

    plt.figure()
    n, bins, patch = plt.hist(dataset_scale_list, bins=num_bins, facecolor='blue', edgecolor='black')
    plt.title("Scale histogram with median = {}".format(np.median(dataset_scales)))
    plt.xlabel('Scale')
    plt.ylabel('Occurrences')
    plt.savefig("scale.png")
    
    plt.figure()
    n, bins, patch = plt.hist(dataset_ratio_list, bins=num_bins, facecolor='blue', edgecolor='black')
    plt.title("Aspect Ratio histogram with median = {}".format(np.median(dataset_ratios)))
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Occurrences')
    plt.savefig("ratio.png")
