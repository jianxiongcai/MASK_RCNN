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

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path, augmentation):
        """

        :param path:
        :param augmentation: If Ture, horizontal flipping with 0.5 prob
        """
        #############################################
        # Initialize  Dataset
        #############################################
        self.augmentation = augmentation
        # all files
        imgs_path, masks_path, labels_path, bboxes_path = path

        # load dataset
        self.images_h5 = h5py.File(imgs_path, 'r')
        self.masks_h5 = h5py.File(masks_path, 'r')
        self.labels_all = np.load(labels_path, allow_pickle=True)
        self.bboxes_all = np.load(bboxes_path, allow_pickle=True)

        # As the mask are saved sequentially, compute the mask start index for each images
        n_objects_img = [len(self.labels_all[i]) for i in range(len(self.labels_all))]  # Number of objects per list
        self.mask_offset = np.cumsum(n_objects_img)  # the start index for each images
        # Add a 0 to the head. offset[0] = 0
        self.mask_offset = np.concatenate([np.array([0]), self.mask_offset])

    # In this function for given index we rescale the image and the corresponding  masks, boxes
    # and we return them as output
    # output:
    # transed_img: (3, 800, 1088)  (3,h,w)
    # label: (n_obj, )
    # transed_mask: (n_box, 800, 1088)
    # transed_bbox: (n_box, 4), unnormalized bounding box coordinates (resized)
    # index: if data augmentation is performed, the index will be a virtual unique identify (i.e. += len(dataset))
    def __getitem__(self, index):
        ################################
        # return transformed images,labels,masks,boxes,index
        ################################
        # images
        img_np = self.images_h5['data'][index] / 255.0  # (3, 300, 400)
        img = torch.tensor(img_np, dtype=torch.float)

        # annotation
        # label: start counting from 1
        label = torch.tensor(self.labels_all[index], dtype=torch.long)
        # collect all object mask for the image
        mask_offset_s = self.mask_offset[index]
        mask_list = []
        for i in range(len(label)):
            # get the mask of the ith object in the image
            mask_np = self.masks_h5['data'][mask_offset_s + i] * 1.0
            mask_tmp = torch.tensor(mask_np, dtype=torch.float)
            mask_list.append(mask_tmp)
        # (n_obj, 300, 400)
        mask = torch.stack(mask_list)

        # prepare data and do augumentation (if set)
        bbox_np = self.bboxes_all[index]
        bbox = torch.tensor(bbox_np, dtype=torch.float)
        transed_img, transed_mask, transed_bbox, index = self.pre_process_batch(img, mask, bbox, index)

        assert transed_img.shape == (3, 800, 1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]

        # for synthesized image, index is assigned a new one (unique id)
        index += self.images_h5['data'].shape[0]

        return transed_img, label, transed_mask, transed_bbox, index

    def __len__(self):
        # return len(self.imgs_data)
        return self.images_h5['data'].shape[0]

    # This function preprocess the given image, mask, box by rescaling them appropriately
    # output:
    #        img: (3,800,1088)
    #        mask: (n_box,800,1088)
    #        box: (n_box,4)
    def pre_process_batch(self, img, mask, bbox, index):
        #######################################
        # apply the correct transformation to the images,masks,boxes
        ######################################
        assert isinstance(img, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert isinstance(bbox, torch.Tensor)
        # image
        img = self.torch_interpolate(img, 800, 1066)  # (3, 800, 1066)
        img = torchvision.transforms.functional.normalize(img, mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
        img = F.pad(img, [11, 11])

        # mask: (N_obj * 300 * 400)
        mask = self.torch_interpolate(mask, 800, 1066)  # (N_obj, 800, 1066)
        mask = F.pad(mask, [11, 11])  # (N_obj, 800, 1088)

        # transfer bounding box
        # 1) from (x1, y1, x2, y2) => (c_x, c_y, w, h)
        centralized = torch.zeros_like(bbox)
        centralized[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2.0
        centralized[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2.0
        centralized[:, 2] = bbox[:, 2] - bbox[:, 0]
        centralized[:, 3] = bbox[:, 3] - bbox[:, 1]

        # 2) to the resized image
        trans_bbox = torch.zeros_like(centralized)
        trans_bbox[:, 0] = centralized[:, 0] / 400.0 * 1066.0 + 11  # c_x
        trans_bbox[:, 1] = centralized[:, 1] / 300.0 * 800.0  # c_y
        trans_bbox[:, 2] = centralized[:, 2] / 400.0 * 1066.0  # w
        trans_bbox[:, 3] = centralized[:, 3] / 300.0 * 800.0  # h

        # do augmentation (if set)
        if self.augmentation and (np.random.rand(1).item() > 0.5):
            # perform horizontally flipping (data augmentation)
            assert img.dim() == 3
            assert mask.dim() == 3
            ret_img = torch.flip(img, dims=[2])
            ret_mask = torch.flip(mask, dims=[2])
            # bbox transform for augmentation (flip horizontal axis)
            ret_bbox = trans_bbox.clone()
            ret_bbox[:, 0] = 1088.0 - ret_bbox[:, 0]
            # for aug, create a unique index for unique identification
            index = index + self.images_h5['data'].shape[0]
        else:
            ret_img = img
            ret_mask = mask
            ret_bbox = trans_bbox

        assert ret_img.squeeze(0).shape == (3, 800, 1088)
        # assert bbox.shape[0] == mask.squeeze(0).shape[0]
        assert ret_bbox.shape[0] == ret_mask.shape[0]

        return ret_img, ret_mask, ret_bbox, index

    @staticmethod
    def torch_interpolate(x, H, W):
        """
        A quick wrapper fucntion for torch interpolate
        :return:
        """
        assert isinstance(x, torch.Tensor)
        C = x.shape[0]
        # require input: mini-batch x channels x [optional depth] x [optional height] x width
        x_interm = torch.unsqueeze(x, 0)
        x_interm = torch.unsqueeze(x_interm, 0)

        tensor_out = F.interpolate(x_interm, (C, H, W))
        tensor_out = tensor_out.squeeze(0)
        tensor_out = tensor_out.squeeze(0)
        return tensor_out

    @staticmethod
    def unnormalize_img(img):
        """
        Unnormalize image to [0, 1]
        :param img:
        :return:
        """
        assert img.shape == (3, 800, 1088)
        img = torchvision.transforms.functional.normalize(img, mean=[0.0, 0.0, 0.0],
                                                          std=[1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225])
        img = torchvision.transforms.functional.normalize(img, mean=[-0.485, -0.456, -0.406],
                                                          std=[1.0, 1.0, 1.0])
        return img


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    # output:
    #  dict{images: (bz, 3, 800, 1088)
    #       labels: list:len(bz)
    #       masks: list:len(bz){(n_obj, 800,1088)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #       index: list:len(bz)
    def collect_fn(self, batch):
        out_batch = {}
        tmp_image_list = []
        label_list = []
        mask_list = []
        bbox_list = []
        indice = []
        for transed_img, label, transed_mask, transed_bbox, index in batch:
            tmp_image_list.append(transed_img)
            label_list.append(label)
            mask_list.append(transed_mask)
            bbox_list.append(transed_bbox)
            indice.append(index)

        out_batch['images'] =torch.stack(tmp_image_list, dim=0)
        out_batch['labels'] = label_list
        out_batch['masks'] = mask_list
        out_batch['bbox'] = bbox_list
        out_batch['index'] = indice

        return out_batch

    def loader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn)


## Visualize debugging
if __name__ == '__main__':
    # file path and make a list
#     todo (jianxiong): change this back before submitting
    imgs_path = '../data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = '../data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = '../data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = '../data/hw3_mycocodata_bboxes_comp_zlib.npy'

#    imgs_path = '/workspace/data/hw3_mycocodata_img_comp_zlib.h5'
#    masks_path = '/workspace/data/hw3_mycocodata_mask_comp_zlib.h5'
#    labels_path = '/workspace/data/hw3_mycocodata_labels_comp_zlib.npy'
#    bboxes_path = '/workspace/data/hw3_mycocodata_bboxes_comp_zlib.npy'
    
#    os.makedirs("testfig", exist_ok=True)

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_scale_list = []
    dataset_ratio_list = []
    for idx, batch in enumerate(train_loader, 0):
        boxes = batch['bbox']
                  
        batch_size=len(boxes)
        for i in range(batch_size):
            w=boxes[i][:,2].clone().detach().float()
            h=boxes[i][:,3].clone().detach().float()
            ratio=w/h
            scale=torch.sqrt(w*h)
            dataset_ratio_list+=ratio
            dataset_scale_list+=scale
            
    def median(lst):
        n = len(lst)
        s = sorted(lst)
        return (sum(s[n//2-1:n//2+1])/2.0, s[n//2])[n % 2] if n else None
    num_bins = 50
    
    fig1 = plt.figure()
    n, bins, patch = plt.hist(dataset_scale_list, num_bins, facecolor='blue', edgecolor='black')
    plt.title("Scale histogram with median = {}".format(median(dataset_scale_list)))
    plt.xlabel('Scale')
    plt.ylabel('Occurrences')
    plt.savefig("scale.png")
    
    fig2 = plt.figure()
    n, bins, patch = plt.hist(dataset_ratio_list, num_bins, facecolor='blue', edgecolor='black')
    plt.title("Aspect Ratio histogram with median = {}".format(median(dataset_ratio_list)))
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Occurrences')
    plt.savefig("ratio.png")
