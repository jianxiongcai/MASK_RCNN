import torch
from torchvision import transforms
import torchvision.transforms.functional
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from rpn import *
import matplotlib.patches as patches


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path, augmentation=True):
        """

        :param path: the path to the dataset root: /workspace/data or XXX/data/SOLO
        :param argumentation: if True, perform horizontal flipping argumentation
        """
        self.augmentation = augmentation
        # all files
        imgs_path, masks_path, labels_path, bboxes_path = path

        # load dataset
        # all images and masks will be lazy read
        self.images_h5 = h5py.File(imgs_path, 'r')
        self.masks_h5 = h5py.File(masks_path, 'r')
        self.labels_all = np.load(labels_path, allow_pickle=True)
        self.bboxes_all = np.load(bboxes_path, allow_pickle=True)

        # As the mask are saved sequentially, compute the mask start index for each images
        n_objects_img = [len(self.labels_all[i]) for i in range(len(self.labels_all))]  # Number of objects per list
        self.mask_offset = np.cumsum(n_objects_img)  # the start index for each images
        # Add a 0 to the head. offset[0] = 0
        self.mask_offset = np.concatenate([np.array([0]), self.mask_offset])

    # output:
    # transed_img
    # label
    # transed_mask
    # transed_bbox
    # Note: For data augmentation, number of items is 2 * N_images
    def __getitem__(self, index):
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

        # normalize bounding box
        bbox_np = self.bboxes_all[index]
        bbox = torch.tensor(bbox_np, dtype=torch.float)
        transed_img, transed_mask, transed_bbox = self.pre_process_batch(img, mask, bbox)
        if self.augmentation and (np.random.rand(1).item() > 0.5):
            # perform horizontally flipping (data augmentation)
            assert transed_img.ndim == 3
            assert transed_mask.ndim == 3
            transed_img = torch.flip(transed_img, dims=[2])
            transed_mask = torch.flip(transed_mask, dims=[2])
            # bbox transform
            transed_bbox_new = transed_bbox.clone()
            transed_bbox_new[:, 0] = 1 - transed_bbox[:, 2]
            transed_bbox_new[:, 2] = 1 - transed_bbox[:, 0]
            transed_bbox = transed_bbox_new

            assert torch.all(transed_bbox[:, 0] < transed_bbox[:, 2])

        # check flag
        assert transed_img.shape == (3, 800, 1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]
        return transed_img, label, transed_mask, transed_bbox

    def __len__(self):
        # return len(self.imgs_data)
        return self.images_h5['data'].shape[0]

    # This function take care of the pre-process of img,mask,bbox
    # in the input mini-batch
    # input:
    # img: 3*300*400
    # mask: 3*300*400
    # bbox: n_box*4
    def pre_process_batch(self, img, mask, bbox):
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

        # normalize bounding box
        # (x1, y1, x2, y2)
        bbox_normed = torch.zeros_like(bbox)
        for i in range(bbox.shape[0]):
            bbox_normed[i, 0] = bbox[i, 0] / 400.0
            bbox_normed[i, 1] = bbox[i, 1] / 300.0
            bbox_normed[i, 2] = bbox[i, 2] / 400.0
            bbox_normed[i, 3] = bbox[i, 3] / 300.0
        assert torch.max(bbox_normed) <= 1.0
        assert torch.min(bbox_normed) >= 0.0
        bbox = bbox_normed

        # check flag
        assert img.shape == (3, 800, 1088)
        # todo (jianxiong): following commmented was provided in code release
        # assert bbox.shape[0] == mask.squeeze(0).shape[0]
        assert bbox.shape[0] == mask.shape[0]
        return img, mask, bbox

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
    def unnormalize_bbox(bbox):
        """
        Unnormalize one bbox annotation. from 0-1 => 0 - 1088
        x_res = x * 1066 + 11
        y_res = x * 800
        :param bbox: the normalized bounding box (4,)
        :return: the absolute bounding box location (4,)
        """
        bbox_res = torch.tensor(bbox, dtype=torch.float)
        bbox_res[0] = bbox[0] * 1066 + 11
        bbox_res[1] = bbox[1] * 800
        bbox_res[2] = bbox[2] * 1066 + 11
        bbox_res[3] = bbox[3] * 800
        return bbox_res

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

    # img: (bz, 3, 800, 1088)
    # label_list: list, len:bz, each (n_obj,)
    # transed_mask_list: list, len:bz, each (n_obj, 800,1088)
    # transed_bbox_list: list, len:bz, each (n_obj, 4)
    def collect_fn(self, batch):
        transed_img_list = []
        label_list = []
        transed_mask_list = []
        transed_bbox_list = []
        for transed_img, label, transed_mask, transed_bbox in batch:
            transed_img_list.append(transed_img)
            label_list.append(label)
            transed_mask_list.append(transed_mask)
            transed_bbox_list.append(transed_bbox)
        return torch.stack(transed_img_list, dim=0), label_list, transed_mask_list, transed_bbox_list

    def loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers,
                          collate_fn=self.collect_fn)


if __name__ == '__main__':
    # file path and make a list
    imgs_path = 'data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = 'data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = 'data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = 'data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths, augmentation=False)

    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    rpn_net = RPNHead()
    # push the randomized training data into the dataloader

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 1
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    for i, batch in enumerate(train_loader, 0):
        images = batch['images'][0, :, :, :]
        indexes = batch['index']
        boxes = batch['bbox']
        gt, ground_coord = rpn_net.create_batch_truth(boxes, indexes, images.shape[-2:])

        # Flatten the ground truth and the anchors
        flatten_coord, flatten_gt, flatten_anchors = output_flattening(ground_coord, gt, rpn_net.get_anchors())

        # Decode the ground truth box to get the upper left and lower right corners of the ground truth boxes
        decoded_coord = output_decoding(flatten_coord, flatten_anchors)

        # Plot the image and the anchor boxes with the positive labels and their corresponding ground truth box
        images = transforms.functional.normalize(images,
                                                 [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                 [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(images.permute(1, 2, 0))

        find_cor = (flatten_gt == 1).nonzero()
        find_neg = (flatten_gt == -1).nonzero()

        for elem in find_cor:
            coord = decoded_coord[elem, :].view(-1)
            anchor = flatten_anchors[elem, :].view(-1)

            col = 'r'
            rect = patches.Rectangle((coord[0], coord[1]), coord[2] - coord[0], coord[3] - coord[1], fill=False,
                                     color=col)
            ax.add_patch(rect)
            rect = patches.Rectangle((anchor[0] - anchor[2] / 2, anchor[1] - anchor[3] / 2), anchor[2], anchor[3],
                                     fill=False, color='b')
            ax.add_patch(rect)

        plt.show()

        if (i > 20):
            break