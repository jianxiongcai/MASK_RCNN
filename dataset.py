import h5py
import torch
from torchvision import transforms
import torchvision.transforms.functional
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from rpn import RPNHead
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from rpn import *
import matplotlib.cm as cm
import matplotlib.patches as patches
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')


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
        self.imgs_path = imgs_path
        self.masks_path = masks_path

        # get dataset length
        with h5py.File(self.imgs_path, 'r') as images_h5:
            self.N_images = images_h5['data'].shape[0]

        # load dataset
        # all images and masks will be lazy read
        self.labels_all = np.load(labels_path, allow_pickle=True)
        self.bboxes_all = np.load(bboxes_path, allow_pickle=True)
#        self.images_h5 = h5py.File(imgs_path, 'r')
#        self.masks_h5 = h5py.File(masks_path, 'r')
#        self.labels_all = np.load(labels_path, allow_pickle=True)
#        self.bboxes_all = np.load(bboxes_path, allow_pickle=True)

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
        images_h5 = h5py.File(self.imgs_path, 'r')
        masks_h5 = h5py.File(self.masks_path, 'r')
        img_np = images_h5['data'][index] / 255.0  # (3, 300, 400)
        img = torch.tensor(img_np, dtype=torch.float)

        # annotation
        # label: start counting from 1
        label = torch.tensor(self.labels_all[index], dtype=torch.long)
        # collect all object mask for the image
        mask_offset_s = self.mask_offset[index]
        mask_list = []
        for i in range(len(label)):
            # get the mask of the ith object in the image
            mask_np = masks_h5['data'][mask_offset_s + i] * 1.0
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

        # close files
        images_h5.close()
        masks_h5.close()

        return transed_img, label, transed_mask, transed_bbox, index

    def __len__(self):
        return self.N_images

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
            index = index + self.N_images
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

    @staticmethod
    def unnormalize_bbox(bbox):
        """
        Unnormalize one bbox annotation. from 0-1 => 0 - 1088
        x_res = x * 1066 + 11
        y_res = x * 800
        :param bbox: the normalized bounding box (4,)
        :return: the absolute bounding box location (4,)
        """
        bbox_res = torch.tensor(bbox, dtype=torch.float).clone().detach()
        bbox_res[0] = bbox[0] * 1066 + 11
        bbox_res[1] = bbox[1] * 800
        bbox_res[2] = bbox[2] * 1066 + 11
        bbox_res[3] = bbox[3] * 800
        return bbox_res

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

def keep_top_K_batch(cls_out, top_K):
    """
    Keep the top K bounding box for each image, set others to 0
    :param cls_out: (bz, 1, H, W)
    :return:
        cls_out: (bz, 1, H, W). All non-keeping ones are set to 0
        top_K:
    Examples:
        Input: cls_out=[[0.9, 0.3], [0.6, 0.1]]
        Input: top_K=2
        output=[[0.9, 0.0], [0.6, 0.0]]

    """
    out_res = torch.zeros_like(cls_out)
    for i in range(cls_out.shape[0]):
        # get the 20th top value
        tmp = torch.flatten(cls_out[i])
        top_values, _ = torch.topk(tmp, top_K)
        # only keep classification score > last value
        last_value = top_values[-1]
        mask = cls_out[i] >= last_value
        out_res[i, mask] = cls_out[i, mask]
        if torch.count_nonzero(out_res[i]) != top_K:
            print("[WARN] top_K keeping {} values".format(torch.count_nonzero(out_res[i])))
    return out_res

def plot_mask_batch(rpn_net, cls_out_raw, reg_out, images, boxes, indice, result_dir, top_K, mode):
    """

    :param rpn_net:
    :param cls_out_raw: the classification output (/gt)
    :param reg_out: the regression output (/ground_coord)
    :param boxes: input bounding boxes
    :param images: input images
    :param indice: unique index from dataloader
    :param result_dir: the result directory to save figures
    :param top_K: if not None, only plot the top K bounding box
    :param mode:
        If "groundtruth", plot the ground-truth bbox and its anchors.
        If "preNMS", select and plot top-K the inferred bounding box
        If "postNMS", plot all inferred boundind box with prob > 0
    :return:
    """
    # do filtering if needed
    if top_K is None:
        cls_out = cls_out_raw
    else:
        cls_out = keep_top_K_batch(cls_out_raw, top_K=top_K)

    # Flatten the ground truth and the anchors
    flatten_coord, flatten_cls, flatten_anchors = output_flattening(reg_out, cls_out, rpn_net.get_anchors())

    # Decode the ground truth box to get the upper left and lower right corners of the ground truth boxes
    decoded_coord = output_decoding(flatten_coord, flatten_anchors)

    # Plot the image and the anchor boxes with the positive labels and their corresponding ground truth box
    # decision threshold depend on mode
    if mode == "groundtruth":
        find_cor_bz = (flatten_cls == 1).nonzero()
    elif mode == "preNMS":
        # only keep top X, so everything non zero is positive
        find_cor_bz = flatten_cls.nonzero()
    elif mode == "postNMS":
        # only keep top X, so everything non zero is positive
        find_cor_bz = flatten_cls.nonzero()
    else:
        raise RuntimeError("[ERROR] mode not recognizable: {}".format(mode))
    find_cor_bz = find_cor_bz.squeeze(dim=1)


    batch_size = len(boxes)
    total_number_of_anchors = cls_out.shape[2] * cls_out.shape[3]
    for i in range(batch_size):
        image = transforms.functional.normalize(images[i],
                                                [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
        fig, ax = plt.subplots(1, 1)
        image_vis = image.permute(1, 2, 0).cpu().detach().numpy()
        ax.imshow(image_vis)
        # mask = [torch.rand(num_cor_bz)>0.5 for x in range(2)]

        mask1 = (find_cor_bz > i * total_number_of_anchors).flatten()
        mask2 = (find_cor_bz < (i + 1) * total_number_of_anchors).flatten()
        mask = torch.logical_and(mask1, mask2)
        find_cor = find_cor_bz[mask]

        # # only keep top_K for at inference stage
        # if (top_K is not None) and (top_K < len(find_cor)):
        #     scores = flatten_cls[find_cor]
        #     _, keep_indice = torch.topk(scores, top_K)
        #     find_cor = find_cor[keep_indice]

        for elem in find_cor:
            coord = decoded_coord[elem, :].view(-1)
            anchor = flatten_anchors[elem, :].view(-1)
            coord = coord.cpu().detach().numpy()
            anchor = anchor.cpu().detach().numpy()
            if mode == "groundtruth":
                # plot bbox
                rect = patches.Rectangle((coord[0], coord[1]), coord[2] - coord[0], coord[3] - coord[1], fill=False,
                                         color='r')
                ax.add_patch(rect)
                # plot positive anchor
                rect = patches.Rectangle((anchor[0] - anchor[2] / 2, anchor[1] - anchor[3] / 2), anchor[2], anchor[3],
                                         fill=False, color='b')
                ax.add_patch(rect)
            else:
                # plot bbox
                rect = patches.Rectangle((coord[0], coord[1]), coord[2] - coord[0], coord[3] - coord[1], fill=False,
                                         color='b')
                ax.add_patch(rect)

        plt.savefig("{}/{}.png".format(result_dir, indice[i]))
        plt.show()
        plt.close('all')

def plot_visual_correctness_batch(img, label, boxes, mask, indexes, visual_dir, rgb_color_list):
    batch_size = len(indexes)
    for i in range(batch_size):
        ## TODO: plot images with annotations
        fig, ax = plt.subplots(1)
        # the input image: to (800, 1088, 3)
        alpha = 0.15
        # img_vis = alpha * BuildDataset.unnormalize_img(img[i])
        img_vis = img[i]
        img_vis = img_vis.permute((1, 2, 0)).cpu().numpy()

        # object mask: assign color with class label
        for obj_i, obj_mask in enumerate(mask[i], 0):
            obj_label = label[i][obj_i]

            rgb_color = rgb_color_list[obj_label - 1]
            # (800, 1088, 3)
            obj_mask_np = np.stack([obj_mask.cpu().numpy(), obj_mask.cpu().numpy(), obj_mask.cpu().numpy()], axis=2)
            # alpha-blend mask
            img_vis[obj_mask_np != 0] = ((1 - alpha) * rgb_color + alpha * img_vis)[obj_mask_np != 0]

        # overlapping objects
        img_vis = np.clip(img_vis, 0, 1)
        ax.imshow(img_vis)

        # bounding box
        for obj_i, obj_bbox in enumerate(boxes[i], 0):
            obj_w = obj_bbox[2]
            obj_h = obj_bbox[3]
            rect = patches.Rectangle((obj_bbox[0] - obj_bbox[2] / 2, obj_bbox[1] - obj_bbox[3] / 2), obj_w, obj_h, linewidth=1, edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)

        plt.savefig("{}/{}.png".format(visual_dir, indexes[i]))
        plt.show()
        plt.close('all')



if __name__ == '__main__':
    # file path and make a list
    imgs_path = 'data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = 'data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = 'data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = 'data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]

    visual_dir = "Visual_image"
    mask_dir = "Grndbox"
    os.makedirs(visual_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)



    # load the data into data.Dataset
    torch.random.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    dataset = BuildDataset(paths, augmentation=False)

    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    rpn_net = RPNHead(device=torch.device('cpu'))
#     push the randomized training data into the dataloader
#
#     train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
#     test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    mask_color_list = ["jet", "ocean", "Spectral", "spring", "cool"]
    # convert to rgb color list
    rgb_color_list = []
    for color_str in mask_color_list:
        color_map = cm.ScalarMappable(cmap=color_str)
        rgb_value = np.array(color_map.to_rgba(0))[:3]
        rgb_color_list.append(rgb_value)


    for idx, batch in enumerate(tqdm(train_loader), 0):
        images = batch['images'][:, :, :, :]
        indexes = batch['index']
        boxes = batch['bbox']
        mask = batch['masks']
        labels = batch['labels']
        gt, ground_coord = rpn_net.create_batch_truth(boxes, indexes, images.shape[-2:])
        plot_mask_batch(rpn_net, gt, ground_coord, images, boxes, indexes, mask_dir, top_K=None, mode="groundtruth")
        plot_visual_correctness_batch(images, labels, boxes, mask, indexes, visual_dir, rgb_color_list)

