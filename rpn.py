import torch
from torch.nn import functional as F
from torchvision import transforms
from torch import nn, Tensor
from dataset import *
from utils import *
import matplotlib.patches as patches
import os.path
import shutil
import torchvision
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import gc
import numpy as np
gc.enable()


class RPNHead(torch.nn.Module):

    def __init__(self,  device=('cuda' if torch.cuda.is_available() else 'cpu'), anchors_param=dict(ratio=1.01,scale= 403, grid_size=(50, 68), stride=16)):
        # Initialize the backbone, intermediate layer clasifier and regressor heads of the RPN
        super(RPNHead,self).__init__()

        self.device=device
        

        # Define Backbone
        self.backbone = nn.Sequential(nn.Conv2d(3, 16, 5, padding=2),           # Block 1
                                      nn.BatchNorm2d(16),
                                      nn.ReLU(inplace=False),
                                      nn.MaxPool2d(2, stride=2, padding=0),
                                      nn.Conv2d(16, 32, 5, padding=2),           # Block 2
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=False),
                                      nn.MaxPool2d(2, stride=2, padding=0),
                                      nn.Conv2d(32, 64, 5, padding=2),           # Block 3
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=False),
                                      nn.MaxPool2d(2, stride=2, padding=0),
                                      nn.Conv2d(64, 128, 5, padding=2),           # Block 4
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=False),
                                      nn.MaxPool2d(2, stride=2, padding=0),
                                      nn.Conv2d(128, 256, 5, padding=2),           # Block 5
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=False))

        # Define Intermediate Layer
        self.interm_layer = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(inplace=False))

        # Define Proposal Classifier Head
        self.cls_head = nn.Sequential(nn.Conv2d(256, 1, 1, padding=0),
                                      nn.Sigmoid())

        # Define Proposal Regressor Head
        # todo (jianxiong): the handout did not specify the Sigmoid, but it makes sense to add that.
        # todo: note: double-check the output and target labels is normalized to [0, 1], (sigmoid has the output of [0, 1])
        self.reg_head = nn.Sequential(nn.Conv2d(256, 4, 1, padding=0),
                                      nn.Sigmoid())

        #  find anchors
        self.anchors_param=anchors_param
        self.anchors=self.create_anchors(self.anchors_param['ratio'],self.anchors_param['scale'],self.anchors_param['grid_size'],self.anchors_param['stride'])
        self.ground_dict={}
        self.anchor_inbound=self.create_anchor_inbound(self.anchors_param['grid_size'], image_shape=(800,1088))





    # Forward  the input through the backbone the intermediate layer and the RPN heads
    # Input:
    #       X: (bz,3,image_size[0],image_size[1])}
    # Ouput:
    #       logits: (bz,1,grid_size[0],grid_size[1])}
    #       bbox_regs: (bz,4, grid_size[0],grid_size[1])}
    # Note: bbox_regs is not raw bounding box coordinates
    
    def forward(self, X):

        # forward through the Backbone
        X = self.backbone(X)

        # forward through the Intermediate layer
        X = self.interm_layer(X)

        # forward through the Classifier Head
        logits = self.cls_head(X)

        # forward through the Regressor Head
        bbox_regs =  self.reg_head(X)

        assert logits.shape[1:4]==(1,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
        assert bbox_regs.shape[1:4]==(4,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return logits, bbox_regs




    # Forward input batch through the backbone
    # Input:
    #       X: (bz,3,image_size[0],image_size[1])}
    # Ouput:
    #       X: (bz,256,grid_size[0],grid_size[1])
    # todo (jianxiong): does not seen useful...
    # def forward_backbone(self,X):
    #     #####################################
    #     # TODO forward through the backbone
    #     #####################################
    #     assert X.shape[1:4]==(256,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
    #
    #     return X




    def create_anchor_inbound(self, grid_size, image_shape):
        anchors=self.anchors
        w=anchors[0,0,2]
        h=anchors[0,0,3]
        anchor_inbound_list=[]
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if anchors[i,j,0]<w*0.5 or anchors[i,j,1]<h*0.5 or anchors[i,j,0]>image_shape[1]-w*0.5 or anchors[i,j,1]>image_shape[0]-h*0.5:
                    continue
                anchor_inbound_list.append(anchors[i,j].float())
        anchor_inbound=torch.stack(anchor_inbound_list) #(1800,4) i,j,w,h
        return anchor_inbound
    
    # This function creates the anchor boxes
    # Output:
    #       anchors: (grid_size[0],grid_size[1],4)
    
    def create_anchors(self, aspect_ratio, scale, grid_sizes, stride):
        ######################################
        # TODO create anchors
        ######################################
        h_2=torch.tensor(scale**2/aspect_ratio, dtype=torch.float, device=self.device)
        h=torch.round(torch.sqrt(h_2)).long()
        w=torch.round(aspect_ratio*h.float()).long()
        anchors=torch.zeros(grid_sizes[0],grid_sizes[1],4, device=self.device, dtype=torch.long)
        x=torch.arange(grid_sizes[0])
        y=torch.arange(grid_sizes[1])
        xx, yy=torch.meshgrid(x,y)
        xx, yy = xx.to(self.device), yy.to(self.device)
        anchors[xx,yy,0]=((yy+0.5)*stride).long()
        anchors[xx,yy,1]=((xx+0.5)*stride).long()
        anchors[xx,yy,2]=w
        anchors[xx,yy,3]=h

        assert anchors.shape == (grid_sizes[0] , grid_sizes[1],4)

        return anchors



    def get_anchors(self):
        return self.anchors



    # This function creates the ground truth for a batch of images by using
    # create_ground_truth internally
    # Input:
    #      bboxes_list: list:len(bz){(n_obj,4)}
    #      indexes:      list:len(bz)
    #      image_shape:  tuple:len(2)
    # Output:
    #      ground_class: (bz,1,grid_size[0],grid_size[1])
    #      ground_coord: (bz,4,grid_size[0],grid_size[1])
    def create_batch_truth(self,bboxes_list,indexes,image_shape):
        #####################################
        # TODO create ground truth for a batch of images
        #####################################
        bz=len(bboxes_list)
        grid_size=(self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
        ground_class=torch.zeros(bz,1,grid_size[0],grid_size[1], device=self.device)
        ground_coord=torch.zeros(bz,4,grid_size[0],grid_size[1], device=self.device)
        for idx in range(bz):
            ground_class[idx,:,:,:],ground_coord[idx,:,:,:]=self.create_ground_truth(bboxes_list[idx], indexes[idx], grid_size, self.anchors, image_shape)

        assert ground_class.shape[1:4]==(1,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
        assert ground_coord.shape[1:4]==(4,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return ground_class, ground_coord




    # This function creates the ground truth for one image
    # It also caches the ground truth for the image using its index
    # Input:
    #       bboxes:      (n_boxes,4)
    #       index:       scalar (the index of the image in the total dataset used for caching)
    #       grid_size:   tuple:len(2)
    #       anchors:     (grid_size[0],grid_size[1],4)
    # Output:
    #       ground_clas:  (1,grid_size[0],grid_size[1])
    #       ground_coord: (4,grid_size[0],grid_size[1])
    def create_ground_truth(self, bboxes, index, grid_size, anchors, image_size):
      key = str(index)
      if key in self.ground_dict:
            groundt, ground_coord = self.ground_dict[key]
            return groundt, ground_coord

        #####################################################
        # TODO create ground truth for a single image
        #####################################################
      stride=self.anchors_param['stride']
      labels=-torch.ones(grid_size[0],grid_size[1], device=self.device).long()
      num_anchor_inbound=self.anchor_inbound.shape[0]
      anchor_inbound=self.anchor_inbound
      iou_inbound_anchor_list=[]
      positive_inbound_anchor_list=[]
      negative_inbound_anchor_list=[]

      for obj_idx in range(bboxes.shape[0]):
        bbox_single=bboxes[obj_idx].view(1,-1)
        bbox_n=bbox_single.repeat(num_anchor_inbound,1)
        iou=IOU(bbox_n,anchor_inbound)        
        iou_inbound_anchor_list.append(iou)
        
        iou_low_mask=(iou<0.3)
        negative_inbound_anchor_list.append(iou_low_mask)
        
        iou_high_mask=(iou>0.7)
        max_iou=torch.max(iou)
        max_iou_idx=(iou>0.99*max_iou).nonzero()
#        max_iou_idx=torch.argmax(iou)
        iou_high_mask[max_iou_idx]=1
        positive_inbound_anchor_list.append(iou_high_mask)
        
      iou_inbound_anchor=torch.stack(iou_inbound_anchor_list)
      positive_mask = torch.sum(torch.stack(positive_inbound_anchor_list), dim=0) != 0
      temp=torch.squeeze(positive_mask.nonzero(), dim=1)
      positive_idx=(anchor_inbound[temp,0:2].float()/stride-0.5).long()
      positive_idx=torch.index_select(positive_idx, 1, torch.tensor([1,0], dtype=torch.long, device=self.device))

      negative_mask = torch.sum(torch.stack(negative_inbound_anchor_list), dim=0) != 0
      temp1=torch.squeeze(negative_mask.nonzero(), dim=1)
      negative_idx=(anchor_inbound[temp1,0:2].float()/stride-0.5).long()     
      negative_idx=torch.index_select(negative_idx, 1, torch.tensor([1,0], dtype=torch.long, device=self.device))

      highest_iou_mask,highest_iou_bbox_idx=torch.max(iou_inbound_anchor, 0)     
      labels[negative_idx[:,0],negative_idx[:,1]]=0
      labels[positive_idx[:,0],positive_idx[:,1]]=1
      ground_coord_orig=anchors.permute((2,0,1))

      highest_bbox_idx=highest_iou_bbox_idx[positive_mask]
      bbox_positive=bboxes[highest_bbox_idx]
      bbox_x=bbox_positive[:,0].float()
      bbox_y=bbox_positive[:,1].float()
      bbox_w=bbox_positive[:,2].float()
      bbox_h=bbox_positive[:,3].float()
      x_a=ground_coord_orig[:,positive_idx[:,0],positive_idx[:,1]][0,:].float()
      y_a=ground_coord_orig[:,positive_idx[:,0],positive_idx[:,1]][1,:].float()
      w_a=ground_coord_orig[:,positive_idx[:,0],positive_idx[:,1]][2,:].float()
      h_a=ground_coord_orig[:,positive_idx[:,0],positive_idx[:,1]][3,:].float()
      
      ground_coord=torch.zeros(4,grid_size[0],grid_size[1], device=self.device)
      
      ground_coord[0,positive_idx[:,0],positive_idx[:,1]]=(bbox_x-x_a)/(w_a+1e-9)
      ground_coord[1,positive_idx[:,0],positive_idx[:,1]]=(bbox_y-y_a)/(h_a+1e-9)
      ground_coord[2,positive_idx[:,0],positive_idx[:,1]]=torch.log(bbox_w/(w_a+1e-9))
      ground_coord[3,positive_idx[:,0],positive_idx[:,1]]=torch.log(bbox_h/(h_a+1e-9))
      ground_class=torch.unsqueeze(labels,0)

      self.ground_dict[key] = (ground_class, ground_coord)
      assert ground_class.shape==(1,grid_size[0],grid_size[1])
      assert ground_coord.shape==(4,grid_size[0],grid_size[1])

      return ground_class, ground_coord

    # Compute the loss of the classifier
    # Input:
    #      p_out:     (positives_on_mini_batch)  (output of the classifier for sampled anchors with positive gt labels)
    #      n_out:     (negatives_on_mini_batch) (output of the classifier for sampled anchors with negative gt labels
    def loss_class(self,p_out,n_out):
        assert p_out.dim() == 1
        assert n_out.dim() == 1
        # compute classifier's loss
        cls_loss = torch.nn.BCELoss(reduction='mean')

        N_pos = p_out.shape[0]
        N_neg = n_out.shape[0]
        sum_count = N_pos + N_neg

        pred = torch.cat([p_out, n_out])
        gt = torch.cat([torch.ones(N_pos, device=self.device, dtype=torch.float),
                        torch.zeros(N_neg, device=self.device, dtype=torch.float)])
        loss = cls_loss(pred, gt)

        return loss,sum_count



    # Compute the loss of the regressor
    # Input:
    #       pos_target_coord: (positive_on_mini_batch,4) (ground truth of the regressor for sampled anchors with positive gt labels)
    #       pos_out_r: (positive_on_mini_batch,4)        (output of the regressor for sampled anchors with positive gt labels)
    def loss_reg(self,pos_target_coord,pos_out_r):
        # compute regressor's loss
        assert pos_out_r.dim() == 2
        assert pos_target_coord.dim() == 2
        assert pos_target_coord.shape == pos_out_r.shape
        assert pos_out_r.shape[1] == 4

        reg_loss = torch.nn.SmoothL1Loss(reduction = 'mean')
        loss = reg_loss(pos_out_r, pos_target_coord)
        sum_count = pos_out_r.shape[0]

        return loss, sum_count



    # Compute the total loss
    # Input:
    #       cls_out: (bz,1,grid_size[0],grid_size[1])
    #       reg_out: (bz,4,grid_size[0],grid_size[1])
    #       targ_cls:(bz,1,grid_size[0],grid_size[1])
    #       targ_reg:(bz,4,grid_size[0],grid_size[1])
    #       l: lambda constant to weight between the two losses
    #       effective_batch: the number of anchors in the effective batch (M in the handout)
    def compute_loss(self,cls_out,reg_out,targ_cls,targ_reg, l=1, effective_batch=50):
        #############################
        # compute the total loss
        #############################
        assert cls_out.shape[1] == 1
        assert reg_out.shape[1] == 4
        assert targ_cls.shape[1] == 1
        assert targ_reg.shape[1] == 4

        N_pos = int(effective_batch / 2)
        N_neg = effective_batch - N_pos

        # sampling
        targ_cls_3 = targ_cls.squeeze(dim=1)                # 3-dimensional target cls tensor
        pos_cord_all = torch.nonzero(targ_cls_3 == 1)       # (N_pos_all, 3): indice on dimension 0, 2, 3
        neg_cord_all = torch.nonzero(targ_cls_3 == 0)

        # sampling positive
        if pos_cord_all.shape[0] > N_pos:                   # sample positive
            # sampling indice: choose N_pos samples from all positive (pos_cord_all.shape[0])
            pos_indice_keep = np.random.choice(pos_cord_all.shape[0], N_pos)
            pos_cord_keep = pos_cord_all[pos_indice_keep, :]        # (N_pos, 3)
        else:               # skip positive sampling, update N_pos / N_neg
            print("[INFO] Not enough samples for positive sampling. Expected: {}, Actual: {}".format(
                N_pos, pos_cord_all.shape[0]))
            pos_cord_keep = pos_cord_all
            N_pos = pos_cord_keep.shape[0]
            N_neg = effective_batch - N_pos

        # sampling negative
        if neg_cord_all.shape[0] < N_neg:
            print("[WARN] not enough sample for negative sampling. Required: {}, Available: {}".format(
                N_neg, neg_cord_all.shape[0]))
        # sampling indice: choose N_pos samples from all positive (pos_cord_all.shape[0])
        neg_indice_keep = np.random.choice(neg_cord_all.shape[0], N_neg)
        neg_cord_keep = neg_cord_all[neg_indice_keep, :]            # (N_neg, 3)

        # sampling (fetching values)
        # assert torch.all(targ_cls[pos_cord_keep[:, 0], 0, pos_cord_keep[:, 1], pos_cord_keep[:, 2]] == 1)
        # assert torch.all(targ_cls[neg_cord_keep[:, 0], 0, neg_cord_keep[:, 1], neg_cord_keep[:, 2]] == 0)
        p_out = cls_out[pos_cord_keep[:, 0], 0, pos_cord_keep[:, 1], pos_cord_keep[:, 2]]
        n_out = cls_out[neg_cord_keep[:, 0], 0, neg_cord_keep[:, 1], neg_cord_keep[:, 2]]
        pos_out_r = reg_out[pos_cord_keep[:, 0], :, pos_cord_keep[:, 1], pos_cord_keep[:, 2]]
        pos_target_coord = targ_reg[pos_cord_keep[:, 0], :, pos_cord_keep[:, 1], pos_cord_keep[:, 2]]

        # compute loss
        loss_c, cls_count = self.loss_class(p_out, n_out)
        loss_r, reg_count = self.loss_reg(pos_target_coord, pos_out_r)
        # todo: (jianxiong): normalize with count number?
        # Note: provided testcase did not normalize
        loss = loss_c + l * loss_r

        return loss, loss_c, loss_r



    # Post process for the outputs for a batch of images
    # Input:
    #       out_c:  (bz,1,grid_size[0],grid_size[1])}
    #       out_r:  (bz,4,grid_size[0],grid_size[1])}
    #       IOU_thresh: scalar that is the IOU threshold for the NMS
    #       keep_num_preNMS: number of masks we will keep from each image before the NMS
    #       keep_num_postNMS: number of masks we will keep from each image after the NMS
    # Output:
    #       nms_clas_list: list:len(bz){(Post_NMS_boxes)} (the score of the boxes that the NMS kept)
    #       nms_prebox_list: list:len(bz){(Post_NMS_boxes,4)} (the coordinates of the boxes that the NMS kept)

    def postprocess(self,out_c,out_r, batch_image, indexes, IOU_thresh=0.5, keep_num_preNMS=50, keep_num_postNMS=10 ):
       ####################################
       # TODO postprocess a batch of images
       #####################################
       batch_size = out_c.shape[0]
       nms_clas_list=[]
       nms_prebox_list=[]
       for bz in range(batch_size):
           nms_clas, nms_prebox = self.postprocessImg(out_c[bz], out_r[bz], batch_image[bz], indexes[bz], IOU_thresh, keep_num_preNMS, keep_num_postNMS)
           nms_clas_list.append(nms_clas)
           nms_prebox_list.append(nms_prebox)

       return nms_clas_list, nms_prebox_list


    # Input:
    #       flatten_cls: (Pre/Post_NMS_boxes)
    #       flatten_box: (Pre/Post_NMS_boxes,4) (upper_left_x,upper_left_y, lower_right_x, lower_right_y)

    def plot_imgae_NMS(self,flatten_box, image, visual_dir, index, mode, top_num):
        ####################################
        # TODO plot image before and after NMS
        #####################################
        image = transforms.functional.normalize(image,
                                                [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
        num_box=flatten_box.shape[0]
        fig, ax = plt.subplots(1, 1)
        image_vis = image.permute(1, 2, 0).cpu().detach().numpy()
        ax.imshow(image_vis)

        for elem in range(num_box):
            coord = flatten_box[elem, :].view(-1)
            coord = coord.cpu().detach().numpy()
            # plot bbox
            rect = patches.Rectangle((coord[0], coord[1]), coord[2] - coord[0], coord[3] - coord[1], fill=False,
                                     color='b')
            ax.add_patch(rect)

        plt.title("{} NMS Top {}".format(mode, top_num))
        plt.savefig("{}/{}.png".format(visual_dir, index))
        plt.show()
        plt.close('all')

    # Post process the output for one image
    # Input:
    #      mat_clas: (1,grid_size[0],grid_size[1])}  (scores of the output boxes)
    #      mat_coord: (4,grid_size[0],grid_size[1])} (encoded coordinates of the output boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4) (decoded coordinates of the boxes that the NMS kept)


    def postprocessImg(self,mat_clas,mat_coord, image, index, IOU_thresh,keep_num_preNMS, keep_num_postNMS):
            ######################################
            # TODO postprocess a single image
            #####################################
            reg_out = torch.unsqueeze(mat_coord, 0)
            cls_out = torch.unsqueeze(mat_clas, 0)
            flatten_coord, flatten_cls, flatten_anchors = output_flattening(reg_out, cls_out, self.anchors)
            decoded_preNMS_flatten_box = output_decoding(flatten_coord, flatten_anchors)
            a = [torch.rand(flatten_coord.shape[0]) > 0.5 for i in range(4)]
            x_low_outbound = (decoded_preNMS_flatten_box[:, 0] < 0)
            y_low_outbound = (decoded_preNMS_flatten_box[:, 1] < 0)
            x_up_outbound = (decoded_preNMS_flatten_box[:, 2] > 1088)
            y_up_outbound = (decoded_preNMS_flatten_box[:, 3] > 800)
            a[0] = x_low_outbound
            a[1] = y_low_outbound
            a[2] = x_up_outbound
            a[3] = y_up_outbound
            outbound_flatten_mask = (torch.sum(torch.stack(a), dim=0) > 0)
            flatten_cls[outbound_flatten_mask] = 0

            top_values, top_indices = torch.topk(flatten_cls, keep_num_preNMS)
            last_value = top_values[-1]
            topk_mask = flatten_cls >= last_value
            topk_cls = flatten_cls[topk_mask]
            topk_box = decoded_preNMS_flatten_box[topk_mask]
            pre_nms_dir = "PreNMS"
            self.plot_imgae_NMS(topk_box, image, pre_nms_dir,index, "Pre", keep_num_preNMS)

            nms_clas, nms_prebox = self.NMS(topk_cls,topk_box, IOU_thresh)

            num = min(nms_prebox.shape[0],keep_num_postNMS)
            top_values, top_indices = torch.topk(nms_clas, num)
            last_value = top_values[-1]
            topk_mask = nms_clas >= last_value
            topn_cls = nms_clas[topk_mask]
            topn_box = nms_prebox[topk_mask]
            post_nms_dir = "PostNMS"
            self.plot_imgae_NMS(topn_box, image, post_nms_dir,index, "Post", keep_num_postNMS)

            return topn_cls, topn_box


    # Input:
    #       clas: (top_k_boxes) (scores of the top k boxes)
    #       prebox: (top_k_boxes,4) (coordinate of the top k boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)

    def NMS(self,clas,prebox, thresh):
        ##################################
        # TODO perform NSM
        ##################################
        num_box = prebox.shape[0]

        # IOU matrix
        iou_mat = torch.zeros((num_box, num_box),device=self.device)
        for x in range(num_box):
            for y in range(num_box):
                iou_mat[x, y] = IOU_edge_point(torch.unsqueeze(prebox[x, :], 0), torch.unsqueeze(prebox[y, :], 0))
        max_index = set()

        # Suppressing small IOU
        for i in range(len(iou_mat)):
            group = []
            for j in range(len(iou_mat)):
                if iou_mat[i, j] > thresh:
                    group.append(j)
            if len(group) == 1:
                max_index.add(i)
                continue
            group_score = clas[group]
            max_conf_bbox_group = torch.max(group_score)
            index = (clas == max_conf_bbox_group).nonzero()
            index = index.cpu().flatten().numpy().astype(int)
            index_num = len(index)
            if len(index) > 1: index = index[0]
            index = int(index)
            max_index.add(index)
        all_index = list(range(len(iou_mat)))
        all_index = [elem for elem in all_index if elem not in list(max_index)]
        if len(all_index) == 0:
            return prebox, clas
        nms_clas = clas[list(max_index)]
        nms_prebox = prebox[list(max_index), :]
        return nms_clas, nms_prebox


if __name__=="__main__":
    pass
