import torch
from torch.nn import functional as F
from torchvision import transforms
from torch import nn, Tensor
from dataset import *
from utils import *

import torchvision


class RPNHead(torch.nn.Module):

    def __init__(self,  device='cuda', anchors_param=dict(ratio=0.8,scale= 256, grid_size=(50, 68), stride=16)):
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





    # Forward  the input through the backbone the intermediate layer and the RPN heads
    # Input:
    #       X: (bz,3,image_size[0],image_size[1])}
    # Ouput:
    #       logits: (bz,1,grid_size[0],grid_size[1])}
    #       bbox_regs: (bz,4, grid_size[0],grid_size[1])}
    def forward(self, X):

        # forward through the Backbone
        X = self.backbone(X)

        # forward through the Intermediate layer
        X = self.interm_layer(X)

        # forward through the Classifier Head
        logits = self.cls_head(X)

        # forward through the Regressor Head
        reg_output =  self.reg_head(X)

        # todo: decode reg_output (w.r.t. anchor cell grid) to bbox_regs

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



    # This function creates the anchor boxes
    # Output:
    #       anchors: (grid_size[0],grid_size[1],4)
    def create_anchors(self, aspect_ratio, scale, grid_sizes, stride):
        ######################################
        # TODO create anchors
        ######################################
        h_2=torch.tensor(scale**2/aspect_ratio,dtype=float)
        h=torch.round(torch.sqrt(h_2)).long()
        w=torch.round(aspect_ratio*h).long()
        anchors=torch.zeros(grid_sizes[0],grid_sizes[1],4).long()
        x=torch.arange(grid_sizes[0])
        y=torch.arange(grid_sizes[1])
        xx,yy=torch.meshgrid(x,y)
        anchors[xx,yy,0]=((xx+0.5)*stride).long()
        anchors[xx,yy,1]=((yy+0.5)*stride).long()
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
    #      ground_clas: (bz,1,grid_size[0],grid_size[1])
    #      ground_coord: (bz,4,grid_size[0],grid_size[1])
    def create_batch_truth(self,bboxes_list,indexes,image_shape):
        #####################################
        # TODO create ground truth for a batch of images
        #####################################
        assert ground_clas.shape[1:4]==(1,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
        assert ground_coord.shape[1:4]==(4,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return ground_clas, ground_coord


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

        self.ground_dict[key] = (ground_clas, ground_coord)

        assert ground_clas.shape==(1,grid_size[0],grid_size[1])
        assert ground_coord.shape==(4,grid_size[0],grid_size[1])

        return ground_clas, ground_coord





    # Compute the loss of the classifier
    # Input:
    #      p_out:     (positives_on_mini_batch)  (output of the classifier for sampled anchors with positive gt labels)
    #      n_out:     (negatives_on_mini_batch) (output of the classifier for sampled anchors with negative gt labels
    def loss_class(self,p_out,n_out):

        #torch.nn.BCELoss()
        # TODO compute classifier's loss

        return loss,sum_count



    # Compute the loss of the regressor
    # Input:
    #       pos_target_coord: (positive_on_mini_batch,4) (ground truth of the regressor for sampled anchors with positive gt labels)
    #       pos_out_r: (positive_on_mini_batch,4)        (output of the regressor for sampled anchors with positive gt labels)
    def loss_reg(self,pos_target_coord,pos_out_r):
            #torch.nn.SmoothL1Loss()
            # TODO compute regressor's loss

            return loss, sum_count



    # Compute the total loss
    # Input:
    #       clas_out: (bz,1,grid_size[0],grid_size[1])
    #       regr_out: (bz,4,grid_size[0],grid_size[1])
    #       targ_clas:(bz,1,grid_size[0],grid_size[1])
    #       targ_regr:(bz,4,grid_size[0],grid_size[1])
    #       l: lambda constant to weight between the two losses
    #       effective_batch: the number of anchors in the effective batch (M in the handout)
    def compute_loss(self,clas_out,regr_out,targ_clas,targ_regr, l=1, effective_batch=50):
            #############################
            # TODO compute the total loss
            #############################
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
    def postprocess(self,out_c,out_r, IOU_thresh=0.5, keep_num_preNMS=50, keep_num_postNMS=10):
       ####################################
       # TODO postprocess a batch of images
       #####################################
        return nms_clas_list, nms_prebox_list



    # Post process the output for one image
    # Input:
    #      mat_clas: (1,grid_size[0],grid_size[1])}  (scores of the output boxes)
    #      mat_coord: (4,grid_size[0],grid_size[1])} (encoded coordinates of the output boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4) (decoded coordinates of the boxes that the NMS kept)
    def postprocessImg(self,mat_clas,mat_coord, IOU_thresh,keep_num_preNMS, keep_num_postNMS):
            ######################################
            # TODO postprocess a single image
            #####################################

            return nms_clas, nms_prebox



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
        return nms_clas,nms_prebox
    
if __name__=="__main__":
