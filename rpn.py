import torch
from torch.nn import functional as F
from torchvision import transforms
from torch import nn, Tensor
from dataset import *
from utils import *
import os.path
import shutil
import torchvision
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import gc
gc.enable()


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
    #      ground_class: (bz,1,grid_size[0],grid_size[1])
    #      ground_coord: (bz,4,grid_size[0],grid_size[1])
    def create_batch_truth(self,bboxes_list,indexes,image_shape):
        #####################################
        # TODO create ground truth for a batch of images
        #####################################     
        bz=len(bboxes_list)
        grid_size=(self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
        ground_class=torch.zeros(bz,1,grid_size[0],grid_size[1])
        ground_coord=torch.zeros(bz,4,grid_size[0],grid_size[1])
        for idx in range(bz):
            ground_class[idx,:,:,:],ground_coord[idx,:,:,:]=self.create_ground_truth(bboxes_list[idx], indexes[idx], grid_size, self.anchors, image_shape)
        
        assert ground_class.shape[1:4]==(1,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
        assert ground_coord.shape[1:4]==(4,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return ground_class, ground_coord


    # This function calculate iou matrix of two sets of bboxes with expression of [c_x,c_y,w,h]
    # bbox:(num_box,4)
    def IOU(self,bbox_1 ,bbox_2):
          x_1up,y_1up,x_1l,y_1l=bbox_1[:,0]-0.5*bbox_1[:,2],bbox_1[:,1]-0.5*bbox_1[:,3],bbox_1[:,0]+0.5*bbox_1[:,2],bbox_1[:,1]+0.5*bbox_1[:,3]
          x_2up,y_2up,x_2l,y_2l=bbox_2[:,0]-0.5*bbox_2[:,2],bbox_2[:,1]-0.5*bbox_2[:,3],bbox_2[:,0]+0.5*bbox_2[:,2],bbox_2[:,1]+0.5*bbox_2[:,3]
          
          x_up=torch.max(x_1up,x_2up)
          y_up=torch.max(y_1up,y_2up)
        
          x_l=torch.min(x_1l,x_2l)
          y_l=torch.min(y_1l,y_2l)
        
          inter_area = (x_l-x_up).clamp(min=0) * (y_l-y_up).clamp(min=0)
        
          area_box1 = (x_1l-x_1up).clamp(min=0) * (y_1l-y_1up).clamp(min=0)
          area_box2 = (x_2l-x_2up).clamp(min=0) * (y_2l-y_2up).clamp(min=0)
          union_area=area_box1+area_box2-inter_area
          iou=(inter_area+ 1e-3)/(union_area+1e-3)  
        
          return iou
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
        labels=-torch.ones(grid_size[0],grid_size[1]).long()
        w=anchors[0,0,2]
        h=anchors[0,0,3]
        anchor_inbound_list=[]
        for i in range(anchors.shape[0]):
          for j in range(anchors.shape[1]):
            if anchors[i,j,0]<w*0.5 or anchors[i,j,1]<h*0.5 or anchors[i,j,0]>image_size[0]-w*0.5 or anchors[i,j,1]>image_size[1]-h*0.5:
              continue
            anchor_inbound_list.append(anchors[i,j])
        anchor_inbound=torch.stack(anchor_inbound_list) #(1800,4)
        num_anchor_inbound=anchor_inbound.shape[0]
        
        iou_inbound_anchor_list=[]
        positive_inbound_anchor_list=[]
        negative_inbound_anchor_list=[]
        for obj_idx in range(bboxes.shape[0]):
          box_cx=bboxes[obj_idx][0].numpy()
          box_cy=bboxes[obj_idx][1].numpy()
          box_w=bboxes[obj_idx][2].numpy()
          box_h=bboxes[obj_idx][3].numpy()
          bbox_single=bboxes[obj_idx].view(1,-1)
          bbox_n=bbox_single.repeat(num_anchor_inbound,1)
          iou=self.IOU(bbox_n,anchor_inbound)
          iou_inbound_anchor_list.append(iou)
          iou_low_mask=(iou<0.3)
          negative_inbound_anchor_list.append(iou_low_mask)
          iou_high_mask=(iou>0.7)
          max_iou=torch.max(iou)
          max_iou_idx=torch.argmax(iou)
          iou_high_mask[max_iou_idx]=True
          positive_inbound_anchor_list.append(iou_high_mask)
        
        iou_inbound_anchor=torch.stack(iou_inbound_anchor_list)
        negative_mask = torch.tensor([all(tup) for tup in list(zip(*negative_inbound_anchor_list))])
        negative_idx=torch.squeeze(anchor_inbound[negative_mask.nonzero(),:2].float()/self.anchors_param['stride']-0.5).long()
        positive_mask = torch.tensor([any(tup) for tup in list(zip(*positive_inbound_anchor_list))])
        positive_idx=torch.squeeze(anchor_inbound[positive_mask.nonzero(),:2].float()/self.anchors_param['stride']-0.5).long()
        highest_iou_mask,highest_iou_bbox_idx=torch.max(iou_inbound_anchor, 0)
        labels[positive_idx[:,0],positive_idx[:,1]]=1
        labels[negative_idx[:,0],negative_idx[:,1]]=0
        ground_coord_orig=anchors.permute((2,0,1))
        ground_coord=ground_coord_orig
        
        highest_bbox_idx=highest_iou_bbox_idx[positive_mask]
        bbox_positive=bboxes[highest_bbox_idx]
        bbox_x=bbox_positive[:,0]
        bbox_y=bbox_positive[:,1]
        bbox_w=bbox_positive[:,2]
        bbox_h=bbox_positive[:,3]
        x_a=ground_coord_orig[:,positive_idx[:,0],positive_idx[:,1]][0,:]
        y_a=ground_coord_orig[:,positive_idx[:,0],positive_idx[:,1]][1,:]
        w_a=ground_coord_orig[:,positive_idx[:,0],positive_idx[:,1]][2,:]
        h_a=ground_coord_orig[:,positive_idx[:,0],positive_idx[:,1]][3,:]
        ground_coord[:,positive_idx[:,0],positive_idx[:,1]][0,:]=(bbox_x-x_a)/(w_a+1e-9)
        ground_coord[:,positive_idx[:,0],positive_idx[:,1]][1,:]=(bbox_y-y_a)/(h_a+1e-9)
        ground_coord[:,positive_idx[:,0],positive_idx[:,1]][2,:]=torch.log(bbox_w/(w_a+1e-9))
        ground_coord[:,positive_idx[:,0],positive_idx[:,1]][3,:]=torch.log(bbox_h/(h_a+1e-9))
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

#            return loss, sum_count
            pass



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
#            return loss, loss_c, loss_r
            pass



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
#        return nms_clas_list, nms_prebox_list
       pass


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

#            return nms_clas, nms_prebox
            pass



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
#        return nms_clas,nms_prebox
        pass
    
if __name__=="__main__":
    imgs_path = '../../data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = '../../data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = '../../data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = '../../data/hw3_mycocodata_bboxes_comp_zlib.npy'

    # set up output dir (for plotGT)
    try:
        shutil.rmtree("plotgt_result")
    except FileNotFoundError:
        pass
    os.makedirs("plotgt_result", exist_ok=True)

    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

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

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()


    resnet50_fpn = Resnet50Backbone()
    solo_head = SOLOHead(num_classes=4) ## class number is 4, because consider the background as one category.
    # loop the image
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # to gpu device
    resnet50_fpn = resnet50_fpn.to(device)
    solo_head = solo_head.to(device)

    for iter, data in enumerate(train_loader, 0):
        img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
        # fpn is a dict
        img = img.to(device)
        backout = resnet50_fpn(img)
        fpn_feat_list = list(backout.values())
        # make the target


        ## demo   
        #cate_pred_list[0]:bz, 3, num_grid,num_grid    len(cate_pred_list)=5               value [0,1]
        #ins_pred_list[0]:bz, num_grid^2, 2*H_feat, 2*W_feat   len(cate_pred_list)=5       value [0,1]
        cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False) 
        
        # len(ins_gts_list[0])=5, ins_gts_list[0][0]:num_grid^2, 2*H_feat, 2*W_feat   len(ins_gts_list)=bz   value:0 or 1
        # len(ins_ind_gts_list[0])=5, ins_gts_list[0][0]:num_grid^2   len(ins_ind_gts_list)=bz   value:0 or 1
        # len(cate_gts_list[0])=5, cate_gts_list[0][0]:num_grid,num_grid   len(ins_gts_list)=bz   value:0,1,2
        ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                        bbox_list,
                                                                        label_list,
                                                                        mask_list)
#        mask_color_list = ["jet", "ocean", "Spectral"]
#        solo_head.PlotGT(ins_gts_list,ins_ind_gts_list,cate_gts_list,mask_color_list,img)
#        # break
#
#        if (iter > 40):
#            break