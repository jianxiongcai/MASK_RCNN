import numpy as np
import torch
from functools import partial
def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))

# This function computes the IOU between two set of boxes
def IOU(bbox_1, bbox_2):
    ##################################
    #TODO compute the IOU between the boxA, boxB boxes
    ##################################
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
      iou=(inter_area+ 1e-9)/(union_area+1e-9)  
        
      return iou
    
    



# This function flattens the output of the network and the corresponding anchors 
# in the sense that it concatenates  the outputs and the anchors from all the grid cells
# from all the images into 2D matrices
# Each row of the 2D matrices corresponds to a specific anchor/grid cell
# Input:
#       out_r: (bz,4,grid_size[0],grid_size[1])
#       out_c: (bz,1,grid_size[0],grid_size[1])
#       anchors: (grid_size[0],grid_size[1],4)
# Output:
#       flatten_coord: (bz*grid_size[0]*grid_size[1],4)
#       flatten_gt: (bz*grid_size[0]*grid_size[1])
#       flatten_anchors: (bz*grid_size[0]*grid_size[1],4)
def output_flattening(out_r,out_c,anchors):
    #######################################
    # TODO flatten the output tensors and anchors
    #######################################
    bz=out_r.shape[0]
    
    out_c=torch.squeeze(out_c, dim=1)
    flatten_gt=torch.flatten(out_c, start_dim=0, end_dim=2)
    
    out_r=out_r.permute(0,2,3,1)
    flatten_coord=torch.flatten(out_r, start_dim=0, end_dim=2)
    
    anchors=torch.unsqueeze(anchors,0)
    anchors=anchors.repeat(bz,1,1,1)
    flatten_anchors=torch.flatten(anchors,start_dim=0, end_dim=2)
    assert flatten_coord.shape==(bz*50*68,4)
    assert flatten_gt.shape==(bz*50*68,)
    assert flatten_anchors.shape==(bz*50*68,4)
    return flatten_coord, flatten_gt, flatten_anchors
    




# This function decodes the output that is given in the encoded format (defined in the handout)
# into box coordinates where it returns the upper left and lower right corner of the proposed box
# Input:
#       flatten_out: (total_number_of_anchors*bz,4)
#       flatten_anchors: (total_number_of_anchors*bz,4)
# Output:
#       box: (total_number_of_anchors*bz,4)
def output_decoding(flatten_out,flatten_anchors, device='cpu'):
    #######################################
    # TODO decode the output
    #######################################
    x=flatten_out.shape[0]
    x_c=flatten_out[:,0]*flatten_anchors[:,2]+flatten_anchors[:,0]
    y_c=flatten_out[:,1]*flatten_anchors[:,3]+flatten_anchors[:,1]
    w=torch.exp(flatten_out[:,2]+torch.log(flatten_anchors[:,2].float()))
    h=torch.exp(flatten_out[:,3]+torch.log(flatten_anchors[:,3].float()))
    xl=x_c-0.5*w
    yl=y_c-0.5*h
    xup=x_c+0.5*w
    yup=y_c+0.5*h
    box=torch.stack((xl,yl,xup,yup)).transpose(1,0)
    
    assert box.shape==(x,4)
      
    return box