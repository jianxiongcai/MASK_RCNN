  import torch
  path='./Ground_Truth/ground_truth_index_[1075].pt'
  a=torch.load(path)
  bboxes=a['bboxes']
  index=a['index']
  grid_size=a['grid_size']
  anchors=a['anchors']
  image_size=a['image_size']
  ground_clas1=a['ground_clas']
  ground_coord1=a['ground_coord']
  stride=16
    #bboxes_cx=(bboxes[:,2]+bboxes[:,0])*0.5
    #bboxes_cy=(bboxes[:,3]+bboxes[:,1])*0.5
    #bboxes_w=bboxes[:,2]-bboxes[:,0]
    #bboxes_h=bboxes[:,3]-bboxes[:,1]
  bboxes_c=bboxes
  aspect_ratio=1.0
  scale=256.0


    
  stride=self.anchors_param['stride']
  labels=-torch.ones(grid_size[0],grid_size[1]).long()
  w=anchors[0,0,2]
  h=anchors[0,0,3]
  anchor_inbound_list=[]
  for i in range(grid_size[0]):
    for j in range(grid_size[1]):
      if anchors[i,j,0]<w*0.5 or anchors[i,j,1]<h*0.5 or anchors[i,j,0]>image_size[1]-w*0.5 or anchors[i,j,1]>image_size[0]-h*0.5:
        continue
      anchor_inbound_list.append(anchors[i,j].float())
  anchor_inbound=torch.stack(anchor_inbound_list) #(1800,4) i,j,w,h
  num_anchor_inbound=anchor_inbound.shape[0]
  iou_inbound_anchor_list=[]
  positive_inbound_anchor_list=[]
  negative_inbound_anchor_list=[]
  max_iou_all=0
  for obj_idx in range(bboxes.shape[0]):    
    bbox_single=bboxes[obj_idx].view(1,-1)
    bbox_n=bbox_single.repeat(num_anchor_inbound,1)
    iou=IOU(bbox_n,anchor_inbound)
    iou_inbound_anchor_list.append(iou)
    iou_low_mask=(iou<0.3)
    negative_inbound_anchor_list.append(iou_low_mask)
    iou_high_mask=(iou>0.7)
    max_iou=torch.max(iou)
    max_iou_all=max(max_iou,max_iou_all)
    max_iou_idx=torch.argmax(iou)
    iou_high_mask[max_iou_idx]=True
    positive_inbound_anchor_list.append(iou_high_mask)
  iou_inbound_anchor=torch.stack(iou_inbound_anchor_list)
  positive_mask = torch.tensor([any(tup) for tup in list(zip(*positive_inbound_anchor_list))])
  positive_idx=torch.squeeze(anchor_inbound[positive_mask.nonzero(),:2].float()/stride-0.5).long()
  if max_iou_all<=0.7:
      positive_idx=torch.unsqueeze(positive_idx,0)
  positive_idx=torch.index_select(positive_idx, 1, torch.LongTensor([1,0]))
  print(positive_idx)
  negative_mask = torch.tensor([all(tup) for tup in list(zip(*negative_inbound_anchor_list))])
  negative_idx=torch.squeeze(anchor_inbound[negative_mask.nonzero()[:],:2].float()/stride-0.5).long()
  negative_idx=torch.index_select(negative_idx, 1, torch.LongTensor([1,0]))
  print(negative_idx.shape)
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
