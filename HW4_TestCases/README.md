# HW4 Part A Ground Truth and Loss test cases
All the files are .pt format so you can load them using torch.load()

## Anchor Test Case 
Inside folder Anchor_Test we provide 4 .pt files.\
Each file contains an anchor tensor (50,68,4) that was created by the
following pairs of aspect ratio, scale\
ratio/scale: 0.5/128, 0.8/512, 1/256, 2/256\
Which file corresponds to which is shown in the name of the file


## Ground Truth test
Inside folder Ground_Truth we provide 5 .pt files\
Each file contains a dictionary with the following\
* The inputs of the function create_ground_truth: 'bboxes', 'index', 'grid_size', 'anchors', 'image_size
* The expected outputs: 'ground_clas', 'ground_coord'

In ground_clas we are assigning with 1 the positive anchors with 0 the negative and with -1 the neither
positive or negative ones \
In ground_coord we save the encoded coordinates t_x,t_y,t_w,t_h

## Loss test
Inside folder Loss we provide 10 .pt files\
Each one contains a dictionary with the following\
* The inputs of compute_loss: 'clas_out', 'regr_out', 'targ_clas', 'targ_regr', 'effective_batch'
* The inputs to the function loss_class after the sampling: 'p_out', 'n_out'
* The inputs to the function loss_reg after the sampling: 'pos_target_coord', 'pos_out_r'
* The output of loss_class: 'loss_c'
* The output of loss_reg: 'loss_r'

The ground truths are encoded the same way as the ground truth test.
The loss are computed using reduction mean so they are normalized with 
p_out.shape[0]+n_out.shape[0] for the loss_c and with pos_out_r.shape[0] for loss_r.
You may consider renormalize them in your implemantation

Also it is difficult to check your sampling using this test cases, in general 
check if the number of positives after the sampling is p_out.shape[0]=min(total_positives,M/2)
and the number of negatives is n_out.shape[0]=min(M-p_out.shape[0],total_negatives)   