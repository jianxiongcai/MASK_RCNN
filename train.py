"""
Training Main
"""

from dataset import BuildDataset, BuildDataLoader
from rpn import RPNHead

import os.path
import torch.backends.cudnn
import torch.utils.data
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm

#reproductivity
torch.random.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

# =========================== Config ==========================
batch_size = 4
init_lr = 1e-3
num_epochs = 60
milestones = [30, 45, 55]
loss_ratio = 4

# =========================== Logging ==========================
# w and b login
# LOGGING = ""
LOGGING = "wandb"
if LOGGING == "wandb":
    assert os.system("wandb login $(cat wandb_secret)") == 0
    wandb.init(project="hw4")
    wandb.config.update({
        'batch_size': batch_size,
        'init_lr': init_lr,
        'num_epochs': num_epochs,
        'loss_ratio': loss_ratio
    })
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# =========================== Dataset ==============================
# file path and make a list
imgs_path = 'data/hw3_mycocodata_img_comp_zlib.h5'
masks_path = 'data/hw3_mycocodata_mask_comp_zlib.h5'
labels_path = "data/hw3_mycocodata_labels_comp_zlib.npy"
bboxes_path = "data/hw3_mycocodata_bboxes_comp_zlib.npy"
paths = [imgs_path, masks_path, labels_path, bboxes_path]

dataset = BuildDataset(paths,augmentation=True)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# dataset
train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
train_loader = train_build_loader.loader()
test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
test_loader = test_build_loader.loader()

# ============================ Train ================================
rpn_head = RPNHead(device=device).to(device)
optimizer = optim.Adam(rpn_head.parameters(), lr=init_lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

# train_cls_loss_list = []
# train_reg_loss_list = []
# train_tot_loss_list = []
os.makedirs("checkpoints", exist_ok=True)

# watch with wandb
if LOGGING == "wandb":
    wandb.watch(rpn_head)
for epoch in range(num_epochs):
    rpn_head.train()
    train_cls_loss = 0.0
    train_reg_loss = 0.0
    train_tot_loss = 0.0

    # ============================== EPOCH START ==================================
    for iter, data in enumerate(tqdm(train_loader), 0):
        img = data['images'].to(device)
        # label_list = [x.to(device) for x in data['labels']]
        # mask_list = [x.to(device) for x in data['masks']]
        bbox_list = [x.to(device) for x in data['bbox']]
        index_list = data['index']
        img_shape = (img.shape[2], img.shape[3])

        # logits: (bz,1,grid_size[0],grid_size[1])}
        # bbox_regs: (bz,4, grid_size[0],grid_size[1])}
        optimizer.zero_grad()
        cls_out, reg_out = rpn_head(img)
        del img
        targ_cls, targ_reg = rpn_head.create_batch_truth(bbox_list, index_list, img_shape)
        del data, bbox_list, index_list

        # compute loss and optimize
        # set l = 4, the raw regression loss is normalized for each bounding box coordinate
        loss, loss_c, loss_r = rpn_head.compute_loss(
            cls_out, reg_out, targ_cls, targ_reg, l=loss_ratio, effective_batch=50)

        if epoch != 0:          # epoch 0 is reference epoch
            loss.backward()
            optimizer.step()

        # logging
        train_cls_loss += loss_c.item()
        train_reg_loss += loss_r.item()
        train_tot_loss += loss.item()
        if np.isnan(train_tot_loss):
            raise RuntimeError("[ERROR] NaN encountered at iter: {}".format(iter))
    # ================================= EPOCH END ==================================
    # logging per epoch
    # save to files
    # train_cls_loss.append(logging_cls_loss)
    # train_reg_loss.append(logging_reg_loss)
    # train_tot_loss.append(logging_tot_loss)
    print('Epoch:{} Sum. train total loss: {:.4f}, loss cls: {}, loss reg: {}'.format(epoch, train_tot_loss, train_cls_loss, train_reg_loss))
    if LOGGING == "wandb":
        wandb.log({"train/cls_loss": train_cls_loss,
                   "train/reg_loss": train_reg_loss,
                   "train/tot_loss": train_tot_loss}, step=epoch)

    # do validation
    rpn_head.eval()
    test_cls_loss = 0.0
    test_reg_loss = 0.0
    test_tot_loss = 0.0
    for iter, data in enumerate(tqdm(test_loader), 0):
        img = data['images'].to(device)
        bbox_list = [x.to(device) for x in data['bbox']]
        index_list = data['index']
        img_shape = (img.shape[2], img.shape[3])
        with torch.no_grad():
            cls_out, reg_out = rpn_head(img)
            targ_cls, targ_reg = rpn_head.create_batch_truth(bbox_list, index_list, img_shape)
            loss, loss_c, loss_r = rpn_head.compute_loss(
                cls_out, reg_out, targ_cls, targ_reg, l=loss_ratio, effective_batch=50)
            # logging
            test_cls_loss += loss_c.item()
            test_reg_loss += loss_r.item()
            test_tot_loss += loss.item()
    # logging per epoch
    print('Epoch:{} Sum. test total loss: {:.4f}, test cls: {}, test reg: {}'.format(epoch, test_tot_loss,
                                                                                      test_cls_loss,
                                                                                      test_reg_loss))
    if LOGGING == "wandb":
        wandb.log({"test/cls_loss": test_cls_loss,
                   "test/reg_loss": test_reg_loss,
                   "test/tot_loss": test_tot_loss}, step=epoch)



    # save checkpoint
    path = 'checkpoints/epoch_' + str(epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': rpn_head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    if epoch != 0:
        scheduler.step()
