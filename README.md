# MASK_RCNN
680 HW Repo

## Dataset Link
https://drive.google.com/drive/folders/1eP7FtPaWfJ5zLdcsZYl6eyn5EYixkFn8?usp=sharing

## Runtime environment
The network is trained locally with:
- pytorch==1.1.0
- torchvision==0.2.1
You may use the provided docker environment: jianxiongcai/pytorch-1.1.0
```
docker pull jianxiongcai/pytorch-1.1.0
docker run -it --gpus all --ipc host jianxiongcai/pytorch-1.1.0
```
Note: Pytorch has some different behavior in 1.1.0 and 1.7.0
