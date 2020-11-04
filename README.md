# MASK_RCNN
680 HW Repo

## Dataset Link
https://drive.google.com/drive/folders/1eP7FtPaWfJ5zLdcsZYl6eyn5EYixkFn8?usp=sharing

## Dependency
- pytorch==1.1.0
- torchvision == 0.2.1
- matplotlib
- h5py
- wandb (pip install wandb)
    + For logging, you may disable that by setting LOGGING="" instead of "wandb"

## Usage
Train: (checkpoint saved to dir 'checkpoints')
```bash
python train.py
```

Eval + Visualization (checkpoint loaded from dir 'checkpoints_final')
```bash
python eval_rpn.py
```

ground-truth + image visualization
```bash
python dataset.py
```

ratio and scale hisyogram
```bash
python dataset_hist.py
```


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
