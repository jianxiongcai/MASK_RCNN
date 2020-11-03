if [ "$(uname)" == "Darwin" ]; then
  echo "Running on MacOS"
  gpu_option=""
fi

if [ "$(uname)" == "Linux" ]; then
  echo "Running on Linux"
  gpu_option="--gpus all"
fi

docker run -d -it ${gpu_option} --ipc host -p 8890:8888 -p 6008:6006 -v $PWD:/workspace/MASK_RCNN --name 680_maskrcnn jianxiongcai/pytorch-1.1.0
