export INDEX=0 # the index of gpu
export PLATFORM=tensorflow
export DATA_LOCATION="/home/xing/datasets/imagenet1k/train"
CUDA_VISIBLE_DEVICES="0" python train_netmind.py --val_data="/home/xing/datasets/imagenet1k/val"