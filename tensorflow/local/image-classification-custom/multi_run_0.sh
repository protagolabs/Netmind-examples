export INDEX=0 # the index of gpu
export PLATFORM=tensorflow
export PYTHONPATH="/home/xing/ly/NetMind-Mixin":${PYTHONPATH}
# export DATA_LOCATION="/home/xing/datasets/imagenet1k/train"
# CUDA_VISIBLE_DEVICES="0" python train_netmind.py --val_data="/home/xing/datasets/imagenet1k/val"
CUDA_VISIBLE_DEVICES="0" python train_netmind.py --data="/data/food-101/images"