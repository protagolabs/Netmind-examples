export INDEX=1 # the index of gpu
export PLATFORM=tensorflow
export DATA_LOCATION="./data_mlm"
CUDA_VISIBLE_DEVICES="1" python train_netmind.py