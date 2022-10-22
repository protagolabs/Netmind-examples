export INDEX=0 # the index of gpu
export PLATFORM=tensorflow
export DATA_LOCATION="./data_mlm"
CUDA_VISIBLE_DEVICES="0" python train_netmind.py