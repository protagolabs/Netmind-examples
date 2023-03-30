export INDEX=1 # the index of gpu
export PLATFORM=tensorflow
CUDA_VISIBLE_DEVICES="1" python train_netmind.py \
--data="$YOUR_HOME_DIR/Netmind-examples/tensorflow/datasets/tiny-imagenet-200/train" \
--val_data="$YOUR_HOME_DIR/Netmind-examples/tensorflow/datasets/tiny-imagenet-200/val"
