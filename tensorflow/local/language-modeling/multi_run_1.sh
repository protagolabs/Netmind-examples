sudo apt-get install jq
export PLATFORM=tensorflow
export PYTHONPATH="/home/xing/ly/NetMind-Mixin":${PYTHONPATH}
# export DATA_LOCATION="/home/xing/datasets/imagenet1k/train"
#set index to 0,1,2... to index for indicating trainer index
export TF_CONFIG=`cat config.json | jq '.task+={"index" : 1}'`
CUDA_VISIBLE_DEVICES="1" python train_netmind.py --data="/data/tiny-imagenet/tiny-imagenet-200"
