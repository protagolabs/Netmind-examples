export INDEX=0 # the index of gpu
export PLATFORM=tensorflow
export AGENT_ENABLE=0
export PYTHONPATH="/home/yang.li/NetMind-Mixin":${PYTHONPATH}
#export DATA_LOCATION=/home/yang.li/albert_tokenized_wikitext
CUDA_VISIBLE_DEVICES="0" python train_netmind.py --data="/home/yang.li/albert_tokenized_wikitext"
