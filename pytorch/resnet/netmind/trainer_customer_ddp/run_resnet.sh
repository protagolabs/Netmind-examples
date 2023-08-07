python  -m torch.distributed.run --nproc_per_node=2 --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --master_port 57920 \
train_dist.py --data="/data/imagenet2012" \
--per_device_train_batch_size=128 \
--learning_rate=0.05 \
--num_train_epochs=90 \
