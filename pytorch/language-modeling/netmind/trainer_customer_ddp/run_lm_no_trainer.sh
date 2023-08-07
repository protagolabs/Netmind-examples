export PLATFORM=pytorch && export USE_DDP=1 && \
python  -m torch.distributed.run --nproc_per_node=2 --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --master_port 57920 \
train_dist.py --data="data/bert" \
--model_name_or_path=bert-base-uncased \
--per_device_train_batch_size=4 \
--learning_rate=1e-4 \
--num_train_epochs=5 \
--save_step=1000000 \
--output_dir="saved_model" 
