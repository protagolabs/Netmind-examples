IP=/ip4/192.168.0.188/tcp/45533/p2p/QmXWCMLemfYmoV4sY4cqDTW1h5KTQbFJBRoxE4Hcd8NkCm

WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0 python run_trainer.py \
--experiment_prefix resnet_experiment \
--initial_peers $IP \
--logging_first_step \
--output_dir ./outputs \
--overwrite_output_dir \
--logging_dir ./logs \
--do_eval=False \
--arch=resnet50 \
--dataset_path="/data/imagenet2012" \
--warmup_ratio 0 \
--warmup_steps 0 \
--per_device_train_batch_size 32 \
--learning_rate 1.6 \
--momentum 0.9 \
--weight_decay 1e-4 \
--total_steps=450450 \
--max_steps=125000000 \
--overwrite_output_dir \
--target_batch_size 4096 \
--save_steps=10000000 \
--matchmaking_time=20 
