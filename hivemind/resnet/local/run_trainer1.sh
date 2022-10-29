IP=/ip4/192.168.1.152/tcp/41393/p2p/Qmdqa4E7P7Dz7P4CzSRMZg9z3eaJKGdAH5VEWytXGo1ftG

WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=1 python run_trainer.py \
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
--per_device_train_batch_size 16 \
--learning_rate 0.05 \
--momentum 0.9 \
--weight_decay 1e-4 \
--total_steps=450450 \
--max_steps=125000000 \
--overwrite_output_dir \
--target_batch_size 128 \
--save_steps=10000000 \
--matchmaking_time=20 