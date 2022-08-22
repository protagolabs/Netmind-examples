IP=/ip4/192.168.1.11/tcp/38383/p2p/Qmc2gmwSnGQRy2eXWJ1ntq5SUv1HxqaUYiNbNx29euUTu5

WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0 python run_trainer.py \
--experiment_prefix bert_experiment  \
--initial_peers $IP  \
--logging_first_step  \
--config_path='bert-base-uncased' \
--output_dir ./output_bert \
--overwrite_output_dir \
--logging_dir ./logs \
--do_eval=False \
--dataset_path="/home/protago/Xiangpeng/hivemind/examples/bert/data" \
--warmup_ratio 0.1 \
--warmup_steps 5000 \
--per_device_train_batch_size 16 \
--learning_rate 0.00176 \
--total_steps=125000 \
--max_steps=125000000 \
--overwrite_output_dir \
--target_batch_size 4096 \
--save_steps=10000 
