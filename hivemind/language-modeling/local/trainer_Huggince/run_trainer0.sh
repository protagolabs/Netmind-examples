IP=/ip4/192.168.0.149/tcp/40275/p2p/QmcE3iKtBcR9JNucYTZnLuapU36w7pUXUjTNSxSKraL8eB

WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0 python run_trainer.py \
--experiment_prefix bert_experiment  \
--initial_peers $IP  \
--logging_first_step  \
--model_name_or_path='bert-base-uncased' \
--output_dir ./output_bert \
--overwrite_output_dir \
--logging_dir ./logs \
--do_eval=False \
--dataset_path="bert/data" \
--warmup_ratio 0.1 \
--warmup_steps 5000 \
--per_device_train_batch_size 16 \
--learning_rate 0.00176 \
--total_steps=125000 \
--max_steps=125000000 \
--overwrite_output_dir \
--target_batch_size 4096 \
--save_steps=10000 
