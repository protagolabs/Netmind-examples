from transformers import TrainingArguments
from dataclasses import dataclass, field
import os
import argparse

def setup_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default= 'roberta-base' , type=str, required=True, help='')
    parser.add_argument('--per_device_train_batch_size', default= 16 , type=int, required=False, help='')
    parser.add_argument('--learning_rate', default= 1e-4 , type=float, required=False, help='')
    parser.add_argument('--num_train_epochs', default= 3 , type=int, required=False, help='')
    parser.add_argument('--do_train', default=True, type=bool, required=False, help='')



    # adv
    parser.add_argument("--weight_decay", default=1e-7, type=float)
    parser.add_argument("--max_grad_norm", default=1, type=float)
    parser.add_argument("--warmup_steps", default=5000, type=float)
    parser.add_argument('--output_dir', default= 'model_1' , type=str, required=False, help='')
    parser.add_argument('--save_steps', default=5000, type=int, required=False, help='')
    parser.add_argument('--data', default=os.getenv("DATA_LOCATION"), type=str, required=False, help='')
    # distributed learning
    parser.add_argument("--local_rank",
                        type=int,
                        default=os.getenv('LOCAL_RANK', -1),
                        help="Local rank. Necessary for using the torch.distributed.launch utility")

    return parser.parse_known_args()[0]