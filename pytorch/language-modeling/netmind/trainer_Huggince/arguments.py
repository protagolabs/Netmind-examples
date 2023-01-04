from transformers import TrainingArguments
from dataclasses import dataclass, field
import os

@dataclass
class ModelTrainingArguments(TrainingArguments):
    model_name_or_path: str = 'roberta-base'
    data: str = 'data/bert'
    local_rank: int = os.getenv('LOCAL_RANK', -1) # set to run the distributed training

    dataloader_num_workers: int = 1
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    seq_length: int = 512

    total_steps: int = 125_000  # set to control the total number of optimizer schedule steps
    learning_rate: float = 1e-4
    warmup_steps: int = 5000
    adam_epsilon: float = 1e-6
    weight_decay: float = 1e-7
    max_grad_norm: float = 1.0
    clamp_value: float = 10000.0

    fp16: bool = False
    fp16_opt_level: str = "O2"
    do_train: bool = True
    do_eval: bool = False

    logging_steps: int = 100
    save_total_limit: int = 2
    save_steps: int = 500

    output_dir: str = "outputs"
    max_steps: int = -1  # meant the total training steps
    #avoid log increase sharply
    disable_tqdm: bool = True
    #must set report_to to avoid wandb login
    report_to: str = "none"  # let netmind implement w&b monitor