import transformers
from transformers.trainer import Trainer
from transformers import ProgressCallback, DefaultFlowCallback
import torch
import os
import numpy as np
import argparse
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
import pandas as pd
import matplotlib.pyplot as plt
import logging
# adv
from torch.nn.utils import clip_grad_norm_
from transformers import get_cosine_schedule_with_warmup,get_linear_schedule_with_warmup
from pathlib import Path

logger = logging.getLogger(__name__)

class CustomTrainerCallback(transformers.TrainerCallback):
    def __init__(self):
        super().__init__()

    '''
    Add custom training metrics
    '''
    def on_step_end(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
        kwargs["custom_metrics"] = {}
        return super().on_step_end(args, state, control, **kwargs)

    '''
    Add custom evaluation metrics
    '''
    def on_evaluate(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
        kwargs["custom_metrics"] = {}
        return super().on_evaluate(args, state, control **kwargs)

def train(tokenizer, data_collator, tokenized_datasets, model, optimizer, args):
    print('start training')

    # set up schedule if needed
    # linear warmup
    schedule_total = args.total_steps

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=schedule_total
    )

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=tokenized_datasets['train'] if args.do_train else None,
        eval_dataset=tokenized_datasets['validation'] if args.do_eval else None,
        optimizers=(optimizer, scheduler),
        callbacks=[CustomTrainerCallback],
    )
    trainer.remove_callback(transformers.trainer_callback.PrinterCallback)
    trainer.remove_callback(transformers.trainer_callback.ProgressCallback)
    
    # Training
    if args.do_train:
        latest_checkpoint_dir = max(
            Path(args.output_dir).glob("checkpoint*"), default=None, key=os.path.getctime
        )

        trainer.train(resume_from_checkpoint=latest_checkpoint_dir)
