#!/usr/bin/env python3

import os
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
import math

import torch
import transformers
from torch.utils.data import DataLoader
from torch_optimizer import Lamb
from transformers import DataCollatorForLanguageModeling, HfArgumentParser, TrainingArguments, set_seed
from transformers import BertForMaskedLM, BertConfig, BertConfig, AutoTokenizer
from transformers.models.albert import AlbertConfig, AlbertForPreTraining, AlbertTokenizerFast
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_utils import is_main_process

import argparse
import os
import random
import shutil
import time
import warnings
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from hivemind import DHT, Float16Compression, Optimizer, get_dht_time
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from NetmindMixins.Netmind import htp

from model import get_model
from data import get_data
from optimizer import get_optimizer
from trainer import train, validate, adjust_learning_rate

import utils
from arguments import (
    ModelTrainingArguments,
    AveragerArguments,
    CollaborationArguments,
    DatasetArguments,
    ProgressTrackerArguments,
)

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


def main():
    parser = HfArgumentParser(
        (
            ModelTrainingArguments,
            DatasetArguments,
            CollaborationArguments,
            AveragerArguments,
            ProgressTrackerArguments,
        )
    )
    training_args, dataset_args, collaboration_args, averager_args, tracker_args = parser.parse_args_into_dataclasses()
    logger.info(f"Found {len(collaboration_args.initial_peers)} initial peers: {collaboration_args.initial_peers}")

    logger.info(f"Training/evaluation parameters:\n{training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    model = get_model(training_args)
    model.cuda(device)
    

    # We need to make such a lambda function instead of just an optimizer instance
    # to make hivemind.Optimizer(..., offload_optimizer=True) work
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    cudnn.benchmark = True

    # Data loading code
    train_loader, val_loader = get_data(dataset_args,training_args)

    # define optimzier and callback
    optimizer, collaborative_call = get_optimizer(model, training_args, collaboration_args, averager_args, tracker_args)

    num_update_steps_per_epoch = len(train_loader) // training_args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    if training_args.max_steps > 0:
        htp.set_max_steps(training_args.max_steps)
    else:
        htp.set_max_steps(math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch))
    htp.set_total_train_batch_size(training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size)

    best_acc1 = 0
    acc1_lst = []
    for epoch in range(90):
        adjust_learning_rate(optimizer, epoch, training_args)     
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, collaborative_call, device)

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, device)

if __name__ == "__main__":
    main()
    htp.on_train_end()
