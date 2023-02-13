from NetmindMixins.Netmind import htp
#!/usr/bin/env python3

import os
import pickle
from dataclasses import asdict
from pathlib import Path

import torch
import transformers
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch_optimizer import Lamb
from transformers import DataCollatorForLanguageModeling, HfArgumentParser, TrainingArguments, set_seed
from transformers import BertForMaskedLM, BertConfig, BertConfig, AutoTokenizer
from transformers.models.albert import AlbertConfig, AlbertForPreTraining, AlbertTokenizerFast
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer import Trainer
from transformers.trainer_utils import is_main_process
from model import get_model
from data import get_data
from optimizer import get_optimizer
from trainer import train

from tqdm import tqdm

from hivemind import DHT, Float16Compression, Optimizer, get_dht_time
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from hivemind.utils.networking import log_visible_maddrs

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

    model, tokenizer = get_model(dataset_args)
    model.to(training_args.device)

    tokenized_datasets = get_data(dataset_args)
    # This data collator will take care of randomly masking the tokens.
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    # get optimizer
    optimizer, collaborative_call, local_public_key = get_optimizer(model, training_args, collaboration_args,
                                                                    averager_args, tracker_args)

    # start training
    train(tokenized_datasets, model, tokenizer, training_args, data_collator, optimizer, collaborative_call,
          local_public_key)


if __name__ == "__main__":
    main()
    htp.on_train_end()
