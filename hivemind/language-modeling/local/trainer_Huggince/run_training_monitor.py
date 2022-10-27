#!/usr/bin/env python3

import time
from dataclasses import asdict, dataclass, field
from ipaddress import ip_address
from typing import Optional
import os
import shutil
from pathlib import Path
import re

import requests
import torch
import wandb
from torch_optimizer import Lamb
from transformers import BertConfig, BertForMaskedLM, HfArgumentParser
from transformers.optimization import get_linear_schedule_with_warmup

import hivemind
from hivemind.optim.state_averager import TrainingStateAverager
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from hivemind.utils.networking import log_visible_maddrs
from model import get_model


import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

import utils
from arguments import (
    AveragerArguments, 
    BaseTrainingArguments, 
    OptimizerArguments, 
    DatasetArguments, 
    ModelTrainingArguments)


use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


@dataclass
class TrainingMonitorArguments(BaseTrainingArguments):
    """
    Note: You might want to have several initial peers so that if one dies,
    new workers still can join the collaboration via alive initial peers' addresses.
    Specify initial_peers argument for that purpose
    """

    use_google_dns: bool = field(
        default=False,
        metadata={
            "help": "Use Google DNS to determine the public IP address of this machine (and add it to --announce_maddrs)"
        },
    )
    refresh_period: float = field(default=30, metadata={"help": "Period (in seconds) for fetching the keys from DHT"})
    wandb_project: Optional[str] = field(
        default=None, metadata={"help": "Name of Weights & Biases project to report the training progress to"}
    )
    save_checkpoint_step_interval: int = field(
        default=5000, metadata={"help": "Frequency (in steps) of fetching and saving state from peers"}
    )
    model_config_path: str = field(
        default="https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v2-config.json",
        metadata={"help": "Path to the model config"},
    )
    repo_path: Optional[str] = field(
        default=None, metadata={"help": "Path to local repository to store the model and optimizer states"}
    )
    repo_url: Optional[str] = field(
        default=None, metadata={"help": "URL of Hugging Face Hub repository to upload the model and optimizer states"}
    )
    upload_interval: Optional[float] = field(
        default=None, metadata={"help": "Frequency (in seconds) of uploading the model to Hub"}
    )
    store_checkpoints: bool = field(default=False, metadata={"help": "If True, enables CheckpointHandler"})


class CheckpointHandler:
    def __init__(
        self,
        dataset_args: DatasetArguments,
        training_args: ModelTrainingArguments,
        monitor_args: TrainingMonitorArguments,
        optimizer_args: OptimizerArguments,
        averager_args: AveragerArguments,
        dht: hivemind.DHT,
    ):
        self.save_checkpoint_step_interval = monitor_args.save_checkpoint_step_interval
        self.upload_interval = monitor_args.upload_interval
        self.previous_step = -1

        self.model, _ = get_model(dataset_args)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        opt = Lamb(
                optimizer_grouped_parameters,
                lr=training_args.learning_rate,
                betas=(training_args.adam_beta1, training_args.adam_beta2),
                eps=training_args.adam_epsilon,
                weight_decay=training_args.weight_decay,
                clamp_value=training_args.clamp_value,
                debias=True,
        )

        scheduler = lambda opt: get_linear_schedule_with_warmup(
            opt, num_warmup_steps=training_args.warmup_steps, num_training_steps=training_args.total_steps
        )

        self.state_averager = TrainingStateAverager(
            dht=dht,
            optimizer=opt,
            scheduler = scheduler,
            prefix=experiment_prefix,
            state_compression=hivemind.Float16Compression(),
            bandwidth=optimizer_args.bandwidth,
            client_mode=optimizer_args.client_mode,
            start=True,
            **asdict(averager_args),
        )
        self.previous_timestamp = time.time()

    def is_time_to_save_state(self, cur_step):
        if self.save_checkpoint_step_interval is None:
            return False
        elif cur_step - self.previous_step >= self.save_checkpoint_step_interval:
            return True
        else:
            return False

    def save_state(self, cur_step):
        logger.info("Saving state from peers")
        self.state_averager.load_state_from_peers()
        self.previous_step = cur_step

    def is_time_to_upload(self):
        if time.time() - self.previous_timestamp >= self.upload_interval:
            return True
        else:
            return False

    def upload_checkpoint(self):
        # Upload models to netmind
        self.previous_timestamp = time.time()


if __name__ == "__main__":
    parser = HfArgumentParser((DatasetArguments, ModelTrainingArguments, TrainingMonitorArguments, OptimizerArguments, AveragerArguments))
    dataset_args, training_args, monitor_args, optimizer_args, averager_args = parser.parse_args_into_dataclasses()

    if monitor_args.use_google_dns:
        request = requests.get("https://api.ipify.org")
        request.raise_for_status()

        address = request.text
        logger.info(f"Received public IP address of this machine: {address}")
        version = ip_address(address).version
        monitor_args.announce_maddrs += [f"/ip{version}/{address}/tcp/0"]

    experiment_prefix = monitor_args.experiment_prefix
    validators, local_public_key = utils.make_validators(experiment_prefix)

    dht = hivemind.DHT(
        start=True,
        initial_peers=monitor_args.initial_peers,
        record_validators=validators,
        use_ipfs=monitor_args.use_ipfs,
        host_maddrs=monitor_args.host_maddrs,
        announce_maddrs=monitor_args.announce_maddrs,
        identity_path=monitor_args.identity_path,
    )
    log_visible_maddrs(dht.get_visible_maddrs(), only_p2p=monitor_args.use_ipfs)

    current_step = 0
    monitor_metrics = {}
    if monitor_args.store_checkpoints:
        checkpoint_handler = CheckpointHandler(dataset_args, training_args, monitor_args, optimizer_args, averager_args, dht)
    while True:
        #nmp.update_progress_by_netmind_metris(dht)

        metrics_dict = dht.get(experiment_prefix + "_metrics", latest=True)
        if metrics_dict is not None:
            metrics_dict = metrics_dict.value
            metrics = [utils.LocalMetrics.parse_obj(metrics_dict[peer].value) for peer in metrics_dict]

            latest_step = max(item.step for item in metrics)

            if latest_step != current_step:
                logger.debug(f"Got metrics from {len(metrics)} peers")

                for i, metrics_for_peer in enumerate(metrics):
                    logger.debug(f"{i} peer {metrics_for_peer}")

                current_step = latest_step
                alive_peers = 0
                sum_loss = 0
                num_samples = 0
                sum_perf = 0
                sum_mini_steps = 0

                for item in metrics:
                    sum_loss += item.loss
                    alive_peers += 1
                    sum_perf += item.samples_per_second
                    num_samples += item.samples_accumulated
                    sum_mini_steps += item.mini_steps
                current_loss = sum_loss / sum_mini_steps
                logger.info(f"Step #{current_step}\tloss = {current_loss:.5f}")

                monitor_metrics = {
                    "loss": current_loss,
                    "alive peers": alive_peers,
                    "samples": num_samples,
                    "performance": sum_perf
                }

                if monitor_args.store_checkpoints:
                    if checkpoint_handler.is_time_to_save_state(current_step):
                        checkpoint_handler.save_state(current_step)
                        if checkpoint_handler.is_time_to_upload():
                            checkpoint_handler.upload_checkpoint()
        
        logger.debug("Peer is still alive...")
        time.sleep(monitor_args.refresh_period)
