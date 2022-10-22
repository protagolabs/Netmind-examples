import os
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import transformers
from transformers import TrainingArguments
from transformers.optimization import get_linear_schedule_with_warmup
import os
import torch

from hivemind import DHT, Float16Compression, Optimizer, get_dht_time
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from NetmindMixins.Netmind import HivemindTrainerCallback

import utils


use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)

class CollaborativeCallback(HivemindTrainerCallback):
    """
    This callback monitors and reports collaborative training progress.
    In case of a catastrophic failure, it can also revert training to a backup.
    """

    def __init__(
        self,
        dht: DHT,
        optimizer: Optimizer,
        model: torch.nn.Module,
        local_public_key: bytes,
        statistics_expiration: float,
        backup_every_steps: int,
    ):
        self.model = model
        self.dht, self.optimizer = dht, optimizer
        self.local_public_key = local_public_key
        self.statistics_expiration = statistics_expiration
        self.last_reported_collaboration_step = -1
        self.samples = 0
        self.steps = 0
        self.loss = 0
        self.total_samples_processed = 0
        self.backup_every_steps = backup_every_steps
        self.latest_backup = self.backup_state()
        super().__init__(dht, optimizer, local_public_key, statistics_expiration)

    def on_train_begin(
        self, args: TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs
    ):
        logger.info("Loading state from peers")
        self.optimizer.load_state_from_peers()
        return super().on_train_begin(args, state, control, **kwargs)


    def on_step_end(
        self, args: TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs
    ):
        control.should_log = True
        if not self.params_are_finite():
            self.restore_from_backup(self.latest_backup)
            return control

        local_progress = self.optimizer.local_progress

        if state.log_history:
            self.loss += state.log_history[-1]["loss"]
            self.steps += 1

            if self.optimizer.local_epoch != self.last_reported_collaboration_step:
                self.last_reported_collaboration_step = self.optimizer.local_epoch
                self.total_samples_processed += self.samples
                samples_per_second = local_progress.samples_per_second
                statistics = utils.LocalMetrics(
                    step=self.optimizer.local_epoch,
                    samples_per_second=samples_per_second,
                    samples_accumulated=self.samples,
                    loss=self.loss,
                    mini_steps=self.steps,
                )
                logger.info(f"Step #{self.optimizer.local_epoch}")
                logger.info(f"Your current contribution: {self.total_samples_processed} samples")
                logger.info(f"Performance: {samples_per_second:.3f} samples/sec")
                if self.steps:
                    logger.info(f"Local loss: {self.loss / self.steps:.5f}")
                if self.optimizer.local_epoch % self.backup_every_steps == 0:
                    self.latest_backup = self.backup_state()

                self.loss = 0
                self.steps = 0
                if self.optimizer.is_synchronized_with_peers():
                    self.dht.store(
                        key=self.optimizer.run_id + "_metrics",
                        subkey=self.local_public_key,
                        value=statistics.dict(),
                        expiration_time=get_dht_time() + self.statistics_expiration,
                        return_future=True,
                    )

        self.samples = local_progress.samples_accumulated

        return super().on_step_end(args, state, control, **kwargs)

    @torch.no_grad()
    def params_are_finite(self):
        for param in self.model.parameters():
            if not torch.all(torch.isfinite(param)):
                return False
        return True

    @torch.no_grad()
    def backup_state(self) -> bytes:
        return pickle.dumps({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()})

    @torch.no_grad()
    def restore_from_backup(self, backup: bytes):
        state = pickle.loads(backup)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])