from pathlib import Path
import os
import torch
import transformers
from transformers.trainer import Trainer
from torch.utils.data import DataLoader
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)

LRSchedulerBase = getattr(torch.optim.lr_scheduler, "_LRScheduler", None)

class NoOpScheduler(LRSchedulerBase):
    """Dummy scheduler for transformers.Trainer. The real scheduler is defined in Optimizer.scheduler"""

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def print_lr(self, *args, **kwargs):
        if self.optimizer.scheduler:
            return self.optimizer.scheduler.print_lr(*args, **kwargs)

    def step(self):
        self._last_lr = self.get_lr()

    def state_dict(self):
        return {}

    def load_state_dict(self, *args, **kwargs):
        logger.debug("Called NoOpScheduler.load_state_dict")


def train(tokenized_datasets, model, tokenizer, training_args, data_collator, optimizer, collaborative_call, local_public_key):

    class TrainerWithIndependentShuffling(Trainer):
        def get_train_dataloader(self) -> DataLoader:
            """Shuffle data independently for each peer to avoid duplicating batches [important for quality]"""
            torch.manual_seed(hash(local_public_key))
            return super().get_train_dataloader()

    trainer = TrainerWithIndependentShuffling(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=tokenized_datasets if training_args.do_train else None,
        optimizers=(optimizer, NoOpScheduler(optimizer)),
        callbacks=[collaborative_call],
    )
    trainer.remove_callback(transformers.trainer_callback.PrinterCallback)
    trainer.remove_callback(transformers.trainer_callback.ProgressCallback)

    # Training
    if training_args.do_train:
        latest_checkpoint_dir = max(
            Path(training_args.output_dir).glob("checkpoint*"), default=None, key=os.path.getctime
        )

        trainer.train(model_path=latest_checkpoint_dir)