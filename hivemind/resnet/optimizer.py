from dataclasses import asdict
import torch
from torch.utils.data import DataLoader
import os
import torch
from callback import CollaborativeCallback


from hivemind import DHT, Float16Compression, Optimizer, get_dht_time
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from hivemind.utils.networking import log_visible_maddrs
from NetmindMixins.Netmind import htp

import utils

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

def get_optimizer(model, training_args, collaboration_args, averager_args, tracker_args):

    validators, local_public_key = utils.make_validators(collaboration_args.experiment_prefix)

    dht = DHT(
        start=True,
        initial_peers=collaboration_args.initial_peers,
        client_mode=collaboration_args.client_mode,
        record_validators=validators,
        use_ipfs=collaboration_args.use_ipfs,
        host_maddrs=collaboration_args.host_maddrs,
        announce_maddrs=collaboration_args.announce_maddrs,
        identity_path=collaboration_args.identity_path,
        wait_timeout=10
    )
    log_visible_maddrs(dht.get_visible_maddrs(), only_p2p=collaboration_args.use_ipfs)

    total_batch_size_per_step = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    if torch.cuda.device_count() != 0:
        total_batch_size_per_step *= torch.cuda.device_count()

    adjusted_target_batch_size = collaboration_args.target_batch_size - collaboration_args.batch_size_lead

    # define your optimizer
    opt = lambda params: torch.optim.SGD(
        params,
        lr=training_args.learning_rate,
        momentum=training_args.momentum,
        weight_decay=training_args.weight_decay,
    )

    optimizer = Optimizer(
        dht=dht,
        run_id=collaboration_args.experiment_prefix,
        target_batch_size=adjusted_target_batch_size,
        batch_size_per_step=total_batch_size_per_step,
        optimizer=opt,
        params=model.parameters(),
        matchmaking_time=collaboration_args.matchmaking_time,
        averaging_timeout=collaboration_args.averaging_timeout,
        offload_optimizer=False,
        delay_optimizer_step=False,
        delay_grad_averaging=False,
        client_mode=collaboration_args.client_mode,
        grad_compression=Float16Compression(),
        state_averaging_compression=Float16Compression(),
        load_state_compression=Float16Compression(),
        averager_opts={"bandwidth": collaboration_args.bandwidth, "request_timeout": 5, **asdict(averager_args)},
        tracker_opts=asdict(tracker_args),
        verbose=True,
    )

    collaborative_call = CollaborativeCallback(
        dht,
        optimizer,
        model,
        local_public_key,
        collaboration_args.statistics_expiration,
        collaboration_args.backup_every_steps,
    )
    
    htp.init(
        dht,
        optimizer,
        local_public_key,
        collaboration_args.statistics_expiration
    )

    return optimizer, collaborative_call