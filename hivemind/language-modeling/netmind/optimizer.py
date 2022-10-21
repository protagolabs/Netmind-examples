from dataclasses import asdict
import torch
from torch.utils.data import DataLoader
from torch_optimizer import Lamb
import os
import torch
from callback import CollaborativeCallback
from transformers.optimization import get_linear_schedule_with_warmup


from hivemind import DHT, Float16Compression, Optimizer, get_dht_time
from hivemind.utils.networking import log_visible_maddrs
from NetmindMixins.Netmind import htp

import utils


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
    opt = lambda params: Lamb(
        params,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        clamp_value=training_args.clamp_value,
        debias=True,
    )

    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # linear warmup
    
    scheduler = lambda opt: get_linear_schedule_with_warmup(
        opt, num_warmup_steps=training_args.warmup_steps, num_training_steps=training_args.total_steps
    )

    optimizer = Optimizer(
        dht=dht,
        run_id=collaboration_args.experiment_prefix,
        target_batch_size=adjusted_target_batch_size,
        batch_size_per_step=total_batch_size_per_step,
        optimizer=opt,
        params=params,
        scheduler=scheduler,
        matchmaking_time=collaboration_args.matchmaking_time,
        averaging_timeout=collaboration_args.averaging_timeout,
        offload_optimizer=True,
        delay_optimizer_step=True,
        delay_grad_averaging=True,
        client_mode=collaboration_args.client_mode,
        grad_compression=Float16Compression(),
        state_averaging_compression=Float16Compression(),
        load_state_compression=Float16Compression(),
        averager_opts={"bandwidth": collaboration_args.bandwidth, **asdict(averager_args)},
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
    
    return optimizer, collaborative_call, local_public_key