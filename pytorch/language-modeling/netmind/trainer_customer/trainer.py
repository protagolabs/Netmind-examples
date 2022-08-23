import transformers
import torch
import os
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
import pandas as pd
import matplotlib.pyplot as plt
from model import get_model
from data import get_data
import logging
# adv
from torch.nn.utils import clip_grad_norm_
from transformers import get_cosine_schedule_with_warmup,get_linear_schedule_with_warmup
tqdm.pandas()
from NetmindMixins.Netmind import nmp

logger = logging.getLogger(__name__)

def train(dataloader, model, optimizer, args, device):
    print('start training')

    # set up schedule if needed
    # linear warmup
    schedule_total = len(dataloader) * args.num_train_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=schedule_total
    )
    
    model.train()
    t_total = nmp.cur_step
    _loss = []
    epochs_trained = nmp.cur_epoch
    for epoch in range(epochs_trained, args.num_train_epochs):
        
        print("Local Rank: {}, Epoch: {}, Training ...".format(args.local_rank, epoch))
        for step, batch in enumerate(dataloader):
            if nmp.should_skip_step():
                continue

            t_total += 1
            # initialize calculated gradients (from prev step)
            optimizer.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device,non_blocking=True)
            attention_mask = batch['attention_mask'].to(device,non_blocking=True)
            labels = batch['labels'].to(device,non_blocking=True)
            # process
            outputs = model(input_ids=input_ids,attention_mask=attention_mask,
                            labels=labels)
            # extract loss
            loss = outputs.loss
            # calculate loss for every parameter that needs grad update
            loss.backward()

            _loss.append(loss.item())
    
            # gradient clip
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            # free might_accumulatd tensors for OOM
            del outputs, batch

            # save model
            if t_total % args.save_steps == 0 and args.local_rank == 0: 
                #logger.info('Step: {}\tLearning rate: {}\tLoss: {}\t'.format(t_total, scheduler.get_last_lr()[0], np.mean(_loss)))
                print('Step: {}\tLearning rate: {}\tLoss: {}\t'.format(t_total, scheduler.get_last_lr()[0], np.mean(_loss)))
    
            nmp.step({"loss": loss.item(), "Learning rate": scheduler.get_last_lr()[0]})
            nmp.save_pretrained_by_step(args.save_steps)
            
    # empty cache
    torch.cuda.empty_cache()