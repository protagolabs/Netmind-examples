from NetmindMixins.Netmind import nmp, NetmindDistributedModel, NetmindOptimizer, NetmindDistributedModel
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
import pandas as pd
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling
import matplotlib.pyplot as plt
from model import get_model
from data import get_data
from optimizer import get_optimizer
from trainer import train
import logging
from arguments import setup_args
tqdm.pandas()


#logger = logging.getLogger(__name__)

def main(args):
    assert (torch.cuda.is_available())

    model, tokenizer = get_model(args)
    dataset = get_data(args)['train']

    #set up distributed backend
    torch.manual_seed(0)

    nmp.init()
    dateset_sampler = DistributedSampler(dataset)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    dataloader = DataLoader(
        dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_train_batch_size, pin_memory=True,sampler=dateset_sampler
    )
    # setup device
    device = torch.device("cuda:{}".format(args.local_rank))
    # GPU
    print('setup gpu')
    model.to(device)
    # wrap the model
    ddp_model = NetmindDistributedModel(
        torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    )

    
    
    # Prepare optimizer
    # start train
    optimizer = NetmindOptimizer(get_optimizer(ddp_model,args))
    nmp.init_train_bar(total_epoch=args.num_train_epochs, step_per_epoch=len(dataloader))

    train(dataloader, ddp_model, optimizer, args, device)

    
if __name__ == '__main__':
    try:
        args = setup_args()
        main(args)
    except Exception as e:
        import traceback
        traceback.print_exc()
        exit(1)
    nmp.finish_training()
