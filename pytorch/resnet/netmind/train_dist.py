import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from argument import setup_args
from model import get_model
from data import get_data
from optimizer import get_optimizer
from trainer import train
from NetmindMixins.Netmind import nmp, NetmindDistributedModel, NetmindOptimizer


def main(args):
    assert (torch.cuda.is_available())

    model = get_model(args)
    train_dataset, val_datset = get_data(args)

    #set up distributed backend
    torch.manual_seed(0)
    nmp.init()

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=args.per_device_train_batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    
    val_loader = DataLoader(
        val_datset,
        batch_size=args.per_device_train_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # setup device
    device = torch.device("cuda:{}".format(args.local_rank))
    # GPU
    print('setup gpu')
    model.to(device)
    # wrap the model
    model = NetmindDistributedModel(
        torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    )
    dist.barrier()

    # Prepare optimizer
    optimizer = NetmindOptimizer(get_optimizer(model,args))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    cudnn.benchmark = True
    
    nmp.init_train_bar(total_epoch=args.num_train_epochs, step_per_epoch=len(train_loader))
    nmp.init_eval_bar(total_epoch=args.num_train_epochs)

    # start train
    train(train_loader, train_sampler, val_loader, model, criterion, optimizer, args, device)

    
if __name__ == '__main__':
    args = setup_args()

    best_acc1 = 0
    main(args)
    nmp.finish_training()