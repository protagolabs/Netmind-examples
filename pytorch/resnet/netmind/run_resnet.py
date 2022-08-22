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
import torch.utils.data
import torch.utils.data.distributed
from argument import setup_args
from model import get_model
from data import get_data
from trainer import train, validate
from NetmindMixins.Netmind import nmp, NetmindDistributedModel, NetmindOptimizer, MODE_TRAIN, MODE_EVAL


def main(args):
    assert (torch.cuda.is_available())

    model = get_model(args)
    train_dataset, val_datset = get_data(args)

    #set up distributed backend
    torch.manual_seed(0)
    nmp.init()
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.per_device_train_batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    
    val_loader = torch.utils.data.DataLoader(
        val_datset,
        batch_size=args.per_device_train_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = NetmindOptimizer(
        torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    )

    cudnn.benchmark = True
    
    nmp.init_train_bar(total_epoch=args.epochs, step_per_epoch=len(train_loader))
    nmp.init_eval_bar(total_epoch=args.epochs)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    best_acc1 = 0
    checkpoint = nmp.from_pretrained("checkpoint.pth.tar")
    if checkpoint:
        state = torch.load(checkpoint)
        best_acc1 = state["best_acc1"]

    for epoch in range(nmp.cur_epoch, args.epochs):
    
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, device)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, device)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # save model
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'best_acc1': best_acc1,
        }, is_best)




def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
   
    torch.save(state, filename)
    if is_best:
        # shutil.copyfile(filename, 'model_best.pth.tar')
        nmp.save_pretrained(extra_dir_or_files=filename)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    
if __name__ == '__main__':
    args = setup_args()

    best_acc1 = 0
    main(args)
    nmp.finish_training()