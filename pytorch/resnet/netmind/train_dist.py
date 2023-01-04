from NetmindMixins.Netmind import nmp, NetmindDistributedModel, NetmindOptimizer
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



def main(args):
    assert (torch.cuda.is_available())

    model = get_model(args)
    train_dataset, val_datset = get_data(args)

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
    ddp_model = NetmindDistributedModel(
        torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    )


    # wait until data/model has loaded
    dist.barrier()

    # Prepare optimizer

    optimizer = NetmindOptimizer(get_optimizer(ddp_model,args))
    # define loss function (criterion) and optimizer
    nmp.init_train_bar(total_epoch=args.num_train_epochs, step_per_epoch=len(train_loader))
    criterion = nn.CrossEntropyLoss()
    nmp.init_eval_bar(total_epoch=args.num_train_epochs)
    cudnn.benchmark = True

    # start train
    train(train_loader, train_sampler, val_loader, model, criterion, optimizer, args, device)

    
if __name__ == '__main__':
    try:
        args = setup_args()

        best_acc1 = 0
        main(args)
    except Exception as e:
        import traceback
        traceback.print_exc()
        exit(1)

    nmp.finish_training()

