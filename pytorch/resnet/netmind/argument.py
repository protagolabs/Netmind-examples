import os
import argparse


def setup_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', default=True, type=bool, required=False, help='')

    parser.add_argument('--data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--model_name_or_path', default='resnet18', )
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--num_train_epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--per_device_train_batch_size', default=16, type=int, required=False, help='')
    parser.add_argument('--learning_rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument("--pretrained", dest='pretrained', action='store_true', help='use pretrained model')
    # adv
    parser.add_argument("--weight_decay", default=1e-7, type=float)
    parser.add_argument('--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument("--warmup_steps", default=5000, type=float)
    parser.add_argument('--save_steps', default=5000, type=int, required=False, help='')

    # distributed learning
    parser.add_argument("--local_rank",
                        type=int,
                        default=os.getenv('LOCAL_RANK', -1),
                        help="Local rank. Necessary for using the torch.distributed.launch utility")

    return parser.parse_known_args()[0]