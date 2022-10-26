import os
import argparse

def setup_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--model_name_or_path', default='resnet18', )
    parser.add_argument("--pretrained", dest='pretrained', action='store_true',help='use pretrained model')

    parser.add_argument('--do_train', default=True, type=bool, required=False, help='')
    parser.add_argument('--data', default=os.getenv("DATA_LOCATION"), type=str, required=False, help='')
    parser.add_argument('--category_num', default=1000, type=int, required=False, help='')
    parser.add_argument('--per_device_train_batch_size', default=100, type=int, required=False, help='')

    parser.add_argument("--weight_decay", default=1e-7, type=float)


    parser.add_argument('--label_smoothing', default=0.1, type=float, required=False, help='')
    parser.add_argument('--train_num', default=100, type=int, required=False, help='')
    parser.add_argument("--test_num", default=100, type=int, required=False, help='use distributed training')
    # adv
    parser.add_argument("--learning_rate", default=0.05, type=float)
    parser.add_argument("--minimum_learning_rate", default=0.0001, type=float)
    parser.add_argument("--save_steps", default=100, type=int)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--warmup_steps", default=10, type=int)

    parser.add_argument("--do_predict", default=True, type=bool)
    parser.add_argument("--per_device_eval_batch_size", default=9, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--adam_beta1", default=1.0, type=float)
    parser.add_argument("--adam_beta2", default=1.0, type=float)
    parser.add_argument("--adam_epsilon", default=10, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--max_steps", default=9, type=int)
    parser.add_argument("--warmup_ratio", default=1, type=float)
    parser.add_argument("--logging_steps", default=10, type=int)
    parser.add_argument("--fp16", default=False, type=bool)
    parser.add_argument("--train_list_path", default="data/train_test.txt", type=str)
    parser.add_argument("--test_list_path", default="data/validation_test.txt", type=str)

    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # distributed learning
    parser.add_argument("--local_rank",
                        type=int,
                        default=os.getenv('LOCAL_RANK', -1),
                        help="Local rank. Necessary for using the torch.distributed.launch utility")
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--output_dir', default='model_1', type=str, required=False, help='')

    return parser.parse_args()