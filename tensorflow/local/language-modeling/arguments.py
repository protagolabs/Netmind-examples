import argparse
import os


def setup_args():
    """
    parameter setting
    """
    parser = argparse.ArgumentParser()
    # these basic args are needed to run the train_netmind.py code

    parser.add_argument('--model_name_or_path', default='bert-base-uncased', )
    parser.add_argument("--warmup_steps", default=5000, type=float)
    parser.add_argument('--learning_rate', default=0.0001, type=float, required=False, help='initial learning rate')

    parser.add_argument('--do_train', default=True, type=bool, required=False, help='')
    parser.add_argument('--data', default='/tf/tiny-imagenet-200', type=str, required=False, help='data directory')
    parser.add_argument("--num_train_epochs", default=6, type=int)
    parser.add_argument("--save_steps", default=100, type=int)
    parser.add_argument('--per_device_train_batch_size', default=8, type=int, required=False,
                        help='training batchsize')
    return parser.parse_known_args()[0]