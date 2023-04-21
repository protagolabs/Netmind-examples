import argparse
import os


def setup_args():
    """
    parameter setting
    """
    parser = argparse.ArgumentParser()
    # These basic args are defined here for netmind-ai. Changing these may lead to the conflicts
    parser.add_argument('--do_train', default=True, type=bool, required=False, help='')
    parser.add_argument('--data', default='/tf/tiny-imagenet-200', type=str, required=False, help='data directory')
    parser.add_argument("--num_train_epochs", default=20, type=int)
    parser.add_argument("--save_steps", default=20, type=int)
    parser.add_argument('--model_name_or_path', default='resnet50', )

    # the model training
    parser.add_argument('--per_device_train_batch_size', default=16, type=int, required=False, help='training batchsize')

    # the data setting
    parser.add_argument('--input_shape', default=[224, 224, 3], type=tuple, required=False, help='training input shape')

    # training setting
    parser.add_argument('--learning_rate', default=0.1, type=float, required=False, help='initial learning rate')
    parser.add_argument('--seed', default=1, type=int, required=False, help='training seed')
    parser.add_argument("--warmup_steps", default=5000, type=float)
    
    return parser.parse_known_args()[0]