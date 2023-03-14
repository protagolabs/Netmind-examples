import argparse
import os


def setup_args():
    """
    parameter setting
    """
    parser = argparse.ArgumentParser()
    # these basic args are needed to run the train_netmind.py code
    parser.add_argument('--model_name_or_path', default='resnet18', )
    parser.add_argument('--do_train', default=True, type=bool, required=False, help='')
    parser.add_argument('--data', default=os.getenv("DATA_LOCATION"), type=str, required=False, help='data directory')
    parser.add_argument("--save_steps", default=100, type=int)
    parser.add_argument("--warmup_steps", default=5000, type=float)

    # the model training
    parser.add_argument('--per_device_train_batch_size', default=64, type=int, required=False, help='training batchsize')
    parser.add_argument('--val_data', default=None,  type=str, required=False, help='val data directory')

    # the data setting
    parser.add_argument('--input_shape', default=[224, 224, 3], type=tuple, required=False, help='training input shape')

    # training setting
    parser.add_argument('--learning_rate', default=0.1, type=float, required=False, help='initial learning rate')
    parser.add_argument('--num_train_epochs', default=90, type=int, required=False, help='training epoch num')
    parser.add_argument('--seed', default=1, type=int, required=False, help='training seed')
    return parser.parse_known_args()[0]