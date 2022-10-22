import argparse
import os


def setup_args():
    """
    parameter setting
    """
    parser = argparse.ArgumentParser()
    # these basic args are needed to run the train_netmind.py code
    parser.add_argument('--do_train', default=True, type=bool, required=False, help='')
    parser.add_argument('--data', default=os.getenv("DATA_LOCATION"), type=str, required=False, help='data directory')
    parser.add_argument("--num_train_epochs", default=6, type=int)
    parser.add_argument("--save_steps", default=100, type=int)

    return parser.parse_args()