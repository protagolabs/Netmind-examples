import argparse
import os


def setup_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default='roberta-base', type=str, required=False, help='')
    parser.add_argument('--do_train', default=True, type=bool, required=False, help='')
    parser.add_argument('--data', default=os.getenv("DATA_LOCATION"), type=str, required=False, help='')
    parser.add_argument('--category_num', default=1000, type=int, required=False, help='')
    parser.add_argument('--per_device_train_batch_size', default=100, type=int, required=False, help='')
    parser.add_argument('--weight_decay', default=1e-4, type=float, required=False, help='')
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

    parser.add_argument('--input_shape', default=[224, 224, 3], type=tuple, required=False, help='training input shape')
    parser.add_argument('--initial_learning_rate', default=0.1, type=float, required=False,
                        help='initial learning rate')

    return parser.parse_args()
