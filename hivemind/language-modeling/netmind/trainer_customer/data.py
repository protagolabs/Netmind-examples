from tqdm import tqdm
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
import os

# adv
tqdm.pandas()


def get_data(args):
    dataset = load_from_disk(args.dataset_path + "/train")

    return dataset
