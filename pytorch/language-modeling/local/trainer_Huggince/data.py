from tqdm import tqdm
from datasets import load_from_disk

# adv
tqdm.pandas()

def get_data(args):

    #dataset = load_from_disk(args.data)
    dataset = load_from_disk(args.data)
    

    return dataset