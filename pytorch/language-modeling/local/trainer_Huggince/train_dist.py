import transformers
import torch
from tqdm import tqdm
from transformers import HfArgumentParser
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling
import matplotlib.pyplot as plt
from model import get_model
from data import get_data
from optimizer import get_optimizer
#from train import train
from trainer import train
import logging
#from arguments import setup_args
from arguments import ModelTrainingArguments
tqdm.pandas()


#logger = logging.getLogger(__name__)

def main(args):
    assert (torch.cuda.is_available())

    

    model, tokenizer = get_model(args)
    dataset = get_data(args)

    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    
    # Prepare optimizer
    optimizer = get_optimizer(model,args)    
    # start train
    train(tokenizer, data_collator, dataset, model, optimizer, args)

    
if __name__ == '__main__':
    try:
        parser = HfArgumentParser(
                ModelTrainingArguments,
        )
        args = parser.parse_args_into_dataclasses()[0]

        main(args)

    except Exception as e:
        import traceback
        traceback.print_exc()
        exit(1)
