import transformers
import torch
import os
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
import pandas as pd
from transformers import AdamW
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling
import matplotlib.pyplot as plt
from model import get_model
from data import get_data
import logging
# adv
from torch.nn.utils import clip_grad_norm_
from transformers import get_cosine_schedule_with_warmup,get_linear_schedule_with_warmup
tqdm.pandas()

logger = logging.getLogger(__name__)

import transformers
import torch
import os
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
import pandas as pd
from transformers import AdamW
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling
import matplotlib.pyplot as plt
from model import get_model
from data import get_data
import logging
# adv
from torch.nn.utils import clip_grad_norm_
from transformers import get_cosine_schedule_with_warmup,get_linear_schedule_with_warmup
tqdm.pandas()

logger = logging.getLogger(__name__)

def train(dataloader, model, optimizer, args, device):
    print('start training')


    # set up schedule if needed
    # linear warmup
    schedule_total = len(dataloader) * args.num_train_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=schedule_total
    )
    
    model.train()
    t_total = 0
    _loss = []
    for epoch in range(args.num_train_epochs):
        
        print("Local Rank: {}, Epoch: {}, Training ...".format(args.local_rank, epoch))
        # setup loop with TQDM and dataloader
        for batch in tqdm(dataloader):

            t_total += 1
            # initialize calculated gradients (from prev step)
            optimizer.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device,non_blocking=True)
            attention_mask = batch['attention_mask'].to(device,non_blocking=True)
            labels = batch['labels'].to(device,non_blocking=True)
            # process
            outputs = model(input_ids=input_ids,attention_mask=attention_mask,
                            labels=labels)
            # extract loss
            loss = outputs.loss
            # calculate loss for every parameter that needs grad update
            loss.backward()

            _loss.append(loss.item())
    
            # gradient clip
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            # free might_accumulatd tensors for OOM
            del outputs, batch

            # save model
            if t_total % args.save_steps == 0 and args.local_rank == 0: 
                #logger.info('Step: {}\tLearning rate: {}\tLoss: {}\t'.format(t_total, scheduler.get_last_lr()[0], np.mean(_loss)))
                print('Step: {}\tLearning rate: {}\tLoss: {}\t'.format(t_total, scheduler.get_last_lr()[0], np.mean(_loss)))
                model_save_path = './{}/model_step_{}'.format(args.output_dir,str(t_total))
                if not os.path.exists(model_save_path):
                    os.mkdir(model_save_path)
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(model_save_path)
                logger.info("Saving model checkpoint to %s", model_save_path)

            
    # empty cache
    torch.cuda.empty_cache()