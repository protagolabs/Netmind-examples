from pathlib import Path
import sys
import math
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import transformers
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from hivemind.optim.optimizer import Optimizer
from tqdm import tqdm
tqdm.pandas()

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)

def train(tokenized_datasets, model, tokenizer, training_args, data_collator, optimizer, collaborative_call, local_public_key):

    dataloader = DataLoader(tokenized_datasets, shuffle=True, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size, pin_memory=True)

    num_update_steps_per_epoch = len(dataloader) // training_args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

    device = training_args.device
    model.train()
    
    for step, batch in enumerate(dataloader):

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

        # gradient clip
        clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
        optimizer.step()
        # free might_accumulatd tensors for OOM
        del outputs, batch

        # at the end of the step: on_step_end
        collaborative_call.on_step_end(loss=loss.item())

            
    # empty cache
    torch.cuda.empty_cache()