"""
We define the model arguments here
"""

from dataclasses import dataclass, field
import os
import argparse
import torch

def setup_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default= 't5-small' , type=str, required=False, help='')
    parser.add_argument('--per_device_train_batch_size', default= 4 , type=int, required=False, help='')
    parser.add_argument('--learning_rate', default= 0.002 , type=float, required=False, help='')
    parser.add_argument('--num_train_epochs', default= 10000 , type=int, required=False, help='')



    # adv
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--max_grad_norm", default=1, type=float)
    parser.add_argument("--warmup_steps", default=5000, type=float)
    parser.add_argument('--output_dir', default= 'model_1' , type=str, required=False, help='')
    parser.add_argument('--save_steps', default=5000, type=int, required=False, help='')
    parser.add_argument('--max_steps', default=1000, type=int, required=False, help='')
    
    # distributed learning
    parser.add_argument("--local_rank",
                        type=int,
                        default=os.getenv('LOCAL_RANK', -1),
                        help="Local rank. Necessary for using the torch.distributed.launch utility")

    return parser.parse_known_args()[0]

training_args = setup_args()

# load data
from datasets import load_dataset
note = load_dataset("elricwan/nonDisco_notes")
trend = load_dataset("elricwan/nonDisco_trend")

"""
Load model and tokenizer.
"""
from transformers import (
    T5Tokenizer,
    T5Model,
    T5Config,T5ForConditionalGeneration,
    DataCollatorForSeq2Seq, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
)
# load customer tokenizer
tokenizer = T5Tokenizer(vocab_file = "subword.model")
# add special tokens
special_tokens_dict = {'additional_special_tokens': ['<s>']}
num_addtokens = tokenizer.add_special_tokens(special_tokens_dict)

# Step 0-2: 
model_name_or_path = 't5-small'
config = T5Config.from_pretrained(model_name_or_path)
model = T5ForConditionalGeneration(config)

"""
Prepare dataset.
"""
from torch.utils.data import Dataset
class CDNDataset(Dataset):
    def __init__(self, samples):
        super(CDNDataset, self).__init__()
        self.samples = samples

    def __getitem__(self, ite):
        res = {k_: v_[ite]for k_, v_ in self.samples.items()}
        return res

    def __len__(self):
        return len(self.samples['labels'])

train_data = tokenizer([str_.strip() for str_ in trend['train']['text']], max_length=361, padding=False,truncation=True)

train_data['labels'] = tokenizer([str_.strip() for str_ in note['train']['text']], max_length=801,
                                padding=False,truncation=True)["input_ids"]

train_data = CDNDataset(train_data)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

"""
Load optimizer.
"""
from torch_optimizer import Adafactor
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
# setup optimizer...
model.train()

print('setup optimizer...')
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': training_args.weight_decay},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
#optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)
optimizer = Adafactor(
        optimizer_grouped_parameters, lr=training_args.learning_rate,
    )

schedule_total = training_args.max_steps 

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=schedule_total
)

"""
Load custom trainer.
"""

def train(train_dataloader, training_args, model,optimizer):

    training_args.output_dir = 'saved_model'
    device = torch.device("cuda:{}".format(training_args.local_rank))
    completed_steps = 0
    epoch = nmp.cur_epoch
    for epoch in range(training_args.num_train_epochs):
        progress_bar = tqdm(range( len(train_dataloader) ))
        progress_bar.set_description(f'**Epoch: {epoch}**')

        model.train()
        total_loss = 0

        for train_step, batch in enumerate(train_dataloader):
            
            if nmp.should_skip_step():
                continue
                
            optimizer.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device,non_blocking=True)
            attention_mask = batch['attention_mask'].to(device,non_blocking=True)
            labels = batch['labels'].to(device,non_blocking=True)

            outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            
            total_loss += loss.detach().float()
            # loss = loss / self.gradient_accumulation_steps
            # accelerator.backward(loss)
            loss.backward()
            if training_args.max_grad_norm > 0:
                clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            # average loss in one epoch
            loss2log = total_loss.item()/ (train_step+1)
            lr2log  = scheduler.get_last_lr()[0]
            progress_bar.set_postfix(loss=loss2log , lr=lr2log )
            progress_bar.update(1)
            completed_steps += 1

            monitor_metrics = {
                "loss": loss.item(),
                "Learning rate": scheduler.get_last_lr()[0]
            }

            if isinstance(training_args.save_steps, int):
                if completed_steps % training_args.save_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if training_args.output_dir is not None:
                        output_dir = os.path.join(training_args.output_dir, output_dir)
                    # accelerator.save_state(output_dir)
                    model.save_pretrained(output_dir)

            nmp.step(monitor_metrics)

            if completed_steps == training_args.max_steps:
                return

    # Just for nividia-smi visiable memory release
    torch.cuda.empty_cache()


"""
Load trainer and start training with netmindminxins.
"""

from NetmindMixins.Netmind import nmp, NetmindOptimizer, NetmindDistributedModel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler



nmp.init()
dateset_sampler = DistributedSampler(train_data)
dataloader = DataLoader(
        train_data, shuffle=False, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size, pin_memory=True,
        sampler=dateset_sampler
    )


ddp_model = NetmindDistributedModel(
        torch.nn.parallel.DistributedDataParallel(model, device_ids=[training_args.local_rank], output_device=training_args.local_rank)
    )
optimizer = NetmindOptimizer(get_optimizer(ddp_model, training_args))
nmp.init_train_bar(total_epoch=training_args.num_train_epochs, step_per_epoch=len(dataloader))
train(dataloader, training_args, ddp_model, optimizer)
nmp.finish_training()

