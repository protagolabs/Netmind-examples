from transformers import Seq2SeqTrainingArguments
from dataclasses import dataclass, field
import os
import transformers

"""
We define the model arguments here, noted that we do not setup wandb in the arguments/file, 
because it requres the key. We would allow to add wandb function later.
We need to setup do_train=True to make platform train.
"""
training_args = Seq2SeqTrainingArguments(
    f"saved_model",
    seed = 32,
    #evaluation_strategy="epoch",
    learning_rate=0.002,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    gradient_accumulation_steps=4,
    max_grad_norm=1,
    logging_steps=100,
    save_total_limit=3,
    warmup_steps=2000,
    num_train_epochs=1000,
    max_steps=10000,
    save_steps=5000,
    fp16=True,
    report_to="none",
    do_train=True,
)

"""
Load dataset.
"""
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
tokenizer = T5Tokenizer.from_pretrained("t5-small")
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
Load function requred by netmind platform.
"""
from NetmindMixins.Netmind import nmp, NetmindTrainerCallback

nmp.init()
class CustomTrainerCallback(NetmindTrainerCallback):
    def __init__(self):
        super().__init__()

    '''
    Add custom training metrics
    '''

    def on_step_end(self, args: transformers.TrainingArguments, state: transformers.TrainerState,
                    control: transformers.TrainerControl, **kwargs):
        kwargs["custom_metrics"] = {}
        return super().on_step_end(args, state, control, **kwargs)

    '''
    Add custom evaluation metrics
    '''

    def on_evaluate(self, args: transformers.TrainingArguments, state: transformers.TrainerState,
                    control: transformers.TrainerControl, **kwargs):
        kwargs["custom_metrics"] = {}
        return super().on_evaluate(args, state, control ** kwargs)
    

"""
Load trainer and start training.
"""
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data, 
    tokenizer=tokenizer,
    data_collator=data_collator, 
    optimizers=(optimizer, scheduler),
    callbacks=[CustomTrainerCallback]
)

trainer.remove_callback(transformers.trainer_callback.PrinterCallback)
trainer.remove_callback(transformers.trainer_callback.ProgressCallback)

from pathlib import Path
latest_checkpoint_dir = max(
            Path(training_args.output_dir).glob("checkpoint*"), default=None, key=os.path.getctime
        )

trainer.train(resume_from_checkpoint=latest_checkpoint_dir)
