
## Language model training

Fine-tuning (or training from scratch) the language models on Netmind Platform for BERT, ROBERTA, ALBERT, GPT-2, etc.

There are two sets of scripts provided. The first set leverages the Trainer API. The second set with `no_trainer` in the suffix uses a custom training loop. You can easily customize them to your needs if you need extra processing on your datasets.

**Note:** Most of the code is the same as on local machine, I would highlight the samll changes in the following.

### Custormer Trainer

trainer_custom.py

```diff
from NetmindMixins.Netmind import nmp

model.train()
t_total = nmp.cur_step

+epochs_trained = nmp.cur_epoch

+if nmp.should_skip_step():
+   continue

+nmp.step({"loss": loss.item(), "Learning rate": scheduler.get_last_lr()[0]})
+nmp.save_pretrained_by_step(args.save_steps)

```

run_lm_no_trainer.py

```diff
from NetmindMixins.Netmind import nmp, NetmindDistributedModel, NetmindOptimizer

+nmp.init()

+ddp_model = NetmindDistributedModel(
+        torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
+    )

+optimizer = NetmindOptimizer(get_optimizer(ddp_model,args)) 

+nmp.init_train_bar(total_epoch=args.num_train_epochs, step_per_epoch=len(dataloader))
train(dataloader, ddp_model, optimizer, args, device)

main(args)
+nmp.finish_training()
```

### Hugginface Trainer

trainer.py:

```diff
from NetmindMixins.Netmind import nmp, NetmindTrainerCallback

class CustomTrainerCallback(NetmindTrainerCallback):
    def __init__(self):
        super().__init__()

    '''
    Add custom training metrics
    '''
    def on_step_end(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
        kwargs["custom_metrics"] = {}
        return super().on_step_end(args, state, control, **kwargs)

    '''
    Add custom evaluation metrics
    '''
    def on_evaluate(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
        kwargs["custom_metrics"] = {}
        return super().on_evaluate(args, state, control **kwargs)



 if args.do_train:

    +latest_checkpoint = nmp.last_checkpoint_from_netmind()

    trainer.train(resume_from_checkpoint=latest_checkpoint)
```


run_lm.py


```diff
from NetmindMixins.Netmind import nmp

+nmp.init(load_checkpoint=False)
train(tokenizer, data_collator, dataset, model, optimizer, args)

main(args)
+nmp.finish_training()
```
