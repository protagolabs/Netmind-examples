
## Language model training

Fine-tuning (or training from scratch) the language models on netmind for BERT, ROBERTA, ALBERT, GPT-2, etc.

**Note:** The sample data we use is from the parent dir called data.

### BERT/RoBERTa/DistilBERT and masked language modeling

The following example runs bert model on our sample data:


```bash
run_trainer.py --model_name_or_path=bert-base-uncased
```

### Explanation for the code files

**arguments:** Define the parameters used in the code, in Netmind website, it pre-defines the parameter, so we do not need this file when submit our code, just to modify the value through the website. The target_batch_size represents the total batch size to run one optimization step, could be large (like 4096), the per_device_train_batch_size represents the batch size trained on one GPU, constraints by the capacity of that GPU. Note that the learning rate should corresponding to the target batch size.

**data:** Load the tokenized dataset that processed before, named as tokenized_datasets. 

**model:** Define the model to be trained.

**optimizer:** Define the hivmind optimizer for netmind.

```diff
+define the optimzier
+this part does not need any modification if you intends to use other optimizers
def get_optimizer(model, training_args, collaboration_args, averager_args, tracker_args):

    validators, local_public_key = utils.make_validators(collaboration_args.experiment_prefix)

    dht = DHT(
        start=True,
        initial_peers=collaboration_args.initial_peers,
        client_mode=collaboration_args.client_mode,
        record_validators=validators,
        use_ipfs=collaboration_args.use_ipfs,
        host_maddrs=collaboration_args.host_maddrs,
        announce_maddrs=collaboration_args.announce_maddrs,
        identity_path=collaboration_args.identity_path,
        wait_timeout=10
    )
    log_visible_maddrs(dht.get_visible_maddrs(), only_p2p=collaboration_args.use_ipfs)

    total_batch_size_per_step = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    if torch.cuda.device_count() != 0:
        total_batch_size_per_step *= torch.cuda.device_count()

    adjusted_target_batch_size = collaboration_args.target_batch_size - collaboration_args.batch_size_lead

+define your opmizer here (the only part to be modified)
    opt = lambda params: Lamb(
        params,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        clamp_value=training_args.clamp_value,
        debias=True,
    )

    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
+Then wrap the optmizer with hivemind Optimzier and netmind api, this part of code does not need any modification

    optimizer = Optimizer(
        dht=dht,
        run_id=collaboration_args.experiment_prefix,
        target_batch_size=adjusted_target_batch_size,
        batch_size_per_step=total_batch_size_per_step,
        optimizer=opt,
        params=params,
        scheduler=scheduler,
        matchmaking_time=collaboration_args.matchmaking_time,
        averaging_timeout=collaboration_args.averaging_timeout,
        offload_optimizer=True,
        delay_optimizer_step=True,
        delay_grad_averaging=True,
        client_mode=collaboration_args.client_mode,
        grad_compression=Float16Compression(),
        state_averaging_compression=Float16Compression(),
        load_state_compression=Float16Compression(),
        averager_opts={"bandwidth": collaboration_args.bandwidth, **asdict(averager_args)},
        tracker_opts=asdict(tracker_args),
        verbose=True,
    )

    collaborative_call = CollaborativeCallback(
        dht,
        optimizer,
        model,
        local_public_key,
        collaboration_args.statistics_expiration,
        collaboration_args.backup_every_steps,
    )

    htp.init(
        dht,
        optimizer,
        local_public_key,
        collaboration_args.statistics_expiration
    )
    
    return optimizer, collaborative_call, local_public_key
```

**trainer:** Customer defined Training loop.

```diff
+define training loop here, this part of the code usually does not require any modification, unless the input of the model is slightly different.
def train(tokenized_datasets, model, tokenizer, training_args, data_collator, optimizer, collaborative_call, local_public_key):

    dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size, pin_memory=True)

    num_update_steps_per_epoch = len(dataloader) // training_args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    if training_args.max_steps > 0:
        htp.set_max_steps(training_args.max_steps)
    else:
        htp.set_max_steps(math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch))
    htp.set_total_train_batch_size(training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size)

    device = training_args.device
    model.train()
    
    for step, batch in enumerate(dataloader):
        htp.on_step_begin()

        # initialize calculated gradients (from prev step)
        optimizer.zero_grad()
        # pull all tensor batches required for training
+       input_ids = batch['input_ids'].to(device,non_blocking=True)
+       attention_mask = batch['attention_mask'].to(device,non_blocking=True)
+       labels = batch['labels'].to(device,non_blocking=True)
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
        if htp.on_step_end():
            # shutdown optimizer
            if hasattr(optimizer, "is_alive") and optimizer.is_alive():
                optimizer.shutdown()
            sys.exit(0)

            
    # empty cache
    torch.cuda.empty_cache()
```

**callback:**  callback file, does not need any modification.

**utils:**  utils file, does not need any modification.

**run_training_monitor:** monitor the training process while save the model.

```diff
+ the parameters is defined and does not need any modification
@dataclass
class TrainingMonitorArguments(BaseTrainingArguments):
    """
    Note: You might want to have several initial peers so that if one dies,
    new workers still can join the collaboration via alive initial peers' addresses.
    Specify initial_peers argument for that purpose
    """

    use_google_dns: bool = field(
        default=False,
        metadata={
            "help": "Use Google DNS to determine the public IP address of this machine (and add it to --announce_maddrs)"
        },
    )
    refresh_period: float = field(default=30, metadata={"help": "Period (in seconds) for fetching the keys from DHT"})
    wandb_project: Optional[str] = field(
        default=None, metadata={"help": "Name of Weights & Biases project to report the training progress to"}
    )
    save_checkpoint_step_interval: int = field(
        default=5000, metadata={"help": "Frequency (in steps) of fetching and saving state from peers"}
    )
    model_config_path: str = field(
        default="https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v2-config.json",
        metadata={"help": "Path to the model config"},
    )
    repo_path: Optional[str] = field(
        default=None, metadata={"help": "Path to local repository to store the model and optimizer states"}
    )
    repo_url: Optional[str] = field(
        default=None, metadata={"help": "URL of Hugging Face Hub repository to upload the model and optimizer states"}
    )
    upload_interval: Optional[float] = field(
        default=None, metadata={"help": "Frequency (in seconds) of uploading the model to Hub"}
    )
    store_checkpoints: bool = field(default=False, metadata={"help": "If True, enables CheckpointHandler"})

```

```diff
+define the checkpoint handler to store the model, the only part that needs odification is the opmizer, just to make sure it is the same optimizer you use during the training
class CheckpointHandler:
    def __init__(
        self,
        dataset_args: DatasetArguments,
        training_args: ModelTrainingArguments,
        monitor_args: TrainingMonitorArguments,
        optimizer_args: OptimizerArguments,
        averager_args: AveragerArguments,
        dht: hivemind.DHT,
    ):
        self.save_checkpoint_step_interval = monitor_args.save_checkpoint_step_interval
        self.upload_interval = monitor_args.upload_interval
        self.previous_step = -1

        self.model, _ = get_model(dataset_args)
        self.model = NetmindModel(self.model)

+this part needs to be modified if you choose your own optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        opt = NetmindOptimizer(
            Lamb(
                optimizer_grouped_parameters,
                lr=training_args.learning_rate,
                betas=(training_args.adam_beta1, training_args.adam_beta2),
                eps=training_args.adam_epsilon,
                weight_decay=training_args.weight_decay,
                clamp_value=training_args.clamp_value,
                debias=True,
            )
        )

+the rest would remain unchanged
        scheduler = lambda opt: get_linear_schedule_with_warmup(
            opt, num_warmup_steps=training_args.warmup_steps, num_training_steps=training_args.total_steps
        )

        self.state_averager = TrainingStateAverager(
            dht=dht,
            optimizer=opt,
            scheduler = scheduler,
            prefix=experiment_prefix,
            state_compression=hivemind.Float16Compression(),
            bandwidth=optimizer_args.bandwidth,
            client_mode=optimizer_args.client_mode,
            start=True,
            **asdict(averager_args),
        )
        self.previous_timestamp = time.time()

    def is_time_to_save_state(self, cur_step):
        if self.save_checkpoint_step_interval is None:
            return False
        elif cur_step - self.previous_step >= self.save_checkpoint_step_interval:
            return True
        else:
            return False

    def save_state(self, cur_step):
        logger.info("Saving state from peers")
        self.state_averager.load_state_from_peers()
        self.previous_step = cur_step

    def is_time_to_upload(self):
        if time.time() - self.previous_timestamp >= self.upload_interval:
            return True
        else:
            return False

    def upload_checkpoint(self):
        # Upload models to netmind
        hmp.save_pretrained()
        self.previous_timestamp = time.time()
```

```diff
+main function for monitor, remain unchanged
if __name__ == "__main__":
    parser = HfArgumentParser((DatasetArguments, ModelTrainingArguments, TrainingMonitorArguments, OptimizerArguments, AveragerArguments))
    dataset_args, training_args, monitor_args, optimizer_args, averager_args = parser.parse_args_into_dataclasses()

    if monitor_args.use_google_dns:
        request = requests.get("https://api.ipify.org")
        request.raise_for_status()

        address = request.text
        logger.info(f"Received public IP address of this machine: {address}")
        version = ip_address(address).version
        monitor_args.announce_maddrs += [f"/ip{version}/{address}/tcp/0"]

    experiment_prefix = monitor_args.experiment_prefix
    validators, local_public_key = utils.make_validators(experiment_prefix)

    dht = hivemind.DHT(
        start=True,
        initial_peers=monitor_args.initial_peers,
        record_validators=validators,
        use_ipfs=monitor_args.use_ipfs,
        host_maddrs=monitor_args.host_maddrs,
        announce_maddrs=monitor_args.announce_maddrs,
        identity_path=monitor_args.identity_path,
    )
    hmp.init(dht, local_public_key)
    log_visible_maddrs(dht.get_visible_maddrs(), only_p2p=monitor_args.use_ipfs)

    current_step = 0
    monitor_metrics = {}
    if monitor_args.store_checkpoints:
        checkpoint_handler = CheckpointHandler(dataset_args, training_args, monitor_args, optimizer_args, averager_args, dht)
    while True:
        #nmp.update_progress_by_netmind_metris(dht)

        metrics_dict = dht.get(experiment_prefix + "_metrics", latest=True)
        if metrics_dict is not None:
            metrics_dict = metrics_dict.value
            metrics = [utils.LocalMetrics.parse_obj(metrics_dict[peer].value) for peer in metrics_dict]

            latest_step = max(item.step for item in metrics)

            if latest_step != current_step:
                logger.debug(f"Got metrics from {len(metrics)} peers")

                for i, metrics_for_peer in enumerate(metrics):
                    logger.debug(f"{i} peer {metrics_for_peer}")

                current_step = latest_step
                alive_peers = 0
                sum_loss = 0
                num_samples = 0
                sum_perf = 0
                sum_mini_steps = 0

                for item in metrics:
                    sum_loss += item.loss
                    alive_peers += 1
                    sum_perf += item.samples_per_second
                    num_samples += item.samples_accumulated
                    sum_mini_steps += item.mini_steps
                current_loss = sum_loss / sum_mini_steps
                logger.info(f"Step #{current_step}\tloss = {current_loss:.5f}")

                monitor_metrics = {
                    "loss": current_loss,
                    "alive peers": alive_peers,
                    "samples": num_samples,
                    "performance": sum_perf
                }

                if monitor_args.store_checkpoints:
                    if checkpoint_handler.is_time_to_save_state(current_step):
                        checkpoint_handler.save_state(current_step)
                        if checkpoint_handler.is_time_to_upload():
                            checkpoint_handler.upload_checkpoint()
        
        hmp.step(current_step, monitor_metrics)
        logger.debug("Peer is still alive...")
        time.sleep(monitor_args.refresh_period)

```


**run_trainer:** main file to train the model. The only part needs to modify is data_collator, different language models may require different data_collator

```diff

+this is data_collator for mlm model (BERT)
tokenized_datasets = get_data(dataset_args)

+data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

optimizer, collaborative_call, local_public_key = get_optimizer(model, training_args, collaboration_args, averager_args, tracker_args)

train(tokenized_datasets, model, tokenizer, training_args, data_collator, optimizer, collaborative_call, local_public_key)
```