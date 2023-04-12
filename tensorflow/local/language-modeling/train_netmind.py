from transformers import AutoTokenizer, AutoConfig, DataCollatorForLanguageModeling
import tensorflow as tf
import datasets
from transformers import create_optimizer, TFAutoModelForMaskedLM
import logging
from datetime import datetime
import os
import json
from arguments import setup_args

args = setup_args()

logger = logging.getLogger(__name__)


class CustomTrainerCallbackk(tf.keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, batches_per_epoch, args=args):
        super().__init__(batches_per_epoch, args)


if __name__ == '__main__':
    from tensorflow.python.client import device_lib

    logger.info(device_lib.list_local_devices())

    # data_args
    max_seq_length = 512
    preprocessing_num_workers = 128
    overwrite_cache = True
    checkpoint = None
    config_name = "bert-base-uncased"
    tokenizer_name = "bert-base-uncased"
    model_name_or_path = None  # for training from scratch

    warmup_proportion = 0.15
    mlm_probability = 0.1
    weight_decay = 1e-7
    output_dir = "./recent_saved_model"
    is_xla = False

    n_workers = 1
    if os.getenv('TF_CONFIG'):
        n_workers = len(json.loads(os.environ['TF_CONFIG']).get('cluster', {}).get('worker'))

    global_batch_size = args.per_device_train_batch_size * n_workers

    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    #### you can save/load the preprocessed data here ###

    train_dataset = datasets.load_from_disk(args.data + "/train")

    print(train_dataset)
    # region Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if checkpoint is not None:
        config = AutoConfig.from_pretrained(checkpoint)
    elif config_name:
        config = AutoConfig.from_pretrained(config_name)
    elif model_name_or_path:
        config = AutoConfig.from_pretrained(model_name_or_path)
    else:
        print("You are using unknown config.")

    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    elif model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    # endregion

    # column_names = raw_datasets.column_names
    # text_column_name = "text" if "text" in column_names else column_names[0]
    # print(text_column_name)
    if max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            print(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can reduce that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if max_seq_length > tokenizer.model_max_length:
            print(
                f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(max_seq_length, tokenizer.model_max_length)

    with strategy.scope():
        # # region Prepare model
        if checkpoint is not None:
            model = TFAutoModelForMaskedLM.from_pretrained(checkpoint, config=config)
        elif model_name_or_path:
            model = TFAutoModelForMaskedLM.from_pretrained(model_name_or_path, config=config)
        else:
            print("Training new model from scratch")
            model = TFAutoModelForMaskedLM.from_config(config)

        model.resize_token_embeddings(len(tokenizer))
        # endregion

        # region TF Dataset preparation

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm_probability=mlm_probability, return_tensors="tf"
        )
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

        tf_train_dataset = model.prepare_tf_dataset(
            train_dataset,
            shuffle=True,
            batch_size=global_batch_size,
            collate_fn=data_collator,
        ).with_options(options)

        # endregion

        # region Optimizer and loss
        num_train_steps = len(tf_train_dataset) * int(args.num_train_epochs)

        if warmup_proportion > 0:
            num_warmup_steps = int(num_train_steps * warmup_proportion)
        else:
            num_warmup_steps = 0

        # Bias and layernorm weights are automatically excluded from the decay
        # optimizer, lr_schedule = create_optimizer(
        #     init_lr=learning_rate,
        #     num_train_steps=num_train_steps,
        #     num_warmup_steps=num_warmup_steps,
        #     adam_beta1=adam_beta1,
        #     adam_beta2=adam_beta2,
        #     adam_epsilon=adam_epsilon,
        #     weight_decay_rate=weight_decay,
        #     adam_global_clipnorm=max_grad_norm,
        # )

        optimizer, lr_schedule = create_optimizer(
            init_lr=args.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            weight_decay_rate=weight_decay,
        )

        model.compile(optimizer=optimizer, jit_compile=is_xla, run_eagerly=False)

    # endregion

    logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                                          histogram_freq=1,
                                                          profile_batch=0,
                                                          update_freq=args.save_steps)

    model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(logdir, 'ckpt'),
                                                             monitor='train_loss',
                                                             save_freq="epoch",
                                                             save_best_only=False,
                                                             save_weights_only=False
                                                             )

    batches_per_epoch = len(tf_train_dataset)

    all_callbacks = [tensorboard_callback, model_callback]


    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size = {args.per_device_train_batch_size * n_workers}")

    history = model.fitf(
        tf_train_dataset,
        # validation_data=tf_eval_dataset,
        epochs=int(args.num_train_epochs),
        steps_per_epoch=len(tf_train_dataset),
        callbacks=all_callbacks,
    )

    if output_dir is not None:
        model.save_pretrained(output_dir)