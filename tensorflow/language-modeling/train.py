from transformers import AutoTokenizer, AutoConfig
import tensorflow as tf
import datasets
from transformers import create_optimizer, TFAutoModelForMaskedLM
from functools import partial
import numpy as np
import random
from datetime import datetime



class SavePretrainedCallback(tf.keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, output_dir, **kwargs):
        super().__init__()
        self.output_dir = output_dir
        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1
        # self.model.save_pretrained(self.output_dir+"/checkpoint_epoch{}".format(epoch))

    def on_train_batch_end(self, batch, logs=None):
        if batch % 10000 == 0:
            self.model.save_pretrained(self.output_dir + "/checkpoint_epoch{}_iteration{}".format(self.epoch, batch))


# region Data generator
def sample_generator(dataset, tokenizer, mlm_probability=0.15, pad_to_multiple_of=None):
    # time_start = time.time()
    if tokenizer.mask_token is None:
        raise ValueError("This tokenizer does not have a mask token which is necessary for masked language modeling. ")
    # Trim off the last partial batch if present
    sample_ordering = np.random.permutation(len(dataset))

    for sample_idx in sample_ordering:
        example = dataset[int(sample_idx)]
        # Handle dicts with proper padding and conversion to tensor.

        example = tokenizer.pad(example, return_tensors="np", pad_to_multiple_of=pad_to_multiple_of)
        special_tokens_mask = example.pop("special_tokens_mask", None)
        example["input_ids"], example["labels"] = mask_tokens(
            example["input_ids"], mlm_probability, tokenizer, special_tokens_mask=special_tokens_mask
        )
        if tokenizer.pad_token_id is not None:
            example["labels"][example["labels"] == tokenizer.pad_token_id] = -100
        example = {key: tf.convert_to_tensor(arr) for key, arr in example.items()}

        yield example, example["labels"]  # TF needs some kind of labels, even if we don't use them

    return


def mask_tokens(inputs, mlm_probability, tokenizer, special_tokens_mask):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    labels = np.copy(inputs)
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = np.random.random_sample(labels.shape)
    special_tokens_mask = special_tokens_mask.astype(np.bool_)

    probability_matrix[special_tokens_mask] = 0.0
    masked_indices = probability_matrix > (1 - mlm_probability)
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (np.random.random_sample(labels.shape) < 0.8) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (np.random.random_sample(labels.shape) < 0.5) & masked_indices & ~indices_replaced
    random_words = np.random.randint(low=0, high=len(tokenizer), size=np.count_nonzero(indices_random), dtype=np.int64)
    inputs[indices_random] = random_words

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                                      histogram_freq=1,
                                                      profile_batch=0,
                                                      update_freq=10000)

# data_args
max_seq_length = 512  
preprocessing_num_workers = 128
overwrite_cache = True
checkpoint = None
config_name = "bert-base-uncased"
tokenizer_name = "bert-base-uncased"
model_name_or_path = None  # for training from scratch

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

#### you can save/load the preprocessed data here ###


train_dataset = datasets.load_from_disk("./data_mlm")

print(train_dataset)


per_device_train_batch_size = 16
num_train_epochs = 6
learning_rate = 0.0001
warmup_proportion = 0.1

weight_decay = 1e-7
output_dir = "./bert_saved_model"

with tf.distribute.MirroredStrategy().scope():
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
    num_replicas = tf.distribute.MirroredStrategy().num_replicas_in_sync
    train_generator = partial(sample_generator, train_dataset, tokenizer)
    train_signature = {
        feature: tf.TensorSpec(shape=(None,), dtype=tf.int64)
        for feature in train_dataset.features
        if feature != "special_tokens_mask"
    }
    train_signature["labels"] = train_signature["input_ids"]
    train_signature = (train_signature, train_signature["labels"])
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    tf_train_dataset = (
        tf.data.Dataset.from_generator(train_generator, output_signature=train_signature)
            .with_options(options)
            .batch(batch_size=num_replicas * per_device_train_batch_size, drop_remainder=True)
            .repeat(int(num_train_epochs))
    )
    # endregion

    # region Optimizer and loss
    batches_per_epoch = len(train_dataset) // (num_replicas * per_device_train_batch_size)
    # Bias and layernorm weights are automatically excluded from the decay
    # optimizer, lr_schedule = create_optimizer(
    #     init_lr=learning_rate,
    #     num_train_steps=int(num_train_epochs * batches_per_epoch),
    #     num_warmup_steps=int(warmup_proportion * num_train_epochs * batches_per_epoch),
    #     adam_beta1=adam_beta1,
    #     adam_beta2=adam_beta2,
    #     adam_epsilon=adam_epsilon,
    #     weight_decay_rate=weight_decay,
    # )

    optimizer, lr_schedule = create_optimizer(
        init_lr=learning_rate,
        num_train_steps=int(num_train_epochs * batches_per_epoch),
        num_warmup_steps=int(warmup_proportion * num_train_epochs * batches_per_epoch),
        weight_decay_rate=weight_decay,
    )


    def dummy_loss(y_true, y_pred):
        return tf.reduce_mean(y_pred)


    model.compile(optimizer=optimizer, loss={"output_1": dummy_loss})
    # endregion

    # region Training

    history = model.fit(
        tf_train_dataset,
        # validation_data=tf_eval_dataset,
        epochs=int(num_train_epochs),
        steps_per_epoch=len(train_dataset) // (per_device_train_batch_size * num_replicas),
        callbacks=[SavePretrainedCallback(output_dir=output_dir), tensorboard_callback],
    )

    if output_dir is not None:
        model.save_pretrained(output_dir)