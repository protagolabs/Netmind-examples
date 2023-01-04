from NetmindMixins.Netmind import  TensorflowTrainerCallback
import os
import tensorflow as tf
import config as c
from arguments import setup_args
import logging

import json
import config as c

args = setup_args()

logger = logging.getLogger(__name__)


class CustomTrainerCallback(TensorflowTrainerCallback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, batches_per_epoch, args=args):
        super().__init__(batches_per_epoch, args)

if __name__ == '__main__':

    from tensorflow.python.client import device_lib

    logger.info(device_lib.list_local_devices())
    if not os.getenv('TF_CONFIG'):
        c.tf_config['task']['index'] = int(os.getenv('INDEX'))
        os.environ['TF_CONFIG'] = json.dumps(c.tf_config)


    n_workers = len(json.loads(os.environ['TF_CONFIG']).get('cluster', {}).get('worker'))
    logger.info(f'c.tf_config : {c.tf_config}')
    global_batch_size = args.per_device_train_batch_size * n_workers

    multi_worker_mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        args.data + "/train",
        seed=args.seed,
        image_size=args.input_shape[:2],
        batch_size=global_batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        args.data + "/val",
        seed=args.seed,
        image_size=args.input_shape[:2],
        batch_size=global_batch_size,
    )

    # for x, y in train_ds.take(1):
    #     print(x.shape, y)

    train_num = len(train_ds.file_paths)
    test_num = len(val_ds.file_paths)
    category_num = len(train_ds.class_names)

 
    #train_ds = train_ds.cache().repeat().prefetch(tf.data.AUTOTUNE)
    train_ds = train_ds.repeat().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache()

# First, we create the model and optimizer inside the strategy's scope. This ensures that any variables created with the model and optimizer are mirrored variables.

    with multi_worker_mirrored_strategy.scope():


        inputs = tf.keras.Input(shape=args.input_shape)

        outputs = tf.keras.applications.resnet50.ResNet50(  # Add the rest of the model
            weights=None, input_shape=args.input_shape, classes=category_num, classifier_activation="softmax"
        )(inputs)

        model = tf.keras.Model(inputs, outputs)

        model.summary()

        model.compile(
            optimizer=tf.keras.optimizers.SGD(args.learning_rate *  n_workers),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=tf.keras.metrics.SparseCategoricalAccuracy()
        )


    
# Next, we create the input dataset and call `tf.distribute.Strategy.experimental_distribute_dataset` to distribute the dataset based on the strategy.

    train_data_iterator = multi_worker_mirrored_strategy.experimental_distribute_dataset(train_ds)


    #  eval
    # dataset_eval = test_iterator().batch(global_batch_size, drop_remainder=False)
    test_data_iterator = multi_worker_mirrored_strategy.experimental_distribute_dataset(val_ds)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir="tb_logs/snap",
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        write_steps_per_second=False,
        update_freq="epoch",
        profile_batch=0,
        embeddings_freq=0,
        embeddings_metadata=None,
    )

    model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='tb_logs/checkpoints/', 
        monitor='evaluation_categorical_accuracy_vs_iterations',
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        save_freq="epoch",
    )

    batches_per_epoch = (train_num // global_batch_size)

    all_callbacks = [tensorboard_callback, model_callback]

    netmind_callback = CustomTrainerCallback(batches_per_epoch=batches_per_epoch)
    history = model.fit(
        train_data_iterator,
        validation_data=test_data_iterator if hasattr(args, "do_eval") and args.do_eval else None,
        steps_per_epoch= train_num  // global_batch_size , 
        validation_steps= test_num // global_batch_size if hasattr(args, "do_eval") and args.do_eval else None,
        epochs=args.num_train_epochs,
          callbacks = all_callbacks + [netmind_callback]
    )