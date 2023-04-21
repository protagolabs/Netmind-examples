import os
import tensorflow as tf
from arguments import setup_args
import logging
import json

args = setup_args()

logger = logging.getLogger(__name__)


class CustomTrainerCallback(tf.keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, batches_per_epoch, args=args):
        super().__init__(batches_per_epoch, args)


if __name__ == '__main__':

    from tensorflow.python.client import device_lib

    logger.info(device_lib.list_local_devices())

    n_workers = 1
    if os.getenv('TF_CONFIG'):
        n_workers = len(json.loads(os.environ['TF_CONFIG']).get('cluster', {}).get('worker'))
    global_batch_size = args.per_device_train_batch_size * n_workers

    multi_worker_mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        args.data + "/val",
        seed=1337,
        image_size=args.input_shape[:2],
        batch_size=global_batch_size,
    )


    test_num = len(val_ds.file_paths)
    category_num = 200 # same as the number of class in training data


    # First, we create the model and optimizer inside the strategy's scope. This ensures that any variables created with the model and optimizer are mirrored variables.

    with multi_worker_mirrored_strategy.scope():

        inputs = tf.keras.Input(shape=args.input_shape)

        outputs = tf.keras.applications.resnet50.ResNet50(  # Add the rest of the model
            weights=None, input_shape=args.input_shape, classes=category_num, classifier_activation="softmax"
        )(inputs)

        model = tf.keras.Model(inputs, outputs)

        model.summary()

        model.compile(
            optimizer=tf.keras.optimizers.SGD(args.learning_rate * n_workers),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=tf.keras.metrics.SparseCategoricalAccuracy()
        )


    #  eval
    # dataset_eval = test_iterator().batch(global_batch_size, drop_remainder=False)
    test_data_iterator = multi_worker_mirrored_strategy.experimental_distribute_dataset(val_ds)

    # 
    model.load_weights("tb_logs/checkpoints") #  you may want to add the args here?

    results = model.evaluate(test_data_iterator, 
                             steps=test_num // global_batch_size,
                             batch_size=global_batch_size)
    
    print("test loss, test acc:", results)