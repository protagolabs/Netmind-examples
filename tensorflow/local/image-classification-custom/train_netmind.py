import os
import tensorflow as tf
from arguments import setup_args
import logging
from tqdm import tqdm

import json

args = setup_args()

logger = logging.getLogger(__name__)

if __name__ == '__main__':

    from tensorflow.python.client import device_lib

    logger.info(device_lib.list_local_devices())

    n_workers = 1
    if os.getenv('TF_CONFIG'):
        n_workers = len(json.loads(os.environ['TF_CONFIG']).get('cluster', {}).get('worker'))

    global_batch_size = args.per_device_train_batch_size * n_workers

    mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()

    # you can use smaller data for code checking like food-101 dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        args.data + "/train",
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=args.input_shape[:2],
        batch_size=global_batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        args.data + "/val",
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=args.input_shape[:2],
        batch_size=global_batch_size,
    )



    train_num = len(train_ds.file_paths)
    test_num = len(val_ds.file_paths)
    category_num = len(train_ds.class_names)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)


    # First, we create the model and optimizer inside the strategy's scope. This ensures that any variables created with the model and optimizer are mirrored variables.

    with mirrored_strategy.scope():

        # Set reduction to `NONE` so you can do the reduction afterwards and divide by
        # global batch size.
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(name="train_loss",
                                                                    from_logits=False,
                                                                    reduction=tf.keras.losses.Reduction.NONE)


        def compute_loss(labels, predictions):
            per_example_loss = loss_object(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)


        test_loss = tf.keras.metrics.SparseCategoricalCrossentropy(name="test_loss",
                                                                   from_logits=False)

        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy')

        inputs = tf.keras.Input(shape=args.input_shape)

        outputs = tf.keras.applications.resnet50.ResNet50(  # Add the rest of the model
            weights=None, input_shape=args.input_shape, classes=category_num, classifier_activation="softmax"
        )(inputs)

        model = tf.keras.Model(inputs, outputs)

        model.summary()

        optimizer = tf.keras.optimizers.SGD(args.learning_rate * n_workers)


    # now we define the model compile parts

    @tf.function
    def distributed_train_step(dataset_inputs):

        def train_step(inputs):
            images, labels = inputs

            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = compute_loss(labels, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_accuracy.update_state(labels, predictions)
            return loss

        per_replica_losses = mirrored_strategy.run(train_step, args=(dataset_inputs,))
        return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                        axis=None)


    @tf.function
    def distributed_test_step(dataset_inputs):

        def test_step(inputs):
            images, labels = inputs

            predictions = model(images, training=False)

            test_loss.update_state(labels, predictions)
            test_accuracy.update_state(labels, predictions)

        return mirrored_strategy.run(test_step, args=(dataset_inputs,))


    # Next, we create the input dataset and call `tf.distribute.Strategy.experimental_distribute_dataset` to distribute the dataset based on the strategy.

    train_data_iterator = mirrored_strategy.experimental_distribute_dataset(train_ds)
    #  eval
    # dataset_eval = test_iterator().batch(global_batch_size, drop_remainder=False)
    test_data_iterator = mirrored_strategy.experimental_distribute_dataset(val_ds)

    # epochs_trained = nmp.cur_epoch
    for epoch in range(args.num_train_epochs):
        # TRAIN LOOP
        total_loss = 0.0
        num_batches = 0
        for ds in tqdm(train_data_iterator):
            loss_tmp = distributed_train_step(ds)
            total_loss += loss_tmp
            num_batches += 1

            train_loss = total_loss / train_num
            # netmind relatived
            # print(f'loss : {float(train_loss.numpy())} ')
            train_monitor_metrics = {
                "loss": float(train_loss.numpy())
            }

        eval_monitor_metrics = {
            'eval loss': float(test_loss.result().numpy()),
            'eval-accuracy': float(test_accuracy.result().numpy())
        }

        # TEST LOOP
        for x in tqdm(test_data_iterator):
            distributed_test_step(x)

        template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                    "Test Accuracy: {}")
        print(template.format(epoch + 1, train_loss,
                              train_accuracy.result() * 100, test_loss.result(),
                              test_accuracy.result() * 100))

        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()
    print(f'program exited.')
