import os
import tensorflow as tf
from arguments import setup_args
import logging


args = setup_args()

logger = logging.getLogger(__name__)



if __name__ == '__main__':

    mirrored_strategy = tf.distribute.MirroredStrategy()
    
    num_gpus = mirrored_strategy.num_replicas_in_sync

    print('Number of devices: {}'.format(num_gpus))

    global_batch_size = args.per_device_train_batch_size *  num_gpus

    #  you can use smaller data for code checking like food-101 dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        args.data,
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=args.input_shape[:2],
        batch_size=global_batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        args.data,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=args.input_shape[:2],
        batch_size=global_batch_size,
    )

    # please use these for imagenet1k
    # train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #     args.data,
    #     seed=args.seed,
    #     image_size=args.input_shape[:2],
    #     batch_size=global_batch_size,
    # )
    # val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #     args.val_data,
    #     seed=args.seed,
    #     image_size=args.input_shape[:2],
    #     batch_size=global_batch_size,
    # )

    # for x, y in train_ds.take(1):
    #     print(x.shape, y)

    train_num = len(train_ds.file_paths)
    test_num = len(val_ds.file_paths)
    category_num = len(train_ds.class_names)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

# First, we create the model and optimizer inside the strategy's scope. This ensures that any variables created with the model and optimizer are mirrored variables.

    with mirrored_strategy.scope():


        inputs = tf.keras.Input(shape=args.input_shape)

        outputs = tf.keras.applications.resnet50.ResNet50(  # Add the rest of the model
            weights=None, input_shape=args.input_shape, classes=category_num, classifier_activation="softmax"
        )(inputs)

        model = tf.keras.Model(inputs, outputs)

        model.summary()



        optimizer = tf.keras.optimizers.SGD(args.learning_rate *  num_gpus)

        model.compile(
            optimizer=optimizer,
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=tf.keras.metrics.SparseCategoricalAccuracy()
        )
    
# Next, we create the input dataset and call `tf.distribute.Strategy.experimental_distribute_dataset` to distribute the dataset based on the strategy.

    train_data_iterator = mirrored_strategy.experimental_distribute_dataset(train_ds)


    #  eval
    # dataset_eval = test_iterator().batch(global_batch_size, drop_remainder=False)
    test_data_iterator = mirrored_strategy.experimental_distribute_dataset(val_ds)

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

    history = model.fit(
        train_data_iterator,
        validation_data=test_data_iterator,
        steps_per_epoch= train_num  // global_batch_size , 
        validation_steps= test_num // global_batch_size ,
        epochs=args.num_train_epochs,
        callbacks=[model_callback,tensorboard_callback]
    )

    # #plot the training history
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.legend()
    # plt.xlabel('Epochs')
    # plt.ylabel('Mean Squared Error')
    # plt.savefig('model_training_history')
    # plt.show()

