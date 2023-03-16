# Train Resnet50 on Imagenet without Trainer
## prepare the imagenet dataset

* Please access the download the imagenet-ilsvrc2012 dataset from [here](https://image-net.org/index.php).

* The Training images (Task 1 & 2) is about 138GB and the  Validation images (all tasks) is about 6.3GB.

* unzip the datasets by using the following script in the download directory
```bash
bash extract_ILSVRC.sh
```

## local

* model training
```bash
python train.py --data="/home/xing/datasets/imagenet1k/train" --val_data="/home/xing/datasets/imagenet1k/val"
```

## netmind platform
* Run locally with netmind-mixin
    1. follow the [Installation](https://github.com/protagolabs/Netmind-examples/tree/main) section in main page.

    2. create the config.py file as add your ip information (you can access those information by "ifconfig") to each gpu (2 gpus):
        ```bash
        tf_config = {
            'cluster': {
                'worker' : ['192.168.1.16:30000', '192.168.1.16:30001'],
            },
            'task': {'type': 'worker'}
        }
        ```
        **"192.168.1.16" is the ip on my single machine, if you have different gpus on different machines, please use the correct ones.**

    3. modify your code as follows (you can compare the train_netmind.py in corresponding [local tasks](https://github.com/protagolabs/Netmind-examples/tree/main/tensorflow/local) and current directory for more details):

        * add netmind class
            ![add netmind callbacks](imgs/Screenshot-2022-09-12-103929.jpg)

        * add environment and change the distributed strategy
            ![add environment](imgs/Screenshot-2022-09-12-104011.jpg)

        * model init
            ![model init](imgs/Screenshot-2022-09-12-104420.jpg)

        * training loop changes
            ![training loop changes](imgs/Screenshot-2022-09-12-104918.jpg)
        




        








