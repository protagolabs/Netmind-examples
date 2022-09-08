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
    1. install the [netmind-mixin library](https://github.com/protagolabs/NetMind-Mixin/tree/feature-tf-netmind), make sure you install from the tensorflow branch!
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

    2. modify your code and adopt to the netmind-mixin as follows (you can compare the train.py with train_netmind.py for more details):

        * add netmind class
            ![add netmind callbacks](imgs/netmind_01.jpg)

        * add environment
            ![add environment](imgs/netmind_02.jpg)

        * change the distributed strategy
            ![change the distributed strategy](imgs/netmind_03.jpg)

        * add  the netmind callbacks 
            ![change the tf dataset loading](imgs/netmind_06.jpg)
        




        








