# Train Resnet50 without Trainer
## prepare the dataset

* choose the dataset you want in "dataset" directory.

* we use the tiny-imagenet dataset in train_netmind.py


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
        




        








