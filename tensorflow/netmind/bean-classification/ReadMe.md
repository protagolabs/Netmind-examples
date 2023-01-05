# Train Resnet50 on bean disease classfication with Trainer

## prepare the bean disease dataset

* Please access the download the bean disease dataset from [here](https://github.com/AI-Lab-Makerere/ibean/).

* prepare the datasets by using the following script in the download directory
```bash
bash extract_bean.sh
```

## local

* model training
```bash
python train.py --data="bean_data/train" --val_data="bean_data/validation"
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
        




        








