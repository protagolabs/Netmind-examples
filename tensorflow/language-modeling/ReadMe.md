## install datasets and transformers from huggingface

```bash
pip install transformers==4.21.2
pip install tokenizers==0.12.1
pip install datasets==2.4.0
```

## local

* data preparing
```bash
python predata.py
```

* model training
```bash
python train.py
```

## netmind platform
* Run locally with netmind-mixin
    1. install the [netmind-mixin library](https://github.com/protagolabs/NetMind-Images/tree/feature-tf-netmind/NetmindModelEnv), make sure you install from the tensorflow branch!
    2. create the config.py file as add your ip information (you can access those information by "ifconfig") to each gpu (2 gpus):
        ```bash
        tf_config = {
            'cluster': {
                'worker' : ['192.168.1.16:30000', '192.168.1.16:30001'],
            },
            'task': {'type': 'worker'}
        }
        ```
        "192.168.1.16" is the ip on my single machine, if you have different gpus on different machines, please use the correct ones.

    2. modify your code and adopt to the netmind-mixin as follows (you can compare the train.py with train_netmind.py for more details):
        








