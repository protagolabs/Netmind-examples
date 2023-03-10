# We train the mask language model based on Huggingface

## install datasets and transformers from [huggingface](https://github.com/huggingface/transformers)

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
You should see the new folder named "data_mlm" under the current directory.


* model training
```bash
python train_netmind.py --data="./data_mlm"
```






        








