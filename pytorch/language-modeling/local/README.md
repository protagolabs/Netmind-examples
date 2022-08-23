
## Language model training

Fine-tuning (or training from scratch) the language models on local machine for BERT, ROBERTA, ALBERT, GPT-2, etc.

There are two sets of scripts provided. The first set leverages the Trainer API. The second set with `no_trainer` in the suffix uses a custom training loop. You can easily customize them to your needs if you need extra processing on your datasets.

**Note:** The sample data we use is from the parent dir called data.

### BERT/RoBERTa/DistilBERT and masked language modeling

The following example runs bert model on our sample data:


```bash
bash trainer_Huggince/run_lm.sh
```

To run on your own training loop:

```bash
bash trainer_customer/run_lm_no_trainer.sh
```

### Explanation for the code files

**arguments:** Define the parameters used in the code

**data:** Load the tokenized dataset that processed before. The instruction of how to tokenize data could be seen in [here](https://github.com/protagolabs/Netmind-examples/tree/xiangpeng/pytorch/data/process.py)

**model:** Define the model to be trained.

**optimizer:** Define the optimizer and the scheduler for training.

**trainer:** Customer defined Training loop or Huggingface Trainer.

**run_lm:**  main file for runing the language model.