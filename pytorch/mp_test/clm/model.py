from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

def get_model(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.parallelize()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    return model,tokenizer
