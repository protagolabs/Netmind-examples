from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM

def get_model(args):
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForMaskedLM.from_config(config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    return model,tokenizer
