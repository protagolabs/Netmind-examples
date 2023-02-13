import os
compress_dict=[
        {"hivemind/language-modeling/local" : ("trainer_customer" , "hivemind_mlm_custom_raw.tar.gz")},

        {"hivemind/language-modeling/local" : ("trainer_Huggince" ,"hivemind_mlm_trainer_raw.tar.gz")},

        {"hivemind/resnet" : ("local", "hivemind-resnet-raw.tar.gz")},

        {"pytorch/language-modeling/local" : ("trainer_customer", "torch_mlm_custom_trainer_raw.tar.gz")},

        {"pytorch/language-modeling/local" : ("trainer_Huggince", "torch_mlm_transformers_trainer_raw.tar.gz")},

        {"pytorch/resnet": ("local", "torch_resnet_custom_raw.tar.gz")},

        {"tensorflow/local": ("language-modeling" , "tf-mlm-trainer-raw.tar.gz")},

        {"tensorflow/local": ("image-classification-custom", "tf-resnet-custom-raw.tar.gz")},

        {"tensorflow/local": ("image-classification", "tf-resnet-trainer-raw.tar.gz")}
]

copy_dir = "/Users/yang.li/Desktop/example"
root_dir = os.getcwd()

for info in compress_dict:
    for k,v in info.items():
        assert len(v) == 2
        target_dir = k
        compress_dir = v[0]
        compress_file = v[1]
        command = f"cd {target_dir}" \
                  f"&& tar czvf {compress_file} {compress_dir}  " \
                  f"&& cp {compress_file} {copy_dir } " \
                  f"&& rm  {compress_file}" \
                  f"&& cd {root_dir}"
        ret = os.system(command)
        if ret != 0:
            print(f'command : {command} executed failed, ret : {ret}')
            continue
        print(f'command : {command} executed sucessfully, ret : {ret}')